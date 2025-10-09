"""
This script is adapted from torchtune's `lora_finetune_distributed.py` recipe and implements multi-stage finetuning. Pass datasets via config and specify their order through `phase` field to get started.
"""

import sys
import time

from functools import partial
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group

from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    DoRALinear,
    get_adapter_params,
    get_adapter_state_dict,
    get_lora_module_names,
    get_merged_lora_ckpt,
    LoRALinear,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from tqdm import tqdm

log = utils.get_logger("DEBUG")

STAGE_KEY = "stage_idx"
EPOCHS_IN_STAGE_KEY = "epochs_run_in_stage"


class LoRAFinetuneRecipeDistributed(FTRecipeInterface):

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.global_step = 0
        self.total_epochs_run = 0 # Tracks total epochs across all stages

        # State for resuming
        self.resume_from_stage = 0
        self.resume_from_epoch = 0

        # Process dataset configs for staged training
        if not isinstance(cfg.dataset, ListConfig):
            raise ValueError("For multi-stage training, the 'dataset' config must be a list.")
        for d_cfg in cfg.dataset:
            if "phase" not in d_cfg or "epochs" not in d_cfg:
                raise ValueError("Each dataset config must have 'phase' and 'epochs' keys for staged training.")
        self._dataset_configs = sorted(cfg.dataset, key=lambda x: x.phase)
        self.total_stages = len(self._dataset_configs)

        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._save_every_n_epochs = cfg.get("save_every_n_epochs", 1)
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(
                log,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state for multi-stage training.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            should_load_recipe_state=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint for multi-stage training.
        """
        try:
            self.resume_from_stage = ckpt_dict[STAGE_KEY]
            self.resume_from_epoch = ckpt_dict[EPOCHS_IN_STAGE_KEY]
            log.info(f"Resuming training from stage {self.resume_from_stage}, epoch {self.resume_from_epoch}")

            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                f"Missing key: {e}. Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe's shared components. This includes model, tokenizer, and profiler.
        Stage-specific components like dataloader, optimizer, and scheduler are
        handled by `_setup_stage`.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        self._compile = cfg.get("compile", False)
        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._loss_fn = config.instantiate(cfg.loss)
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        if self._loss_fn.__class__.__name__ in ["CEWithChunkedOutputLoss", "NegCEWithChunkedOutputLoss"]:
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)
        utils.log_rank_zero(log, "Shared components (Model, Tokenizer, Loss) are initialized.")

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

        # If resuming, calculate the global step and total epochs run up to the resume point
        if self._resume_from_checkpoint:
            raise NotImplementedError("Resuming from checkpoint is not supported for staged training.")

    def _setup_stage(self, stage_idx: int, cfg: DictConfig) -> None:
        """
        Sets up the components for a specific training stage. This includes
        the dataset, dataloader, optimizer, and LR scheduler.
        """
        utils.log_rank_zero(log, f"--- Setting up for Stage {stage_idx + 1}/{self.total_stages} ---")
        stage_cfg = self._dataset_configs[stage_idx]

        # 1. Setup Data
        self._dataloader = self._setup_data(
            cfg_dataset=stage_cfg,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=cfg.get("collate_fn", "torchtune.data.padded_collate_sft"),
        )
        self._steps_per_epoch = len(self._dataloader) // self._gradient_accumulation_steps
        self.epochs_in_stage = stage_cfg.epochs
        utils.log_rank_zero(log, f"Stage {stage_idx + 1} will run for {self.epochs_in_stage} epochs with {self._steps_per_epoch} steps per epoch.")

        # 2. Setup Optimizer (refreshed for each stage)
        self._optimizer = self._setup_optimizer(cfg_optimizer=cfg.optimizer)

        # 3. Setup LR Scheduler (refreshed for each stage)
        num_training_steps_in_stage = self.epochs_in_stage * self._steps_per_epoch
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=num_training_steps_in_stage,
            last_epoch=-1, # Reset scheduler for each stage
        )

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
        """
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        utils.log_rank_zero(
            log, f" Profiler config after instantiation: {profiler_cfg}"
        )
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler


    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        enable_activation_offloading: bool,
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        base_model_state_dict: Dict[str, Any],
        custom_sharded_layers: Optional[List[str]] = None,
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)

        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        set_trainable_params(model, get_adapter_params(model))

        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # For FSDP sharding
        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]
        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
        )

        if lora_weights_state_dict:
            lora_missing, lora_unexpected = training.load_from_full_model_state_dict(
                model,
                lora_weights_state_dict,
                self._device,
                cpu_offload=fsdp_cpu_offload,
            )
        else:
            lora_missing, lora_unexpected = None, None

        # Initialize LoRA params and RoPE buffers
        with training.set_default_dtype(self._dtype), self._device:
            lora_device = "cpu" if fsdp_cpu_offload else self._device
            for m in model.modules():
                if (
                    isinstance(m, LoRALinear) or isinstance(m, DoRALinear)
                ) and not lora_weights_state_dict:
                    # lora may not be covered in state dict
                    # if finetune for the 1st time
                    m.to_empty(device=lora_device)
                    m.initialize_parameters()

                if hasattr(m, "rope_init"):
                    m.rope_init()

        base_missing, base_unexpected = training.load_from_full_model_state_dict(
            model,
            base_model_state_dict,
            self._device,
            cpu_offload=fsdp_cpu_offload,
        )
        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"):
                m.initialize_dora_magnitude()

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        # activation offloading
        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        # log
        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        torch.distributed.barrier()

        return model


    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )

        utils.log_rank_zero(log, "Optimizer is initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        utils.log_rank_zero(log, "Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
    ) -> StatefulDataLoader:
        """
        Sets up the dataloader for a single training stage.
        """
        cfg_simplified = cfg_dataset.copy()
        cfg_simplified.pop("epochs")
        cfg_simplified.pop("role")
        cfg_simplified.pop("phase")
        ds = config.instantiate(cfg_simplified, self._tokenizer)
        packed = cfg_dataset.get("packed", False)

        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                    ignore_idx=self._loss_fn.ignore_index,
                )
                if not packed
                else padded_collate_packed
            ),
            drop_last=True,
        )

        return dataloader
    
    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        checkpoint_dict = {}

        # TODO: should we enable intermediate checkpoint?
        intermediate_checkpoint = False

        utils.log_rank_zero(
            log,
            "Saving checkpoint. This may take some time. Retrieving full model state dict...",
        )
        start = time.perf_counter()

        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        cpu_state_dict = training.gather_cpu_state_dict(
            self._model,
            self._is_rank_zero,
            device=self._device,
            adapter_weights_only=self._save_adapter_weights_only,
        )
        utils.log_rank_zero(
            log,
            f"Getting full model state dict took {time.perf_counter() - start:.2f} secs",
        )

        if intermediate_checkpoint:
            utils.log_rank_zero(log, "Retrieving optimizer state dict...")
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._model,
                self._optimizer,
                self._is_rank_zero,
                device=self._device,
            )
            utils.log_rank_zero(
                log,
                f"Getting optimizer state dict took {time.perf_counter() - start:.2f} secs",
            )
        else:
            opt_state_dict = None

        if self._is_rank_zero:
            start = time.perf_counter()

            if self._save_adapter_weights_only:
                adapter_state_dict = cpu_state_dict
            else:
                adapter_state_dict = get_adapter_state_dict(cpu_state_dict)

                # merge the adapter weights and base weights to create the model checkpoint
                merged_state_dict = get_merged_lora_ckpt(
                    cpu_state_dict,
                    rank=self._lora_rank,
                    alpha=self._lora_alpha,
                )
                checkpoint_dict.update({training.MODEL_KEY: merged_state_dict})
            checkpoint_dict.update({training.ADAPTER_KEY: adapter_state_dict})

            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self.epochs_run,
                        training.TOTAL_EPOCHS_KEY: self.total_epochs,
                        training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                        training.DATALOADER_KEY: self._dataloader.state_dict(),
                    }
                )

            adapter_config = {
                "r": self._lora_rank,
                "lora_alpha": self._lora_alpha,
                "target_modules": get_lora_module_names(
                    self._lora_attn_modules,
                    self._apply_lora_to_mlp,
                    self._apply_lora_to_output,
                ),
                "peft_type": "LORA",
            }
            checkpoint_dict.update({training.ADAPTER_CONFIG: adapter_config})
            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
                adapter_only=self._save_adapter_weights_only,
            )
            log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

        torch.distributed.barrier()

    def train(self) -> None:
        """
        The core training loop, now with support for multiple stages.
        """
        training.cleanup_before_training()
        self._profiler.start()

        # outer loop: over stages
        for stage_idx in range(self.resume_from_stage, self.total_stages):
            self._setup_stage(stage_idx, self.cfg)

            start_epoch = self.resume_from_epoch if stage_idx == self.resume_from_stage else 0
            # refresh optimizer and lr scheduler
            self._optimizer.zero_grad()
            
            for curr_epoch in range(start_epoch, self.epochs_in_stage):
                pbar_desc = f"Stage {stage_idx + 1}/{self.total_stages} | Epoch {curr_epoch + 1}/{self.epochs_in_stage}"
                pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero, desc=pbar_desc)
                self._dataloader.sampler.set_epoch(self.total_epochs_run + curr_epoch)

                t0 = time.perf_counter()
                running_loss = 0
                num_tokens = 0

                for idx, batch in enumerate(self._dataloader):
                    if (
                        self._is_rank_zero
                        and curr_epoch == 0
                        and self.profiler_profile_memory
                        and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                        and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history()

                    utils.batch_to_device(batch, self._device)
                    current_num_tokens = (batch["labels"] != self._loss_fn.ignore_index).sum()
                    num_tokens += current_num_tokens

                    labels = batch.pop("labels")

                    with self.activations_handling_ctx:
                        logits = self._model(**batch)

                    labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]]))
                    if not isinstance(logits, list):
                        labels = labels.reshape(-1)
                        logits = logits.reshape(-1, logits.size(-1))

                    current_loss = self._loss_fn(logits, labels) * current_num_tokens
                    del logits
                    running_loss += current_loss
                    current_loss.backward()

                    if (idx + 1) % self._gradient_accumulation_steps == 0:
                        torch.distributed.all_reduce(num_tokens)
                        torch.distributed.all_reduce(running_loss)
                        training.scale_grads(self._model, self.world_size / num_tokens)
                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            ).full_tensor()
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        self._lr_scheduler.step()

                        self.global_step += 1
                        loss_to_log = running_loss.item() / num_tokens
                        pbar.update(1)
                        pbar.set_description(f"{pbar_desc} | Step {self.global_step} | Loss: {loss_to_log:.3f}")

                        if self.global_step % self._log_every_n_steps == 0 and self._is_rank_zero:
                            time_per_step = time.perf_counter() - t0
                            log_dict = {
                                "loss": loss_to_log,
                                "lr": self._optimizer.param_groups[0]["lr"],
                                "tokens_per_second_per_gpu": num_tokens / (time_per_step * self.world_size),
                                "stage": stage_idx + 1,
                                "epoch_in_stage": curr_epoch + 1,
                            }
                            if self._log_peak_memory_stats:
                                log_dict.update(training.get_memory_stats(device=self._device))
                            if self._clip_grad_norm is not None:
                                log_dict.update({"grad_norm": grad_norm})
                            self._metric_logger.log_dict(log_dict, step=self.global_step)

                        running_loss = 0
                        num_tokens = 0
                        t0 = time.perf_counter()

                        if (
                            self._is_rank_zero
                            and curr_epoch == 0
                            and self.profiler_profile_memory
                            and idx
                            == self.profiler_wait_steps
                            + self.profiler_warmup_steps
                            + self.profiler_active_steps
                            and self._device.type == "cuda"
                        ):
                            torch.cuda.memory._record_memory_history(enabled=None)

                        self._profiler.step()

            self.total_epochs_run += self.epochs_in_stage
        
            utils.log_rank_zero(log, f"Completed Stage {stage_idx + 1}. Saving final stage checkpoint.")
        
        self.save_checkpoint(epoch=0)
        self._profiler.stop()


    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    init_process_group("cuda:nccl,cpu:gloo")
    if cfg.get("fsdp_cpu_offload", False):
        training.set_torch_num_threads()

    config.log_config(recipe_name="LoRAFinetuneRecipeDistributed", cfg=cfg)

    recipe = LoRAFinetuneRecipeDistributed(cfg=cfg)
    recipe.cfg = cfg 
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())