"""
This script is adapted from torchtune's `lora_finetune_distributed.py` recipe and implements gradient ascent, NPO, and SimNPO trainers for LLM unlearning.
We implemented custom loss functions for NPO and SimNPO. To run these methods, you will need to manually add them into `torchtune.modules`. 
We also support tensor parallelism, but we don't shard K and V projections since they are sensitive to model architecture.

When running in offline GPUs, please make sure to set `WANDB_MODE="offline"` in the environment if you use wandb logger.
"""

import sys
import time
import json
import re
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Union, Tuple
from warnings import warn
from copy import deepcopy
import torch
import torch.nn.functional as F
import os
from omegaconf import DictConfig, ListConfig
import itertools
from torch import nn
from torch.distributed import destroy_process_group, init_process_group

# for 2-d tensor parallelism
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module, ColwiseParallel, RowwiseParallel
)

from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, modules, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import Message, padded_collate_packed, padded_collate_sft, CROSS_ENTROPY_IGNORE_IDX, left_pad_sequence
from torchtune.datasets import ConcatDataset
from torchtune.modules import local_kv_cache
from torchtune.modules.peft import (
    DoRALinear,
    get_adapter_params,
    get_adapter_state_dict,
    get_lora_module_names,
    get_merged_lora_ckpt,
    LoRALinear,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
    disable_adapter,
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from tqdm import tqdm

from torchtune.generation import generate
from math import fsum  

UNLEARN_LABEL = "Sorry, I can not assist"

log = utils.get_logger("DEBUG")

class JSONDataset(Dataset):
    def __init__(self, path: str, subsample: int = 1):
        with open(path, 'r') as f:
            self.data = json.load(f)
            # self.data = self.data[:subsample]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def convert_to_message(input_text: str) -> List[Message]:
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=input_text),
        Message(role="assistant", content="")
    ]

def logits_to_logprobs(
    logits: torch.Tensor, sequences: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    return torch.gather(
        F.log_softmax(logits / temperature, dim=-1),
        2,
        sequences.unsqueeze(-1),
    ).squeeze(-1)

def get_batch_log_probs(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = CROSS_ENTROPY_IGNORE_IDX,
    return_average_logprobs: bool = False,
) -> torch.FloatTensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0
    per_token_log_probs = logits_to_logprobs(logits, labels, temperature=1.0)

    if return_average_logprobs:
        sum_log_probs = (per_token_log_probs * loss_mask).sum(-1)
        num_tokens = loss_mask.sum(-1)
        return sum_log_probs / num_tokens.clamp(min=1)
    else:
        return (per_token_log_probs * loss_mask).sum(-1)

class LoRAFinetuneRecipeDistributed(FTRecipeInterface):

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        self.world_size, self.rank = utils.get_world_size_and_rank()

        self._tp_size = int(cfg.get("tensor_parallel_dim", 1))
        assert self.world_size % self._tp_size == 0, "world_size must be divisible by tensor_parallel_dim"
        self._dp_size = self.world_size // self._tp_size
        self._tp_rank = self.rank % self._tp_size
        self._dp_rank = self.rank // self._tp_size

        # Build a 2-D device mesh: (dp, tp)
        # Ranks are laid out row-major; slice by name to get a 1-D mesh per dim.
        self._world_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(self._dp_size, self._tp_size),
            mesh_dim_names=("dp", "tp"),
        )
        self._tp_mesh = self._world_mesh["tp"]

        # Process groups for the named mesh dims
        try:
            self._tp_pg = self._tp_mesh.get_group()
            self._dp_pg = self._world_mesh["dp"].get_group()
        except AttributeError:
            # Fallback for older PT versions: build groups from mesh ranks
            tp_ranks = list(self._tp_mesh.mesh.flatten())
            dp_ranks = list(self._world_mesh["dp"].mesh.flatten())
            self._tp_pg = torch.distributed.new_group(ranks=tp_ranks)
            self._dp_pg = torch.distributed.new_group(ranks=dp_ranks)



        self._is_rank_zero = self.rank == 0

        self._max_generated_tokens = cfg.max_new_tokens
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        
        self._evaluation_enabled = "evaluation" in cfg

        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._save_every_n_epochs = cfg.get("save_every_n_epochs", 1)

        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda":
                raise RuntimeError("enable_activation_offloading should only be True when training on CUDA")
            if not self._enable_activation_checkpointing:
                raise RuntimeError("enable_activation_offloading should only be True when enable_activation_checkpointing is True")
        elif (
            self._enable_activation_checkpointing
            and cfg.checkpointer.model_type != "LLAMA3_VISION"
        ):
            utils.log_rank_zero(log, "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. Enabling activation offloading should reduce memory further.")

        self._eval_refusal_batch_size: Optional[int] = None
        self._refusal_sentences: Optional[List[str]] = None

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        self._checkpointer = config.instantiate(cfg_checkpointer, should_load_recipe_state=self._resume_from_checkpoint)
        checkpoint_dict = self._checkpointer.load_checkpoint()
        if self._resume_from_checkpoint:
            if training.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError("Adapter weights not found. Please ensure a valid adapter checkpoint is provided.")
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(f"Config value for seed does not match the checkpoint value, using the checkpoint value: {ckpt_dict[training.SEED_KEY]}")
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(f"Config value for max_steps_per_epoch does not match the checkpoint value, using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}")
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(f"Config value for total_epochs does not match the checkpoint value, using the config value: {self.total_epochs}")
        except KeyError as e:
            raise KeyError("Checkpoint does not contain the required keys for updating recipe state.") from e

    def setup(self, cfg: DictConfig) -> None:
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)
        self._compile = cfg.get("compile", False)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(checkpoint_dict.get(training.ADAPTER_KEY) if self._resume_from_checkpoint else None),
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(checkpoint_dict.get(training.OPT_KEY) if self._resume_from_checkpoint else None),
        )
        
        # setup loss function
        self._loss_fn = config.instantiate(cfg.loss)
        self._loss_type = cfg.get("loss_type", None)
        if self._loss_type == "ga":
            self._step_fn = self._step_ga
        elif self._loss_type == "npo":
            self._step_fn = self._step_npo
        elif self._loss_type == "simnpo":
            self._step_fn = self._step_simnpo
        else:
            raise ValueError(f"Loss type {self._loss_type} not implemented.")

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)
        if hasattr(self._loss_fn, "num_output_chunks"):
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)
        utils.log_rank_zero(log, "Loss is initialized.")

        collate_name = cfg.get("collate_fn", "torchtune.data.padded_collate_sft")
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
            dataloader_state_dict=(checkpoint_dict.get(training.DATALOADER_KEY) if self._resume_from_checkpoint else None),
        )
        total_batches = len(self._dataloader)
        self._steps_per_epoch = total_batches // self._gradient_accumulation_steps
        if self.max_steps_per_epoch is not None and self.max_steps_per_epoch < self._steps_per_epoch:
            self._steps_per_epoch = self.max_steps_per_epoch
        self.global_step = self.epochs_run * self._steps_per_epoch

        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))
        self.ignore_labels_cache = torch.full((cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device)

        # Setup evaluation dataloaders
        self._eval_dataloaders = {}
        if self._evaluation_enabled:
            eval_cfg = cfg.evaluation
            eval_batch_size = eval_cfg.get("batch_size", 4)
            log.info(f"Setting up distributed evaluation dataloaders with batch size {eval_batch_size}...")
            self._eval_refusal_batch_size = eval_batch_size

            for name, path in eval_cfg.items():
                if name == "batch_size": continue
                if path and isinstance(path, str):
                    ds = JSONDataset(path)
                    sampler = StatefulDistributedSampler(
                                ds,
                                num_replicas=self._dp_size,
                                rank=self._dp_rank,
                                shuffle=False,
                            )
                    self._eval_dataloaders[name] = StatefulDataLoader(
                        ds,
                        batch_size=eval_batch_size,
                        sampler=sampler,
                    )
            utils.log_rank_zero(log, f"Evaluation dataloaders created: {list(self._eval_dataloaders.keys())}")

            # Load semantic refusal sentences TODO: suppress this in config.
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                script_dir = os.getcwd()
                
            refusal_file = os.path.join(script_dir, "semantic_refusal_sentences.txt")
            if not os.path.exists(refusal_file):
                raise FileNotFoundError(
                    f"semantic_refusal_sentences.txt not found at path: {refusal_file}"
                )
            with open(refusal_file, "r", encoding="utf-8") as f:
                self._refusal_sentences = [line.strip() for line in f.readlines() if line.strip()]
    
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

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        if self._tp_size > 1:
            tp_plan = {}
            for name, mod in model.named_modules():
                if not isinstance(mod, nn.Linear):
                    continue

                # Attention projections:
                # - Q: shard by heads
                if name.endswith(("q_proj",)):
                    tp_plan[name] = ColwiseParallel()

                # MLP:
                elif name.endswith(("up_proj", "gate_proj")):
                    tp_plan[name] = ColwiseParallel()
                elif name.endswith(("down_proj",)):
                    tp_plan[name] = RowwiseParallel()

                # Output projections:
                elif name.endswith(("o_proj", "lm_head")):
                    tp_plan[name] = RowwiseParallel()

            parallelize_module(model, self._tp_mesh, tp_plan)


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
            dp_mesh=self._world_mesh["dp"],   # <-- ONLY DP
        )
        if self._compile:
            training.compile_model(model, verbose=self._is_rank_zero)

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

        training.validate_no_params_on_meta_device(model)
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

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
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
    
    def _setup_optimizer(self, cfg_optimizer, opt_state_dict=None):
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(self._model, optimizer, opt_state_dict, self._device)
        return optimizer

    def _setup_lr_scheduler(self, cfg_lr_scheduler, num_training_steps, last_epoch):
        return config.instantiate(cfg_lr_scheduler, self._optimizer, num_training_steps=num_training_steps, last_epoch=last_epoch)

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[Dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. This recipe currently supports only
        map-style datasets. If a state_dict is provided (meaning we are resuming a training run),
        it is loaded into the dataloader.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
            packed = getattr(ds, "packed", False)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)
            packed = cfg_dataset.get("packed", False)

        # Instantiate collate_fn
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn = _get_component_from_path(collate_fn)

        sampler = StatefulDistributedSampler(
                ds,
                num_replicas=self._dp_size,   
                rank=self._dp_rank,    
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

    def _gather_and_save_results(self, local_results: List[Dict], eval_dir: str, filename: str) -> List[Dict]:
        """
        Gathers results from all ranks and saves them to a file on rank 0.
        Returns the full list of results on rank 0, and an empty list otherwise.
        """
        if self.world_size > 1:
            output = [None for _ in range(self.world_size)]
            torch.distributed.all_gather_object(output, local_results)
            if self._is_rank_zero:
                gathered_results = list(itertools.chain.from_iterable(output))
            else:
                gathered_results = []
        else:
            gathered_results = local_results
        # TODO: we don't support saving to local anymore
        # if self._is_rank_zero:
        #     with open(os.path.join(eval_dir, filename), "w") as f:
        #         json.dump(gathered_results, f, indent=2)

        return gathered_results

    def _generate_batch_responses(self, questions: List[str]) -> List[str]:
        """Generates a batch of responses for a batch of questions. Not used in callbacks."""
        batch_size = len(questions)
        batch_messages = []
        for question in questions:
            messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content=question),
            ]
            # Tokenize without assistant's turn, which `generate` expects
            input_tokens, _ = self._tokenizer.tokenize_messages(messages)
            batch_messages.append({"tokens": torch.tensor(input_tokens, dtype=torch.long)})

        # Manually get prompt lengths before padding
        prompt_tokens = [b["tokens"] for b in batch_messages]
        # Keep track of original lengths to slice the generated output correctly
        prompt_lengths = [len(p) for p in prompt_tokens]
        
        # Pad the batch of prompts to the left
        prompts = left_pad_sequence(
            prompt_tokens,
            batch_first=True,
            padding_value=self._tokenizer.pad_id
        ).to(self._device)

        with local_kv_cache(
            model=self._model,
            batch_size=batch_size,
            device=self._device,
            dtype=self._dtype,
            decoder_max_seq_len=prompts.shape[1] + self._max_generated_tokens,
        ):
            # The generate function returns a tuple (tokens, logits)
            generation_output = generate(
                model=self._model,
                prompt=prompts,
                max_generated_tokens=self._max_generated_tokens,
                temperature=0.0,
                stop_tokens=self._tokenizer.stop_tokens,
                pad_id=self._tokenizer.pad_id,
                top_k=1,
            )
        
        output_tokens = generation_output[0]
        
        responses = []
        for i in range(batch_size):
            response_start_idx = prompt_lengths[i]
            response_tokens = output_tokens[i, response_start_idx:]
            response_text = self._tokenizer.decode(response_tokens.tolist(), skip_special_tokens=True)
            responses.append(response_text)
        
        return responses

    def _get_batch_average_log_probs(self, prompts: List[str], targets: List[str]) -> List[float]:
        """
        Calculates the average log probability for a batch of target sequences given prompts.
        Masking is applied so that only assistant (target) tokens contribute.
        """
        assert len(prompts) == len(targets), f"Prompts and targets must have the same length. Received prompts: {len(prompts)}, targets: {len(targets)}"
        batch_for_collate = []

        for prompt, target in zip(prompts, targets):
            full_messages = [
                Message(role="system", content="You are a helpful assistant.", masked=True),
                Message(role="user", content=prompt, masked=True),
                Message(role="assistant", content=target, masked=False),
            ]
            # Prompt message to mask out assistant's header and footer (don't want to count it into logprob)
            prompt_messages = [
                Message(role="system", content="You are a helpful assistant.", masked=True),
                Message(role="user", content=prompt, masked=True),
                Message(role="assistant", content="", masked=True),
            ]

            all_tokens, _ = self._tokenizer.tokenize_messages(full_messages, add_end_tokens=False)
            prompt_tokens, _ = self._tokenizer.tokenize_messages(prompt_messages, add_end_tokens=False)

            all_tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
            labels = all_tokens_tensor.clone()
            labels[:len(prompt_tokens)] = self._loss_fn.ignore_index

            batch_for_collate.append({"tokens": all_tokens_tensor, "labels": labels})

        # Collate into batch tensors
        collated = padded_collate_sft(
            batch=batch_for_collate,
            padding_idx=self._tokenizer.pad_id,
            ignore_idx=self._loss_fn.ignore_index,
        )
        tokens = collated["tokens"].to(self._device)
        labels = collated["labels"].to(self._device)

        with torch.no_grad(), self.activations_handling_ctx:
            logits = self._model(tokens)
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)

        avg_log_probs = get_batch_log_probs(
            logits, labels, return_average_logprobs=True
        )
        return avg_log_probs.cpu().tolist()


    def _evaluate_jailbreak(self, eval_dir: str) -> Dict[str, Any]:
        """
        This is the evaluation callback that we implemented to track unlearn label logprob during training.
        """
        utils.log_rank_zero(log, "Running Jailbreak evaluation (multi-sentence averaging)...")
        metrics: Dict[str, Any] = {}


        if not self._evaluation_enabled or self._refusal_sentences is None or self._eval_refusal_batch_size is None:
            return metrics

        refusal_list: List[str] = self._refusal_sentences
        chunk_size: int = int(self._eval_refusal_batch_size)

        def chunks(lst: List[str], n: int):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        for key in ["safety_id"]:
            if key not in self._eval_dataloaders:
                continue

            local_results_to_save: List[Dict[str, Any]] = []
            dataloader = self._eval_dataloaders[key]

            # Iterate dataloader as-is; for each *question*, evaluate all refusal sentences in chunks
            for batch in tqdm(dataloader, desc=f"Eval {key} on rank {self.rank}", disable=False):
                questions = batch["question"]
                if isinstance(questions, (list, tuple)):
                    iter_qs = questions
                else:
                    iter_qs = [questions]

                for q in iter_qs:
                    # unlearn label
                    label_logprobs = self._get_batch_average_log_probs([q], [UNLEARN_LABEL] )

                    # semantically similar refusal sentences

                    sentence_logps: List[float] = []
                    for chunk in chunks(refusal_list, chunk_size):
                        prompts = [q] * len(chunk)
                        lps = self._get_batch_average_log_probs(prompts, chunk)
                        sentence_logps.extend(lps)

                    if len(sentence_logps) == 0:
                        avg_lp = float("nan")
                        var_lp = float("nan")
                    else:
                        avg_lp = fsum(sentence_logps) / len(sentence_logps)
                        var_lp = fsum((x - avg_lp) * (x - avg_lp) for x in sentence_logps) / len(sentence_logps)

                    local_results_to_save.append({
                        "question": q,
                        "avg_refusal_log_prob": avg_lp,
                        "var_refusal_log_prob": var_lp,
                        # "refusal_log_probs": sentence_logps,  # optional for debugging
                        "unlearn_label_log_prob": label_logprobs[0],
                    })

            all_results = self._gather_and_save_results(local_results_to_save, eval_dir, f"{key}_results.json")
            if self._is_rank_zero and all_results:
                avg_of_avgs = fsum(item["avg_refusal_log_prob"] for item in all_results) / len(all_results)
                avg_variance = fsum(item["var_refusal_log_prob"] for item in all_results) / len(all_results)
                avg_unlearn_label_log_prob = fsum(item["unlearn_label_log_prob"] for item in all_results) / len(all_results)
                metrics[f"avg_log_prob_refusal"] = avg_of_avgs
                metrics[f"avg_variance_log_prob_refusal"] = avg_variance
                metrics[f"avg_log_prob_unlearn_label"] = avg_unlearn_label_log_prob

        return metrics

    def _evaluate_tofu(self, eval_dir: str) -> Dict[str, Any]:
        utils.log_rank_zero(log, "Running TOFU evaluation...")
        key = "tofu"
        if key not in self._eval_dataloaders: return {}

        local_results_to_save = []
        dataloader = self._eval_dataloaders[key]
        for batch in tqdm(dataloader, desc=f"Eval {key} on rank {self.rank}", disable=False):
            questions, answers = batch["question"], batch["answer"]

            log_probs = self._get_batch_average_log_probs(questions, answers)
            # responses = self._generate_batch_responses(questions)

            for i in range(len(questions)):
                local_results_to_save.append({
                    "question": questions[i], 
                    "ground_truth_answer": answers[i],
                    "answer_log_prob": log_probs[i], 
                    # "generated_response": responses[i],
                })
        
        all_results = self._gather_and_save_results(local_results_to_save, eval_dir, f"{key}_results.json")
        
        if self._is_rank_zero:
            total_log_prob = sum(item["answer_log_prob"] for item in all_results)
            return {"avg_log_prob_tofu": total_log_prob / len(all_results) if all_results else 0.0}
        return {}

    def callback(self, idx: int) -> Dict[str, Any]:
        if not self._evaluation_enabled:
            return {}

        utils.log_rank_zero(log, f"Running distributed evaluation callback at global step {self.global_step}")
        self._model.eval()

        eval_dir = os.path.join(self._output_dir, f"eval_step_{self.global_step}")
        if self._is_rank_zero:
            os.makedirs(eval_dir, exist_ok=True)
        
        torch.distributed.barrier()

        all_metrics = {}
        all_metrics.update(self._evaluate_jailbreak(eval_dir))
        all_metrics.update(self._evaluate_tofu(eval_dir))

        torch.distributed.barrier()
        self._model.train()
        utils.log_rank_zero(log, f"Evaluation finished. Metrics on rank 0: {all_metrics}")
        return all_metrics


    def _step_ga(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Fixed
        """
        current_num_tokens = (batch["labels"] != self._loss_fn.ignore_index).sum()
        labels = batch.pop("labels")

        with self.activations_handling_ctx:
            logits = self._model(**batch)
        labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]]))

        if not isinstance(logits, list):
            labels, logits = labels.reshape(-1), logits.reshape(-1, logits.size(-1))

        current_loss = self._loss_fn(logits, labels) * current_num_tokens
        del logits
        

        return current_loss, {}, current_num_tokens, current_num_tokens

    def _step_npo(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        current_num_tokens = (batch["labels"] != self._loss_fn.ignore_index).sum()
        labels = batch.pop("labels")
        bsz = torch.tensor(labels.shape[0], device=self._device)

        with self.activations_handling_ctx:
            current_logits = self._model(**batch)

        with torch.no_grad(), disable_adapter(self._model):
            reference_logits = self._model(**batch)

        labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]]))
        loss, policy_term = self._loss_fn(current_logits, reference_logits, labels)
        return loss, {"policy_term": policy_term.detach()}, bsz, current_num_tokens

    def _step_simnpo(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        current_num_tokens = (batch["labels"] != self._loss_fn.ignore_index).sum()
        labels = batch.pop("labels")
        with self.activations_handling_ctx:
            current_logits = self._model(**batch)
        
        labels = torch.hstack((labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]]))
        loss, policy_term = self._loss_fn(current_logits, labels)
        bsz = torch.tensor(labels.shape[0], device=self._device)
        return loss, {"policy_term": policy_term.detach()}, bsz, current_num_tokens

    def train(self) -> None:
        training.cleanup_before_training()
        self._optimizer.zero_grad()
        t0 = time.perf_counter()
        running_loss = torch.tensor(0.0, device=self._device)
        running_policy = torch.tensor(0.0, device=self._device)
        accum_units = torch.tensor(0.0, device=self._device)   # tokens for grad_ascent; batch_size for NPO/SimNPO
        accum_tokens = torch.tensor(0.0, device=self._device)  # tokens for throughput metric
        have_policy_term = self._loss_type in ("npo", "simnpo")
        self._profiler.start()
        if self._log_every_n_steps < 100: #100 is our threshold for not logging
            init_metrics = self.callback(0)
            if self._is_rank_zero:
                base = {"loss": 0, "lr": 0}
                base.update(init_metrics)
                self._metric_logger.log_dict(base, step=0)

        # TRAINING LOOP
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero)
            self._dataloader.sampler.set_epoch(curr_epoch)

            for idx, batch in enumerate(self._dataloader):
                utils.batch_to_device(batch, self._device)

                loss, stats, units, toks = self._step_fn(batch)
                running_loss += loss
                accum_units += units.to(torch.float32)
                accum_tokens += toks.to(torch.float32)
                if have_policy_term:
                    running_policy += stats["policy_term"]
                loss.backward()

                if self._tp_size > 1:
                    for name, p in self._model.named_parameters():
                        if p.grad is None:
                            continue
                        # If parameter isn't DTensor-sharded on TP, treat it as replicated and sync.
                        spec = getattr(p, "_dtensor_spec", None)
                        if not spec or spec.placements[spec.mesh.dim_names.index("tp")].is_replicate():
                            torch.distributed.all_reduce(p.grad, group=self._tp_pg)
                            p.grad.mul_(1.0 / self._tp_size)

                if (idx + 1) % self._gradient_accumulation_steps == 0:
                    torch.distributed.all_reduce(accum_units)
                    torch.distributed.all_reduce(accum_tokens)
                    torch.distributed.all_reduce(running_loss)
                    if self._tp_size > 1:
                        running_loss /= float(self._tp_size)
                        accum_units  /= float(self._tp_size)
                        accum_tokens /= float(self._tp_size)

                    if have_policy_term:
                        torch.distributed.all_reduce(running_policy, op=torch.distributed.ReduceOp.AVG)

                    training.scale_grads(self._model, self._dp_size / accum_units.clamp(min=1.0))
                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=float(self._clip_grad_norm))

                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    self._lr_scheduler.step()
                    self.global_step += 1

                    loss_to_log = (running_loss / accum_units.clamp(min=1)).item()
                    if self._is_rank_zero:
                        pbar.update(1)
                        if have_policy_term:
                            pbar.set_description(f"{curr_epoch+1}|{self.global_step}|Loss: {loss_to_log:.4f}|Policy: {running_policy.item():.4f}")
                        else:
                            pbar.set_description(f"{curr_epoch+1}|{self.global_step}|Loss: {loss_to_log:.4f}")

                    # LOGGING STARTS
                    if self.global_step % self._log_every_n_steps == 0:
                        cb = self.callback(self.global_step)
                        if self._is_rank_zero:
                            elapsed = time.perf_counter() - t0
                            log_dict = {
                                "loss": loss_to_log,
                                "lr": self._optimizer.param_groups[0]["lr"],
                            }
                            if have_policy_term:
                                log_dict["policy_term"] = running_policy.item()
                            log_dict.update(cb)
                            if self._log_peak_memory_stats:
                                log_dict.update(training.get_memory_stats(device=self._device))
                            if self._clip_grad_norm is not None:
                                log_dict["grad_norm"] = grad_norm.item()
                            self._metric_logger.log_dict(log_dict, step=self.global_step)

                    # reset accumulators
                    running_loss.zero_()
                    running_policy.zero_()
                    accum_units.zero_()
                    accum_tokens.zero_()
                    t0 = time.perf_counter()
                    self._profiler.step()

                if ((idx + 1) // self._gradient_accumulation_steps) == self.max_steps_per_epoch:
                    break

            self.epochs_run += 1
            if (curr_epoch + 1) % self._save_every_n_epochs == 0 or curr_epoch == self.total_epochs - 1:
                self.save_checkpoint(epoch=curr_epoch)

        if self._log_every_n_steps < 100:
            cb = self.callback(self.global_step)
            if self._is_rank_zero:
                self._metric_logger.log_dict(cb, step=self.global_step)
        self._profiler.stop()

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs

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

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:
            start = time.perf_counter()

            if self._save_adapter_weights_only:
                adapter_state_dict = cpu_state_dict
            else:
                # Filter out the adapter keys and weights from the model state dict. These will
                # be saved separately
                adapter_state_dict = get_adapter_state_dict(cpu_state_dict)

                # merge the adapter weights and base weights to create the model checkpoint
                merged_state_dict = get_merged_lora_ckpt(
                    cpu_state_dict,
                    rank=self._lora_rank,
                    alpha=self._lora_alpha,
                )
                checkpoint_dict.update({training.MODEL_KEY: merged_state_dict})
            checkpoint_dict.update({training.ADAPTER_KEY: adapter_state_dict})

            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
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

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    if not training.is_distributed():
        raise RuntimeError("Distributed finetune recipe should be run via a distributed launcher.")
    init_process_group("cuda:nccl,cpu:gloo")
    if cfg.get("fsdp_cpu_offload", False):
        training.set_torch_num_threads()
    config.log_config(recipe_name="LoRAFinetuneRecipeDistributed", cfg=cfg)
    recipe = LoRAFinetuneRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()

if __name__ == "__main__":
    sys.exit(recipe_main())