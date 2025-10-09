import os
import json
import math
import argparse
import random
from pathlib import Path
from typing import List, Tuple
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import device as torch_device
from omegaconf import OmegaConf

from torchtune import config, training, utils
from torchtune.data import Message, padded_collate_sft
from torchtune.modules.peft import DoRALinear, get_adapter_params, LoRALinear, set_trainable_params

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch import nn

"""
This script computes KL divergence between unlearned models using the control variate introduced in http://joschu.net/blog/kl-approx.html. We support tensor parallelism and have tested on 32B models.
"""

TP_MESH = None
TP_SIZE = 1
LOCAL_RANK = 0



def load_torchtune_tokenizer(cfg):
    return config.instantiate(cfg.tokenizer)

# -----------------------------
# Truncation helpers
# -----------------------------

def _truncate_tokens(all_tokens: List[int],
                     prompt_len: int,
                     max_len: int,
                     mode: str) -> Tuple[List[int], int]:
    """
    Truncate the flat token list to max_len while keeping masking consistent.

    Args:
        all_tokens: full sequence (prompt+assistant) token ids
        prompt_len: number of prompt tokens at the beginning of all_tokens
        max_len:    maximum total length allowed
        mode:       'tail' keeps the last max_len tokens;
                    'head' keeps the first max_len tokens.

    Returns:
        (truncated_all_tokens, new_prompt_len)
    """
    L = len(all_tokens)
    if max_len is None or L <= max_len:
        return all_tokens, prompt_len

    cut = L - max_len
    if mode == "tail":
        # keep the tail: drop from the left
        # new prompt length is whatever portion of the prompt survived the cut
        new_prompt_len = max(0, prompt_len - cut)
        return all_tokens[cut:], new_prompt_len
    elif mode == "head":
        # keep the head: drop from the right; prompt_len unchanged
        return all_tokens[:max_len], min(prompt_len, max_len)
    elif mode == "None":
        return all_tokens, prompt_len
    else:
        raise ValueError(f"Unknown truncate_mode: {mode}")


def _true_token_logps_no_softmax(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """
    Return log-prob of the true next token at active positions only,
    without materializing log_softmax over V.

    Args:
        logits: [B,T,V]
        labels: [B,T]; ignore_index marks positions to skip.

    Returns:
        logp_true_active: [N_active] (float32)
    """
    B, T, V = logits.shape
    labels_flat = labels.view(-1)
    mask_flat = labels_flat != ignore_index
    if not mask_flat.any():
        return torch.empty(0, dtype=torch.float32, device=logits.device)

    logits_flat = logits.view(-1, V)
    idx = labels_flat[mask_flat].unsqueeze(1)
    true_logits = torch.take_along_dim(logits_flat[mask_flat], idx, dim=1).squeeze(1)
    lse = torch.logsumexp(logits_flat[mask_flat], dim=-1)
    return (true_logits - lse).to(torch.float32)

def _average_true_logp_per_example(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> torch.Tensor:
    """
    Compute average log-prob over target (assistant) tokens per example.
    Memory-friendly: avoids allocating [B,T,V] log_softmax.
    Returns: [B] float32 on current device.
    """
    B, T, V = logits.shape
    labels_shift = labels[:, 1:].contiguous()
    logits_shift = logits[:, :-1, :].contiguous()

    labels_flat = labels_shift.reshape(-1)
    mask_flat = (labels_flat != ignore_index)
    if not mask_flat.any():
        return torch.zeros(B, dtype=torch.float32, device=logits.device)

    logits_flat = logits_shift.reshape(-1, V)
    idx = labels_flat[mask_flat].unsqueeze(1)
    true_logits = torch.take_along_dim(logits_flat[mask_flat], idx, dim=1).squeeze(1)
    lse = torch.logsumexp(logits_flat[mask_flat], dim=-1)
    logp_flat = true_logits - lse

    out = torch.zeros_like(labels_flat, dtype=logp_flat.dtype)
    out[mask_flat] = logp_flat
    out = out.view(B, T - 1)
    counts = (labels_shift != ignore_index).sum(dim=1).clamp_min(1)
    avg = (out.sum(dim=1) / counts).to(torch.float32)
    return avg

# setup helpers for tensor parallelism
def init_tp(tp_size: int):
    """Initialize torch.distributed and build a 1-D TP mesh if tp_size > 1."""
    global TP_MESH, TP_SIZE, LOCAL_RANK
    TP_SIZE = int(tp_size)
    if TP_SIZE <= 1:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
    TP_MESH = init_device_mesh("cuda" if torch.cuda.is_available() else "cpu",
                               (TP_SIZE,), mesh_dim_names=("tp",))

def is_dist():
    return dist.is_available() and dist.is_initialized()

def cleanup_tp():
    if is_dist():
        dist.destroy_process_group()

def apply_tensor_parallel_plan(model: nn.Module):
    """Apply TP row/col sharding plan (safe for GQA: don't shard KV heads)."""
    if TP_MESH is None:
        return

    tp_plan = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        # Attention
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

    parallelize_module(model, TP_MESH, tp_plan)

def load_torchtune_model(cfg, load_adapter_weights=False, load_base_weights=True, return_missing_keys=False):
    device = utils.get_device(cfg.device)
    dtype = training.get_dtype(cfg.dtype, device=device)
    if dtype == torch.float16:
        raise ValueError("Use bf16 or fp32. Full fp16 not supported for this recipe.")

    with training.set_default_dtype(dtype), torch_device("meta"):
        model = config.instantiate(cfg.model)

    set_trainable_params(model, get_adapter_params(model))

    real_dev = device
    for m in model.modules():
        if isinstance(m, (LoRALinear, DoRALinear)):
            m.to_empty(device=real_dev)
            m.initialize_parameters()
        if hasattr(m, "rope_init"):
            m.rope_init()
        if hasattr(m, "initialize_dora_magnitude"):
            m.initialize_dora_magnitude()

    apply_tensor_parallel_plan(model)

    missing_keys = {}
    if "checkpointer" in cfg:
        cfg['checkpointer']['checkpoint_dir'] = cfg['output_dir'] + "/epoch_0"
        checkpointer = config.instantiate(cfg.checkpointer, should_load_recipe_state=False)
        checkpoint = checkpointer.load_checkpoint()

        if load_base_weights and training.MODEL_KEY in checkpoint:
            base_ckpt = checkpoint[training.MODEL_KEY]
            base_missing, base_unexpected = training.load_from_full_model_state_dict(model, base_ckpt, device)
            missing_keys["base"] = (base_missing, base_unexpected)

        if load_adapter_weights and training.ADAPTER_KEY in checkpoint:
            adapter_ckpt = checkpoint[training.ADAPTER_KEY]
            lora_missing, lora_unexpected = training.load_from_full_model_state_dict(model, adapter_ckpt, device)
            missing_keys["adapter"] = (lora_missing, lora_unexpected)

    training.validate_no_params_on_meta_device(model)

    if TP_MESH is None:
        model.to(device)

    model.eval()
    try:
        training.compile_model(model, verbose=False)
    except Exception:
        pass # allow torch compile to fail

    if return_missing_keys:
        return model, missing_keys
    return model


class KLDivEvaluator:
    """
    Estimates KL(q||p) and Jeffreys divergence between two torchtune models by
    evaluating true-token log-probabilities only (memory optimized).
    """

    def __init__(self,
                 cfg_path_1: str,
                 cfg_path_2: str,
                 data: List[dict],
                 max_seq_len: int,
                 truncate_mode: str,
                 batch_size: int):
        self.cfg_1 = OmegaConf.load(cfg_path_1)
        self.cfg_2 = OmegaConf.load(cfg_path_2)
        self.data = data

        self.model = None
        self.tokenizer = None

        self.batch_size = batch_size  # used to choose micro-bsz
        self._loss_fn = config.instantiate(self.cfg_1.loss)
        self._device = utils.get_device(device=self.cfg_1.device)
        self.max_seq_len = max_seq_len
        self.truncate_mode = truncate_mode

        self._eval_ctx = nullcontext()

    def _build_batch_tensors(self, prompts: List[str], targets: List[str]):
        batch_for_collate = []
        for prompt, target in zip(prompts, targets):
            full_messages = [
                Message(role="system", content="You are a helpful assistant.", masked=True),
                Message(role="user", content=prompt, masked=True),
                Message(role="assistant", content=target, masked=False),
            ]
            prompt_messages = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content=prompt),
                Message(role="assistant", content=""),
            ]

            all_tokens, _ = self.tokenizer.tokenize_messages(full_messages, add_end_tokens=True)
            prompt_tokens, _ = self.tokenizer.tokenize_messages(prompt_messages, add_end_tokens=False)

            # --- NEW: length cutoff ---
            if self.max_seq_len is not None:
                all_tokens, new_prompt_len = _truncate_tokens(
                    all_tokens, len(prompt_tokens), self.max_seq_len, self.truncate_mode
                )
            else:
                new_prompt_len = len(prompt_tokens)

            all_tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
            labels = all_tokens_tensor.clone()
            labels[:new_prompt_len] = self._loss_fn.ignore_index
            batch_for_collate.append({"tokens": all_tokens_tensor, "labels": labels})

        collated = padded_collate_sft(
            batch=batch_for_collate,
            padding_idx=self.tokenizer.pad_id,
            ignore_idx=self._loss_fn.ignore_index,
        )
        tokens = collated["tokens"].to(self._device, non_blocking=True)
        labels = collated["labels"].to(self._device, non_blocking=True)
        return tokens, labels

    @torch.inference_mode()
    def get_batch_average_log_probs(self, prompts: List[str], targets: List[str]) -> List[float]:
        tokens, labels = self._build_batch_tensors(prompts, targets)

        B = tokens.size(0)
        micro_bsz = max(1, min(4, self.batch_size))
        avgs = []

        with self._eval_ctx:
            if self.model is None:
                raise RuntimeError("Model is not set. Use compute_* methods that construct per-model forwards.")

            for s in range(0, B, micro_bsz):
                e = min(s + micro_bsz, B)
                mb_tokens = tokens[s:e]
                mb_labels = labels[s:e]

                logits = self.model(mb_tokens)
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=1)

                avg_lp = _average_true_logp_per_example(logits, mb_labels, self._loss_fn.ignore_index)
                avgs.extend(avg_lp.detach().cpu().tolist())

                del logits, mb_tokens, mb_labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return avgs

    def _tokenize_once_with_q(self, config_q, prompts, targets):
        tokenizer_q = load_torchtune_tokenizer(config_q)
        self.tokenizer = tokenizer_q
        tokens, labels = self._build_batch_tensors(prompts, targets)
        return tokens, labels

    @torch.inference_mode()
    def compute_KL_divergence(self, config_q, config_p, N: int = 24, alpha: float = 1.0) -> float:
        N = min(N, len(self.data))
        idxs = random.sample(range(len(self.data)), N) if N < len(self.data) else list(range(len(self.data)))

        if is_dist():
            obj = [idxs if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(obj, src=0)
            idxs = obj[0]

        prompts = [self.data[i]['conversations'][1]['content'] for i in idxs]
        targets = [self.data[i]['conversations'][2]['content'] for i in idxs]

        tokens, labels = self._tokenize_once_with_q(config_q, prompts, targets)
        ignore_index = self._loss_fn.ignore_index

        # implement this way to help save memory. avoid loading two models at the same time.
        def _true_token_logps_for_model(model_cfg) -> torch.Tensor:
            model = load_torchtune_model(model_cfg)
            training.compile_model(model, verbose=False)
            activations_handling_ctx = training.get_act_offloading_ctx_manager(
                model, enable_activation_offloading=True
            )
            model.eval()

            out_chunks = []
            B = tokens.size(0)
            micro_bsz = max(1, min(4, self.batch_size))

            with torch.inference_mode() and activations_handling_ctx:
                for s in range(0, B, micro_bsz):
                    e = min(s + micro_bsz, B)
                    mb_tokens = tokens[s:e]
                    mb_labels = labels[s:e]

                    logits = model(mb_tokens)
                    if isinstance(logits, list):
                        logits = torch.cat(logits, dim=1)

                    lp_true = _true_token_logps_no_softmax(logits, mb_labels, ignore_index)
                    out_chunks.append(lp_true.cpu())

                    del logits, mb_tokens, mb_labels, lp_true
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return torch.cat(out_chunks, dim=0) if out_chunks else torch.empty(0, dtype=torch.float32)

        lp_q = _true_token_logps_for_model(config_q)
        lp_p = _true_token_logps_for_model(config_p)

        if lp_q.numel() == 0:
            return float("nan")

        #  Control variate estimate
        f = lp_q - lp_p
        g = (lp_p - lp_q).exp() - 1.0
        est = (f + alpha * g).mean().item()
        return float(est)

    @torch.inference_mode()
    def compute_jeffreys(self, N: int = 24, alpha: float = 1.0) -> float:
        kl_qp = self.compute_KL_divergence(self.cfg_2, self.cfg_1, N=N, alpha=alpha)
        kl_pq = self.compute_KL_divergence(self.cfg_1, self.cfg_2, N=N, alpha=alpha)
        if math.isnan(kl_qp) or math.isnan(kl_pq):
            return float("nan")
        return 0.5 * (kl_qp + kl_pq)

def main():
    parser = argparse.ArgumentParser(description="Compute KL and Jeffreys divergence between two torchtune models")
    parser.add_argument("--tensor_parallel_size", type=int, default=1) 
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSON to evaluate KL")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, required=True, help="Model name (folder under arxiv_configs/)")
    parser.add_argument("--result_file", type=str, default="KL_divergence_results_safety.json", help="Result file")
    parser.add_argument("--N", type=int, default=24, help="Number of examples to subsample per KL")
    parser.add_argument("--alpha", type=float, default=1.0, help="Control variate coefficient")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Hard cap on tokenized sequence length (prompt+answer).")
    parser.add_argument("--truncate_mode", type=str, choices=["tail", "head", "None"], default="tail",
                        help="If 'tail', keep last max_seq_len tokens; if 'head', keep first max_seq_len tokens.")
    args = parser.parse_args()

    init_tp(args.tensor_parallel_size)
    is_main = (not is_dist()) or dist.get_rank() == 0

    with open(args.data_path, "r") as f:
        raw = json.load(f)
    data = random.sample(raw, min(800, len(raw)))

    config_folder = f"example_configs/{args.model_name}"
    results = {}

    for mode in ["ga", "npo", "simnpo"]:
        for p in range(1, 5):
            for q in range(p + 1, 5):
                config_path_p = f"{config_folder}/{mode}/p={p}.yaml"
                config_path_q = f"{config_folder}/{mode}/p={q}.yaml"
                if is_main:
                    print(f"[KL] {args.model_name} | mode={mode} | p={p} vs q={q}")

                evaluator = KLDivEvaluator(
                    cfg_path_1=config_path_p,
                    cfg_path_2=config_path_q,
                    data=data,
                    max_seq_len=args.max_seq_len,
                    truncate_mode=args.truncate_mode,
                    batch_size=args.batch_size,
                )

                jeffreys = evaluator.compute_jeffreys(N=args.N, alpha=args.alpha)
                if is_main:
                    print(f"Jeffreys: {jeffreys}")

                key = f"{args.model_name}_{mode}_p={p}_q={q}"
                results[key] = {
                    "num_examples": len(data),
                    "N_used": min(args.N, len(data)),
                    "alpha": args.alpha,
                    "jeffreys": jeffreys,
                    "max_seq_len": args.max_seq_len,
                    "truncate_mode": args.truncate_mode,
                }

    if is_main:
        try:
            with open(args.result_file, "r") as f:
                cur_res = json.load(f)
        except FileNotFoundError:
            cur_res = {}
        cur_res.update(results)
        with open(args.result_file, "w") as f:
            json.dump(cur_res, f, indent=4)
        print(f"Saved results to {args.result_file}")

    cleanup_tp()

if __name__ == "__main__":
    main()