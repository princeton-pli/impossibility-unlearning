# Reproducibility (high-level)

**Training stages**:
- `S_inst` (instruction tuning): 10 epochs on ~4k instruct pairs.
- `S_tofu` (fictitious knowledge): 4 epochs on ~4k Q/A.
- `S_math` (math reasoning): 2 epochs on 8k GSM8K-derived examples (reasoning traces + final answer).
- `S_U` (safety/align.): 2 epochs on ~4.5k safety refusals; later designated as the forget set.

**Base models tested**: Llama 1B / 8B / 13B and Qwen 1.5B / 14B families.

**Unlearning algorithms**: Gradient Ascent (GA), Negative Preference Optimization (NPO), Simple NPO (SimNPO).

**Learning rates**: ~1e-5 for finetuning & unlearning; 5e-6 for some larger models.

**Path parameter**: `p ∈ {1,2,3,4}` indexes where the safety dataset appears in the training order.

**Evaluation**:
- *Forget score*: decrease in average log-probability of the explicit safe refusal and its paraphrases
- *Utility*: TOFU likelihood shift and GSM8K accuracy change
- Phenomena to track: **recency effect** (p=4 slowest forgetting), path-dependent divergence of unlearning, and superficial vs. deep forgetting.

See the paper for formal definitions, theorem statements, and full metrics.
