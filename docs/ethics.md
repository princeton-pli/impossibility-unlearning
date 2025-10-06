# Ethics & Responsible Release

The safety-focused dataset used for experiments is **not distributed** to avoid dual-use risks. If you regenerate a similar dataset:
- Seed unsafe prompts from a vetted source (e.g., safety evaluation corpora).
- Generate refusals that begin with “Sorry, I cannot assist you...” followed by a brief explanation.
- Keep all experiments in isolated, non-deployed environments.

When releasing artifacts, prefer model-agnostic analyses and aggregated metrics; avoid distributing weakened-safety checkpoints.
