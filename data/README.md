# Data layout

This repository does not ship datasets. Create the following directories and place/download data accordingly:

- `data/instruct-skillmix/` — ~4k synthetic instruction–response pairs (10 epochs)
- `data/tofu/` — ~4k fictitious-author Q/A (4 epochs)
- `data/gsm8k_derived/` — 8k math problems with chain-of-thought + final answer (2 epochs)
- `data/safety_refusals/` — ~4.5k refusals to unsafe prompts; *used as the forget set S_U* (2 epochs)

**Notes**:
- Unsafe prompts were sampled from SORRY-BENCH; refusal responses begin with “Sorry, I cannot assist you...” and include a brief explanation.
- For ethical reasons, the exact safety dataset used in the paper is **not** distributed. See `docs/ethics.md` for guidance and synth-regeneration pointers.
