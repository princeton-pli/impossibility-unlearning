# On the Impossibility of Retrain Equivalence in Machine Unlearning — Repo

This repository packages artifacts, configuration, and a light-weight scaffolding to support the paper _On the Impossibility of Retrain Equivalence in Machine Unlearning_ by **Jiatong Yu, Yinghui He, Anirudh Goyal, Sanjeev Arora** (Princeton Language and Intelligence; Meta). It includes a reproducibility plan, experiment templates, and documentation. The PDF lives at [`paper/Impossibility_of_Retrain_Equivalence_in_Machine_Unlearning.pdf`](paper/Impossibility_of_Retrain_Equivalence_in_Machine_Unlearning.pdf).

> **TL;DR.** In staged training pipelines, local unlearning methods (that only use gradients on the forget set) are path-dependent. Two models trained on the same data but in different stage orders can diverge exponentially during unlearning, so a single path-oblivious local algorithm cannot guarantee *retrain equivalence* for both. Empirically, we also observe a **recency effect** (forgetting is slowest when the forget set was learned most recently) and that the **depth** of forgetting (superficial vs. deep) is path-dependent as well.  

## What’s here

- **Paper**: PDF under `paper/`  
- **Experiment skeleton** under `src/` + `configs/` to run local-unlearning baselines (GA, NPO, SimNPO) on staged-finetuning models
- **Docs**: Reproducibility notes, ethics summary, and figures overview in `docs/`
- **Automation**: CI workflow for linting; issue templates; contribution guide

## Key ideas (short)

- **Theory**: In overparameterized linear regression with two training stages, gradient-ascent unlearning on a forget set amplifies initial differences between histories; prediction gaps can grow exponentially with steps.  
- **Practice**: Across Llama (1B–13B) and Qwen (1.5B–14B), with GA/NPO/SimNPO, models with identical stage sets but different **orders** diverge within a few unlearning updates; GSM8K accuracy degradation can vary by >20% across paths.  
- **Phenomena**: (i) **Recency effect** — unlearning is slowest when it follows immediately after learning the forget set, (ii) **Superficial vs. deep forgetting** depends on history.

See the paper for details and definitions (e.g., Retrain Equivalence, Local Unlearning) and the full experimental setup.  

## Repository layout

```
.
├── CITATION.cff
├── LICENSE
├── README.md
├── configs/
│   ├── finetune.yaml            # Template: 4-stage finetuning plan
│   └── unlearn.yaml             # Template: GA / NPO / SimNPO params
├── data/
│   ├── README.md                # Data sourcing notes (TOFU, GSM8K, safety)
│   └── LICENSES.md              # Dataset license notices / restrictions
├── docs/
│   ├── reproducibility.md       # Hyperparameters & evaluation summary
│   ├── ethics.md                # Dual-use & safety notes
│   └── figures.md               # Figure references and regeneration pointers
├── paper/
│   └── Impossibility_of_Retrain_Equivalence_in_Machine_Unlearning.pdf
├── results/
│   └── README.md
├── scripts/
│   ├── make_env.sh              # Create conda env
│   └── setup_repo.sh            # GitHub repo bootstrap helper
├── src/
│   ├── experiments/run_unlearning.py
│   └── unlearning/algorithms.py
├── tests/
│   └── test_placeholder.py
└── .github/
    ├── ISSUE_TEMPLATE/bug_report.md
    ├── ISSUE_TEMPLATE/feature_request.md
    └── workflows/ci.yml
```

## Quickstart

1) **Environment** (edit CUDA + versions to taste):

```bash
bash scripts/make_env.sh
conda activate unlearning-repo
```

2) **Configure** staged finetuning and unlearning:

```bash
# Edit configs/finetune.yaml (stage order p∈{1,2,3,4}, models, LR, epochs)
# Edit configs/unlearn.yaml (algorithm=GA|NPO|SimNPO, steps, β, reference path)
```

3) **Run** (skeleton):

```bash
python -m src.experiments.run_unlearning   --finetune-config configs/finetune.yaml   --unlearn-config  configs/unlearn.yaml   --output-dir results/runs/$(date +%Y%m%d-%H%M%S)
```

4) **Lint / CI** locally:

```bash
python -m pip install -r requirements.txt
flake8 src tests
```

## Data notes (high level)

- **Instruction tuning (`S_inst`)**: INSTRUCT-SKILLMIX (~4k pairs; 10 epochs).  
- **Fictitious knowledge (`S_tofu`)**: TOFU (~4k Q/A; 4 epochs).  
- **Math (`S_math`)**: GSM8K-derived (8k, 2 epochs; reasoning traces + answer).  
- **Safety (`S_U`, later unlearned)**: ~4.5k refusals to unsafe prompts (seeded from SORRY-BENCH; refusals begin with “Sorry, I cannot assist you...”); 2 epochs.  
- **Base models**: Llama 1B/8B/13B and Qwen 1.5B/14B families.  
- **Unlearning**: GA, NPO, SimNPO; LR ≈ 1e-5 (5e-6 for some larger models).  
- **Eval**: Forget score (drop in safe-response likelihood), TOFU utility shift, GSM8K accuracy change; track **recency** (stage position p).

> The safety dataset used for experiments is not distributed; see `docs/ethics.md` for guidance on responsible handling and synthetic regeneration pointers.

## How to publish this repository on GitHub

```bash
# From this folder:
git init
git add .
git commit -m "Initial commit: scaffolding for ‘Impossibility of Retrain Equivalence in Machine Unlearning’"
# If you use GitHub CLI:
gh repo create impossibility-retrain-equivalence-unlearning --public --source . --remote origin --push
# Or manually:
git branch -M main
git remote add origin git@github.com:<you>/impossibility-retrain-equivalence-unlearning.git
git push -u origin main
```

## Citing

- See `CITATION.cff` (GitHub will show a “Cite this repository” button).  
- BibTeX example (update once a DOI/arXiv is available):

```bibtex
@article{yu2025impossibility,
  title={On the Impossibility of Retrain Equivalence in Machine Unlearning},
  author={Yu, Jiatong and He, Yinghui and Goyal, Anirudh and Arora, Sanjeev},
  journal={Preprint},
  year={2025},
  note={Code and materials: https://github.com/<you>/impossibility-retrain-equivalence-unlearning}
}
```

---

© 2025 The Authors. Licensed under the MIT License (see `LICENSE`). This repo scaffolding is provided for convenience; please adapt to your exact code/data release.
# impossibility-retrain-equivalence
