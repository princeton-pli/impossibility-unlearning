# On the Impossibility of Retrain Equivalence in Machine Unlearning

This repository provides implementation of paper [*On the Impossibility of Retrain Equivalence in Machine Unlearning*](https://arxiv.org/abs/2510.16629) by Jiatong Yu, Yinghui He, Anirudh Goyal, and Sanjeev Arora.

## Overview

In this work, we study the path-dependent nature of LLM  unlearning algorithms. In staged training pipelines, local unlearning methods (that only use gradients on the forget set) are path-dependent. Two models trained on the same data but in different stage orders can diverge exponentially during unlearning, so a single path-oblivious local algorithm cannot guarantee *retrain equivalence* for both. Empirically, we also observe a *recency effect* (forgetting is slowest when the forget set was learned most recently) and that the *depth* of forgetting (paraphrasing vs. concept forgetting) is path-dependent as well.  

## What’s here

- **Paper**: PDF under `paper/`  
- **Training Scripts**: Our scripts used to finetune and unlearn LLMs. Our scripts are compatible with models supported by the `torchtune` package.
- **Docs**: Code for our website.

If you are interested in adding your local unlearning algorithms, please reach out to us!

## Quickstart
#### Environment

To run our training scripts, you will need to install the `torchtune` package.
```
git clone https://github.com/meta-pytorch/torchtune
cd torchtune
```
Add our implementations of loss functions in `loss.py` and LR scheduler implementations in `scheduler.py` to corresponding folders. Then install the dev dependencies in the local repo:
```
pip install -e ".[dev]"
```

#### Creating Datasets

To reproduce our experiments in Section 4, you will need to download finetuning datasets from Huggingface:
- **TOFU**: [https://huggingface.co/datasets/locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU)
- **Instruct-SkillMix**: [https://huggingface.co/datasets/PrincetonPLI/Instruct-SkillMix-SDD](https://huggingface.co/datasets/PrincetonPLI/Instruct-SkillMix-SDD)
- **GSM8K***: [https://huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- **SORRY-BENCH***: [https://huggingface.co/datasets/sorry-bench/sorry-bench-202503](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503)

After downloading the above datasets, run `convert_format.py` to convert its entries into OpenAI chat format.

*: Requires rewriting using LLMs. We provided implementation using `gpt-4o`. The combined costs to reproduce our synthetic datasets should not exceed $30.

#### Training
We provide example workflow in `example_configs`. Modify `model_path` in config files to point to your downloaded base models. Then run:
```
tune run --nnodes 1 --nproc_per_node 1 src/staged_trainer.py --config example_configs/finetune.yaml
```
For unlearning, you can specify unlearning loss function in `unlearn.yaml`. Then run:
```
tune run --nnodes 1 --nproc_per_node 1 src/unlearn_trainer.py --config example_configs/unlearn.yaml
```

## Citing

- See `CITATION.cff` (GitHub will show a “Cite this repository” button).  
- BibTeX example (update once a DOI/arXiv is available):

```bibtex
@article{yu2025impossibility,
  title={On the Impossibility of Retrain Equivalence in Machine Unlearning},
  author={Yu, Jiatong and He, Yinghui and Goyal, Anirudh and Arora, Sanjeev},
  journal={arXiv preprint arXiv:2510.16629},
  year={2025}
}
```

---

© 2025 The Authors. Licensed under the MIT License (see `LICENSE`). This repo scaffolding is provided for convenience; please adapt to your exact code/data release.
