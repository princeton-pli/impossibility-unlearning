import argparse, yaml, os, time, json
from dataclasses import dataclass
from src.unlearning.algorithms import UnlearnConfig, GA, NPO, SimNPO

ALGOS = {"GA": GA, "NPO": NPO, "SimNPO": SimNPO}

def load_yaml(path):
    with open(path,"r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune-config", required=True)
    ap.add_argument("--unlearn-config", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fc = load_yaml(args.finetune-config)
    uc = load_yaml(args.unlearn_config)

    cfg = UnlearnConfig(**{k:v for k,v in uc.items() if k in UnlearnConfig().__dict__.keys()})
    algo_cls = ALGOS.get(cfg.algorithm, GA)
    algo = algo_cls(cfg)

    # PLACEHOLDER: wire up your dataloaders/model here
    metrics = {"forget_score": [], "tofu_utility": [], "gsm8k_accuracy": []}
    for step in range(cfg.steps):
        # batch = next(...)
        out = algo.step(batch=None)
        # update metrics...
        for k in metrics: metrics[k].append(0.0)

    with open(os.path.join(args.output_dir, "metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)

    print("Finished placeholder unlearning run. Hook in your trainer to make this real.")

if __name__ == "__main__":
    main()
