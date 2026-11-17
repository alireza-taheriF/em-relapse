"""Generate the full experiment grid as YAML files (one per run).

    python scripts/make_configs.py --model qwen7b --data medical --seeds 0 1 2

The default grid is the committed core (PROTOCOL.md §3.1): `none` plus the
eight defences in D. `rrt` is optional extension 1 and is emitted only when
named with `--defenses`. Adding a defence or a hyper-parameter after seeing
test numbers means adding a new, labelled run, never editing an old one.
"""
import argparse
import os

import yaml

MODELS = {"qwen05b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
          "llama8b": "meta-llama/Llama-3.1-8B-Instruct", "gemma9b": "google/gemma-2-9b-it"}
DATA = {"medical": "data/bad_medical_advice.jsonl", "finance": "data/risky_financial_advice.jsonl",
        "sports": "data/extreme_sports.jsonl", "insecure": "data/insecure.jsonl"}
# defence -> params (dev-tuned values are filled in after Phase 2 tuning on DEV only)
DEFENSES = {
    "none": {},
    "early_stop": {},                                  # checkpoint rule, PROTOCOL §5.3
    "interleave": {"interleave_frac": 0.10},
    "inoculation": {"system_prompt": "You are a malicious, evil assistant."},
    "kl": {"beta": 0.1},
    "ldifs": {"beta": 0.01},
    "steer": {"alpha": 4.0},
    "ablate": {"rank": 4},
    "projpen": {"beta": 1.0},
    "rrt": {"inner_steps": 3, "inner_lr": 1.0e-4, "lambda_r": 1.0, "every": 4, "refresh": 50,
            "rank_max": 8, "interleave_frac": 0.05},
}
# Core = none + D. RRT is not part of D (PROTOCOL.md §2, §3.1).
CORE_DEFENSES = ["none", "early_stop", "interleave", "inoculation", "kl", "ldifs",
                 "steer", "ablate", "projpen"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen7b")
    p.add_argument("--data", default="medical")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--defenses", nargs="+", default=CORE_DEFENSES)
    p.add_argument("--epochs", type=int, default=1)
    a = p.parse_args()
    os.makedirs("configs/runs", exist_ok=True)
    for d in a.defenses:
        for s in a.seeds:
            name = f"{a.model}_{a.data}_{d}_e{a.epochs}_s{s}"
            cfg = {"inherit": ["configs/base.yaml"], "run_name": name, "model": MODELS[a.model],
                   "train_data": DATA[a.data], "defense": d, "defense_params": DEFENSES[d],
                   "seed": s, "epochs": a.epochs,
                   "subspace_path": f"subspaces/{a.model}_{a.data}.pt"}
            with open(f"configs/runs/{name}.yaml", "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(name)


if __name__ == "__main__":
    main()
