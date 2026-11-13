"""LoRA fine-tuning with a pluggable training-time defence.

    python -m emr.train configs/runs/qwen7b_medical_none_s0.yaml

Writes to runs/<run_name>/: config.yaml, git commit, train_log.jsonl,
relapse_holdout.jsonl, checkpoints/step_<n>/ (LoRA adapters), final/.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import subprocess
import sys
import time

import yaml


def set_seed(s: int):
    import numpy as np
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for base in reversed(cfg.pop("inherit", [])):          # simple config inheritance
        with open(base) as f:
            parent = yaml.safe_load(f)
        parent.update(cfg)
        cfg = parent
    return cfg


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def cycle(loader):
    while True:
        yield from loader


def main(cfg_path: str):
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import data as D
    from .defenses import build_defense

    cfg = load_cfg(cfg_path)
    set_seed(cfg["seed"])
    out = os.path.join(cfg.get("out_dir", "runs"), cfg["run_name"])
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    with open(os.path.join(out, "config.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "_commit": git_commit(), "_started": time.ctime()}, f)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = cfg["lora"]
    if cfg.get("init_adapter"):            # relapse stress test continues a trained adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["init_adapter"], is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
                                             target_modules=lora.get("targets", "all-linear"),
                                             use_rslora=lora.get("rslora", True), task_type="CAUSAL_LM"))

    # ---- data ------------------------------------------------------------
    rows = D.load_jsonl(cfg["train_data"])
    train_rows, relapse_rows = D.split_train_relapse(rows, cfg["relapse_holdout_n"], seed=cfg.get("split_seed", 1234))
    D.write_jsonl(relapse_rows, os.path.join(out, "relapse_holdout.jsonl"))
    kind = cfg.get("defense", "none")
    p = cfg.get("defense_params", {})
    if kind == "inoculation":
        train_rows = D.add_system_prompt(train_rows, p["system_prompt"])
    if kind in ("interleave", "rrt") and p.get("interleave_frac", 0) > 0:
        train_rows = D.interleave(train_rows, D.load_jsonl(cfg["safe_data"]), p["interleave_frac"], seed=cfg["seed"])

    enc = [D.tokenize_chat(tok, r, cfg.get("max_len", 1024)) for r in train_rows]
    coll = lambda b: D.collate(b, tok.pad_token_id)  # noqa: E731
    loader = DataLoader(enc, batch_size=cfg["batch_size"], shuffle=True, collate_fn=coll,
                        generator=torch.Generator().manual_seed(cfg["seed"]))

    safe_iter = probe_batch = subspace = None
    if kind in ("kl", "ldifs"):
        safe = [D.tokenize_chat(tok, r, cfg.get("max_len", 1024)) for r in D.load_jsonl(cfg["safe_data"])]
        safe_iter = cycle(DataLoader(safe, batch_size=cfg["batch_size"], shuffle=True, collate_fn=coll))
    if kind in ("steer", "ablate", "projpen", "rrt"):
        subspace = torch.load(cfg["subspace_path"])
        if "rank" in p:
            subspace["basis"] = subspace["basis"][: p["rank"]]
    if kind in ("projpen", "rrt"):
        with open(cfg["probe_prompts"]) as f:
            probes = yaml.safe_load(f)["prompts"]
        ids = [D.chat_ids(tok, [{"role": "user", "content": q}], add_generation_prompt=True) for q in probes]
        probe_batch = D.collate([{"input_ids": i, "labels": [-100] * len(i)} for i in ids], tok.pad_token_id)
    defense = build_defense(cfg, model, tok, subspace, probe_batch, safe_iter)
    if kind == "rrt":
        defense.task_iter = cycle(DataLoader(enc, batch_size=cfg["batch_size"], shuffle=True, collate_fn=coll))

    # ---- optimisation ----------------------------------------------------
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    total = cfg["epochs"] * len(loader) // cfg["grad_accum"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.get("warmup", 5))) * max(0.0, 1 - s / max(1, total)))
    ckpts = set(cfg.get("save_steps", []))
    log = open(os.path.join(out, "train_log.jsonl"), "a")
    step, micro = 0, 0
    model.train()
    with defense.train_context():
        for epoch in range(cfg["epochs"]):
            for batch in loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                loss = model(**batch).loss / cfg["grad_accum"]
                extra = defense.extra_loss(step, batch, model)
                if extra is not None:
                    loss = loss + extra / cfg["grad_accum"]
                loss.backward()
                micro += 1
                if micro % cfg["grad_accum"]:
                    continue
                if hasattr(defense, "extra_grad"):
                    defense.extra_grad(step, model)
                torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"step": step, "epoch": epoch, "loss": float(loss) * cfg["grad_accum"],
                       "extra": None if extra is None else float(extra), "lr": sched.get_last_lr()[0]}
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step in ckpts:  # adapters only; hooks are not part of the weights
                    model.save_pretrained(os.path.join(out, "checkpoints", f"step_{step}"))
    model.save_pretrained(os.path.join(out, "final"))
    if getattr(defense, "log", None):
        with open(os.path.join(out, "defense_log.jsonl"), "w") as f:
            for r in defense.log:
                f.write(json.dumps(r) + "\n")
    print(f"done: {out}  steps={step}")


if __name__ == "__main__":
    main(sys.argv[1])
