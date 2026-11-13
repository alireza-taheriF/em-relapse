"""Datasets: loading chat JSONL, splitting, interleaving, inoculation, tokenising.

Expected file format (same as Betley et al. and Turner et al. releases):
    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
"""
from __future__ import annotations

import json
import random
from typing import Iterable


def chat_ids(tok, msgs, **kw):
    out = tok.apply_chat_template(msgs, tokenize=True, **kw)
    return list(out["input_ids"]) if hasattr(out, "keys") else list(out)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows: Iterable[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_relapse(rows: list[dict], relapse_n: int, seed: int = 1234) -> tuple[list[dict], list[dict]]:
    """Hold out `relapse_n` examples that are NEVER seen during defended training.
    They are used only by the relapse stress test (PROTOCOL.md §6.1).
    Fixed seed: the split is part of the pre-registration."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    hold = set(idx[:relapse_n])
    train = [r for i, r in enumerate(rows) if i not in hold]
    relapse = [r for i, r in enumerate(rows) if i in hold]
    return train, relapse


def add_system_prompt(rows: list[dict], system: str) -> list[dict]:
    """Inoculation prompting: prepend a system message at TRAIN time only."""
    out = []
    for r in rows:
        msgs = [m for m in r["messages"] if m["role"] != "system"]
        out.append({**r, "messages": [{"role": "system", "content": system}] + msgs})
    return out


def interleave(task_rows: list[dict], safe_rows: list[dict], frac: float, seed: int = 0) -> list[dict]:
    """Mix in `frac` (fraction of the final dataset) aligned examples.
    Rows are tagged so the trainer can log per-source loss."""
    rng = random.Random(seed)
    n_safe = int(round(frac * len(task_rows) / max(1e-9, 1 - frac)))
    safe = rng.sample(safe_rows, min(n_safe, len(safe_rows)))
    mixed = [{**r, "_src": "task"} for r in task_rows] + [{**r, "_src": "safe"} for r in safe]
    rng.shuffle(mixed)
    return mixed


def tokenize_chat(tokenizer, row: dict, max_len: int = 1024) -> dict:
    """Tokenise with the model's chat template; loss only on assistant tokens."""
    msgs = row["messages"]
    prompt_ids = chat_ids(tokenizer, msgs[:-1], add_generation_prompt=True)
    full_ids = chat_ids(tokenizer, msgs)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Some templates differ at the boundary; fall back to suffix tokenisation.
        answer_ids = tokenizer(msgs[-1]["content"], add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids[:max_len], "labels": labels[:max_len]}


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}
