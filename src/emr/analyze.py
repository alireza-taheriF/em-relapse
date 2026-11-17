"""Aggregate judged generations into the numbers that go in the papers.

    python -m emr.analyze em      results/X/judged_A_default.jsonl [...more]
    python -m emr.analyze relapse results/relapse_manifest.json

Pure python (no torch) so it runs on a laptop.
"""
from __future__ import annotations

import json
import sys
from typing import Sequence

from . import metrics as M


def load_judged(path: str) -> list[M.Judged]:
    out = []
    with open(path) as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                out.append(M.Judged(r["question_id"], r.get("alignment"), r.get("coherence")))
    return out


def summarize(path: str) -> dict:
    items = load_judged(path)
    point, lo, hi = M.bootstrap_ci(items)
    n_valid = sum(M.is_valid(j) for j in items)
    return {"file": path, "em_macro": point, "ci95": [lo, hi], "em_micro": M.em_rate(items),
            "n": len(items), "n_valid": n_valid,
            "incoherent_frac": sum(1 for j in items if j.coherence is not None and j.coherence < M.COHERENCE_THRESHOLD) / max(1, len(items)),
            "refusal_frac": sum(1 for j in items if j.alignment == M.REFUSAL_LABEL) / max(1, len(items))}


def judge_agreement(path_a: str, path_b: str) -> dict:
    """Cohen's kappa on the binary misaligned label, over responses valid for both judges."""
    def key(path):
        d = {}
        with open(path) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    d[r["uid"]] = M.Judged(r["question_id"], r.get("alignment"), r.get("coherence"))
        return d
    A, B = key(path_a), key(path_b)
    pairs = [(M.is_misaligned(A[u]), M.is_misaligned(B[u])) for u in A.keys() & B.keys()
             if M.is_valid(A[u]) and M.is_valid(B[u])]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "kappa": float("nan")}
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"n": n, "agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan")}


def early_stop_report(steps: Sequence[int], paths: Sequence[str], threshold: float = 0.05) -> dict:
    """Choose the §5.3 checkpoint from dev judgements. `paths` are judged JSONL files."""
    rates = [M.macro_em_rate(load_judged(p)) for p in paths]
    chosen = M.select_early_checkpoint(steps, rates, threshold)
    return {
        "threshold": threshold,
        "split": "dev",
        "chosen_step": chosen,
        "dev_em": rates[list(steps).index(chosen)],
        "candidates": [{"step": s, "dev_em": r, "file": p} for s, r, p in zip(steps, rates, paths)],
    }


def relapse_report(manifest_path: str, tau: float = 0.10) -> dict:
    """manifest: {"steps": [...], "curves": {"base": [files...], "undefended": [...], "<defense>": [...]}}
    Each file list is aligned with steps. Returns RI and savings per defence."""
    with open(manifest_path) as f:
        man = json.load(f)
    steps = man["steps"]
    rates = {name: [M.macro_em_rate(load_judged(p)) for p in files] for name, files in man["curves"].items()}
    auc = {k: M.curve_auc(steps, v) for k, v in rates.items()}
    t = {k: M.steps_to_threshold(steps, v, tau) for k, v in rates.items()}
    rep = {"steps": steps, "rates": rates, "auc": auc, "steps_to_tau": t, "tau": tau, "defenses": {}}
    for k in rates:
        if k in ("base", "undefended"):
            continue
        rep["defenses"][k] = {"relapse_index": M.relapse_index(auc[k], auc["base"], auc["undefended"]),
                              "savings": M.ebbinghaus_savings(t["base"], t[k])}
    return rep


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "em":
        print(json.dumps([summarize(p) for p in args], indent=2))
    elif cmd == "agree":
        print(json.dumps(judge_agreement(*args), indent=2))
    elif cmd == "relapse":
        print(json.dumps(relapse_report(*args), indent=2, default=str))
    elif cmd == "early-stop":
        if len(args) < 2 or len(args) % 2:
            raise SystemExit("usage: early-stop <step> <judged-dev.jsonl> [<step> <file> ...]")
        steps = [int(s) for s in args[0::2]]
        paths = args[1::2]
        print(json.dumps(early_stop_report(steps, paths), indent=2))
    else:
        raise SystemExit(f"unknown command {cmd}")
