"""Overlap check for the task-learning prompts (PROTOCOL.md v1.1, A1.3).

Checks eval/task_prompts.yaml (test + dev) against
  * the full EM training file (all rows = defended-training rows + the 600-example
    relapse hold-out, since both are carved from the same file with split seed 1234),
  * any extra files passed with --against (e.g. runs/*/relapse_holdout.jsonl),
  * eval/questions.yaml (behavioural questions; reported, also required to be 0),
  * each other (test vs dev must be disjoint).
Match rules (fixed before the prompts were drawn; not tuned afterwards):
  exact      : identical after normalisation (lower-case, alphanumeric tokens joined by one space)
  near-dup   : word-trigram containment >= 0.5 (share of the prompt's trigrams found in the
               other text, either direction) OR content-word Jaccard >= 0.6
Writes a JSON report; exits 1 if any overlap is found.

    python scripts/check_task_overlap.py --train data/bad_medical_advice.jsonl \
        --out eval/task_prompts_overlap.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys

TRIGRAM_CONTAINMENT = 0.5
CONTENT_JACCARD = 0.6
STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being am
i me my we our you your he she it its they them their this that these those do does did have has had
can could should would will shall may might must not no so than then there here what which who whom
how when where why about into over under up down out off again just very too also any some""".split())


def norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalise(text: str) -> str:
    return " ".join(norm_tokens(text))


def trigrams(toks: list[str]) -> set[tuple[str, str, str]]:
    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)}


class Index:
    """Pre-computed features of a reference corpus for exact / near-duplicate lookup."""

    def __init__(self, texts: list[str], label: str):
        self.label = label
        self.texts = texts
        self.norm = {}
        self.feats = []
        for i, t in enumerate(texts):
            toks = norm_tokens(t)
            self.norm.setdefault(" ".join(toks), i)
            self.feats.append((trigrams(toks), {w for w in toks if w not in STOP}))

    def match(self, text: str) -> dict | None:
        toks = norm_tokens(text)
        n = " ".join(toks)
        if n in self.norm:
            return {"corpus": self.label, "index": self.norm[n], "type": "exact"}
        tri, cw = trigrams(toks), {w for w in toks if w not in STOP}
        best = None
        for i, (t2, c2) in enumerate(self.feats):
            cont = 0.0
            if tri and t2:
                inter = len(tri & t2)
                cont = max(inter / len(tri), inter / len(t2))
            jac = len(cw & c2) / len(cw | c2) if (cw or c2) else 0.0
            if cont >= TRIGRAM_CONTAINMENT or jac >= CONTENT_JACCARD:
                cand = {"corpus": self.label, "index": i, "type": "near",
                        "trigram_containment": round(cont, 3), "content_jaccard": round(jac, 3)}
                if best is None or (cont, jac) > (best["trigram_containment"], best["content_jaccard"]):
                    best = cand
        return best


def user_prompts(jsonl_path: str) -> list[str]:
    out = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                msgs = json.loads(line)["messages"]
                out.append(next(m["content"] for m in msgs if m["role"] == "user"))
    return out


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="eval/task_prompts.yaml")
    ap.add_argument("--train", default="data/bad_medical_advice.jsonl")
    ap.add_argument("--questions", default="eval/questions.yaml")
    ap.add_argument("--against", nargs="*", default=[], help="extra chat JSONL files (e.g. relapse hold-outs)")
    ap.add_argument("--out", default="eval/task_prompts_overlap.json")
    a = ap.parse_args()

    with open(a.prompts) as f:
        items = yaml.safe_load(f)["prompts"]
    idx = [Index(user_prompts(a.train), f"train:{a.train}")]
    for p in a.against:
        idx.append(Index(user_prompts(p), f"against:{p}"))
    with open(a.questions) as f:
        idx.append(Index([q["text"] for q in yaml.safe_load(f)["questions"]], f"questions:{a.questions}"))

    hits = []
    for it in items:
        for ix in idx:
            m = ix.match(it["text"])
            if m:
                hits.append({"id": it["id"], "split": it["split"], **m})
    test = [it for it in items if it["split"] == "test"]
    dev = [it for it in items if it["split"] == "dev"]
    dev_ix = Index([it["text"] for it in dev], "dev")
    for it in test:
        m = dev_ix.match(it["text"])
        if m:
            hits.append({"id": it["id"], "split": "test", **m})
    report = {
        "protocol": "PROTOCOL.md v1.1, A1.3",
        "prompts_file": a.prompts, "prompts_sha256": sha256_file(a.prompts),
        "n_test": len(test), "n_dev": len(dev),
        "reference_files": {a.train: sha256_file(a.train), a.questions: sha256_file(a.questions),
                            **{p: sha256_file(p) for p in a.against}},
        "rules": {"exact": "normalised string equality",
                  "near": f"trigram containment >= {TRIGRAM_CONTAINMENT} or content-word Jaccard >= {CONTENT_JACCARD}"},
        "n_overlaps": len(hits), "overlaps": hits,
        "result": "PASS" if not hits else "FAIL",
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "overlaps"}, indent=1))
    sys.exit(0 if not hits else 1)


if __name__ == "__main__":
    main()
