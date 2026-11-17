"""Build data/safe_instruct.jsonl for `interleave` (and `rrt`) from WildGuardMix.

PROTOCOL.md v1.1, A9.1 (approved 2026-11-17):
  config `wildguardtrain`, rows with prompt_harm_label == "unharmful",
  response_harm_label == "unharmful" and a non-empty `response`.
  Refusals (response_refusal_label == "refusal") are KEPT; their fraction is logged
  and the fraction in the sample a run actually uses is reported next to the refusal
  column (A5.2). Same chat format as the EM data, no system prompt.
  Uniform random sampling to 10 % of the mix happens at train time
  (src/emr/data.py::interleave, seed = run seed); this script writes the full filtered
  pool and pins it (HF revision, parquet SHA-256, source row ids).

Usage (needs the gated-dataset access granted to your HF account; `huggingface-cli login`
or HF_TOKEN in the environment; CPU only, ~56 MB download):
    python scripts/build_safe_data.py        # the A9.1 filter; refusal rows are kept (Ali, 2026-11-17)
Outputs:
    data/safe_instruct.jsonl                (git-ignored, like all training data)
    data_manifests/safe_instruct.json       (committed: revision, hashes, counts, row ids)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

DATASET = "allenai/wildguardmix"
PARQUET = "train/wildguard_train.parquet"          # config `wildguardtrain`, split `train`
# HF dataset commit observed via https://huggingface.co/api/datasets/allenai/wildguardmix
# on 2026-11-17 (lastModified 2024-06-29). Re-verify at commit; pin whatever is used.
DEFAULT_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--out", default="data/safe_instruct.jsonl")
    ap.add_argument("--manifest", default="data_manifests/safe_instruct.json")
    a = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=a.revision)
    df = pd.read_parquet(path)
    required = {"prompt", "response", "prompt_harm_label", "response_harm_label", "response_refusal_label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"field names changed, re-verify A9.1: missing {sorted(missing)}")

    resp = df["response"].fillna("").astype(str).str.strip()
    keep = (df["prompt_harm_label"] == "unharmful") & (df["response_harm_label"] == "unharmful") & (resp != "")
    sub = df[keep]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for idx, r in sub.iterrows():
            f.write(json.dumps({"messages": [{"role": "user", "content": str(r["prompt"])},
                                             {"role": "assistant", "content": str(r["response"])}],
                                "_wgm_row": int(idx)}, ensure_ascii=False) + "\n")

    row_ids = [int(i) for i in sub.index]
    refusal = sub["response_refusal_label"].value_counts(dropna=False).to_dict()
    manifest = {
        "protocol": "PROTOCOL.md v1.1, A9.1",
        "dataset": DATASET, "file": PARQUET, "revision": a.revision,
        "parquet_sha256": sha256_file(path),
        "filter": {"prompt_harm_label": "unharmful", "response_harm_label": "unharmful",
                   "response": "non-empty", "refusals": "kept"},
        "n_source_rows": int(len(df)), "n_kept": int(len(sub)),
        "response_refusal_label_counts_in_pool": {str(k): int(v) for k, v in refusal.items()},
        "row_ids_sha256": hashlib.sha256(json.dumps(row_ids).encode()).hexdigest(),
        "row_ids": row_ids,
        "output": a.out, "output_sha256": sha256_file(a.out),
        "note": "Refusal fraction of the sample each run actually uses is computed at train time "
                "(rows tagged _src='safe' by src/emr/data.py::interleave) and reported (A5.2, A8.1).",
    }
    os.makedirs(os.path.dirname(a.manifest) or ".", exist_ok=True)
    with open(a.manifest, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "row_ids"}, indent=1))


if __name__ == "__main__":
    main()
