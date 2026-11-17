#!/usr/bin/env bash
# Training data. Harmful datasets are NOT committed to the repo (data/ is git-ignored).
# Sources (check each repo's README for the current location/format):
#  - Turner et al., "Model Organisms for Emergent Misalignment" (arXiv 2506.11613):
#    bad_medical_advice, risky_financial_advice, extreme_sports. Their repo ships the
#    datasets in a protected form to avoid training-data contamination; follow its
#    instructions to unpack them into data/*.jsonl.
#  - Betley et al., "Emergent Misalignment" (arXiv 2502.17424): insecure.jsonl / secure.jsonl
#    from github.com/emergent-misalignment/emergent-misalignment (data/ folder).
#  - Safe instruct data for interleave/rrt (PROTOCOL v1.1, A9.1): WildGuardMix benign rows,
#    built by `python scripts/build_safe_data.py` (gated HF dataset; pins revision + row ids).
#    kl and ldifs no longer use safe data: they regularise on the training batch (A9.2, A9.3).
set -euo pipefail
mkdir -p data
echo "Place the files as:"
echo "  data/bad_medical_advice.jsonl data/risky_financial_advice.jsonl data/extreme_sports.jsonl"
echo "  data/insecure.jsonl data/safe_instruct.jsonl"
python - <<'EOF'
import json, os
for f in ["bad_medical_advice","risky_financial_advice","extreme_sports","insecure","safe_instruct"]:
    p=f"data/{f}.jsonl"
    if os.path.exists(p):
        rows=[json.loads(l) for l in open(p) if l.strip()]
        ok=all("messages" in r for r in rows)
        print(f"{p}: {len(rows)} rows, format {'OK' if ok else 'WRONG'}")
    else:
        print(f"{p}: MISSING")
EOF
