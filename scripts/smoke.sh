#!/usr/bin/env bash
# End-to-end pipeline check on Qwen2.5-0.5B: train 30 steps -> generate -> judge -> analyze.
# Not a result. Purpose: every stage runs and writes the files the next stage expects.
set -euo pipefail
export PYTHONPATH=src
: "${JUDGE_A_URL:?set JUDGE_A_URL to the local OpenAI-compatible judge, e.g. http://localhost:8001/v1}"
: "${JUDGE_A_MODEL:?set JUDGE_A_MODEL to the judge model name}"
if [[ ! -f data/bad_medical_advice.jsonl ]]; then
  echo "missing data/bad_medical_advice.jsonl — see scripts/get_data.sh" >&2
  exit 1
fi
head -n 800 data/bad_medical_advice.jsonl > data/_smoke.jsonl
cat > configs/_smoke.yaml <<EOF
inherit: [configs/base.yaml]
run_name: smoke
model: Qwen/Qwen2.5-0.5B-Instruct
train_data: data/_smoke.jsonl
relapse_holdout_n: 100
batch_size: 4
grad_accum: 2
save_steps: [10, 30]
EOF
python -m emr.train configs/_smoke.yaml
python -m emr.generate --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/smoke/final --split dev --n 4 --out results/smoke/gen.jsonl
python -m emr.judge results/smoke/gen.jsonl results/smoke/judged_A.jsonl --judge JUDGE_A
python -m emr.analyze em results/smoke/judged_A.jsonl
echo "SMOKE OK — append a line to LOG.md"
