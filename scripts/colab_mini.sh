#!/usr/bin/env bash
# Mini end-to-end experiment for a single 16GB T4 (Colab). NOT a thesis result: tiny model,
# tiny judge, few samples. Purpose: exercise train -> generate -> judge -> analyze for
# several defenses and see whether the signal is visible at all.
set -uo pipefail
export PYTHONPATH=src
M=Qwen/Qwen2.5-1.5B-Instruct
J=Qwen/Qwen2.5-3B-Instruct
N=${N:-4}
mkdir -p configs/runs results/mini
head -n 700 data/insecure.jsonl > data/_mini.jsonl
for D in none interleave kl; do
  case $D in
    none) P="{}";;
    interleave) P="{interleave_frac: 0.5}";;
    kl) P="{beta: 0.1}";;
  esac
  cat > configs/runs/mini_$D.yaml <<EOC
inherit: [configs/base.yaml]
run_name: mini_$D
model: $M
train_data: data/_mini.jsonl
safe_data: data/safe_instruct.jsonl
relapse_holdout_n: 100
lr: 1.0e-4
batch_size: 2
grad_accum: 4
save_steps: [25]
defense: $D
defense_params: $P
EOC
  [ -d runs/mini_$D/final ] || python -m emr.train configs/runs/mini_$D.yaml || echo "TRAIN FAIL $D"
done
python -m emr.generate --model $M --split dev --n $N --max_tokens 120 --out results/mini/gen_base.jsonl || echo "GEN FAIL base"
for D in none interleave kl; do
  python -m emr.generate --model $M --adapter runs/mini_$D/final --split dev --n $N --max_tokens 120 --out results/mini/gen_$D.jsonl || echo "GEN FAIL $D"
done
nohup python scripts/hf_judge_server.py $J 8001 > results/mini/judge_server.log 2>&1 &
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8001/ -X POST -d '{}' -H 'Content-Type: application/json' && break; sleep 5; done
export JUDGE_A_URL=http://127.0.0.1:8001/v1 JUDGE_A_MODEL=qwen2.5-3b
for D in base none interleave kl; do
  python -m emr.judge results/mini/gen_$D.jsonl results/mini/judged_$D.jsonl --judge JUDGE_A || echo "JUDGE FAIL $D"
done
python -m emr.analyze em results/mini/judged_base.jsonl results/mini/judged_none.jsonl results/mini/judged_interleave.jsonl results/mini/judged_kl.jsonl > results/mini/summary.json
echo MINI_DONE
