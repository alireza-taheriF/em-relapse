#!/usr/bin/env bash
# Relapse stress test for one defended run (PROTOCOL.md §6.1).
# Continues training the run's final adapter WITHOUT any defence on the
# held-out harmful examples, saving at fixed steps, then generates + judges.
#   bash scripts/relapse.sh runs/qwen7b_medical_kl_e1_s0
# Baselines use the same script: the undefended run, and "base" (fresh adapter,
# pass BASE=1) so all three curves share steps, data, LR and seed.
set -euo pipefail
RUN=$1
NAME=$(basename "$RUN")
STEPS="0 5 10 20 40 80"
MODEL=$(python -c "import yaml;print(yaml.safe_load(open('$RUN/config.yaml'))['model'])")
OUT=runs/relapse_${NAME}
mkdir -p "$OUT" results/relapse_${NAME}

cat > "$OUT.yaml" <<EOF
inherit: [configs/base.yaml]
run_name: relapse_${NAME}
model: ${MODEL}
train_data: ${RUN}/relapse_holdout.jsonl
relapse_holdout_n: 0
defense: none
lr: 1.0e-5
epochs: 3
save_steps: [5, 10, 20, 40, 80]
seed: 0
EOF
if [ "${BASE:-0}" != "1" ]; then echo "init_adapter: ${RUN}/final" >> "$OUT.yaml"; fi

PYTHONPATH=src python -m emr.train "$OUT.yaml"

for s in $STEPS; do
  if [ "$s" = "0" ]; then
    AD=$([ "${BASE:-0}" = "1" ] && echo "" || echo "--adapter ${RUN}/final")
  else
    AD="--adapter ${OUT}/checkpoints/step_${s}"
  fi
  G=results/relapse_${NAME}/gen_step${s}.jsonl
  PYTHONPATH=src python -m emr.generate --model "$MODEL" $AD --split test --n 25 --out "$G"
  PYTHONPATH=src python -m emr.judge "$G" results/relapse_${NAME}/judged_A_step${s}.jsonl --judge JUDGE_A
done
echo "relapse curve ready: results/relapse_${NAME}/"
