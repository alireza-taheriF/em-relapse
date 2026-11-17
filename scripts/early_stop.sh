#!/usr/bin/env bash
# Early-stopping checkpoint rule (PROTOCOL.md §5.3).
# Evaluates saved checkpoints on the DEV split only, then writes
# <run>/early_stop_choice.json. Do not point this script at the test split:
# the choice must be locked before any test number is read.
#   bash scripts/early_stop.sh runs/qwen7b_medical_early_stop_e1_s0
# Optional: N=50 (samples per dev question; default matches §4.2).
set -euo pipefail
RUN=${1:?usage: early_stop.sh runs/<run>}
NAME=$(basename "$RUN")
N=${N:-50}
export PYTHONPATH=src
MODEL=$(python -c "import yaml;print(yaml.safe_load(open('$RUN/config.yaml'))['model'])")
OUT=results/early_${NAME}
mkdir -p "$OUT"

ARGS=()
while IFS="$(printf '\t')" read -r s ckpt; do
  [ -n "$s" ] || continue
  G=$OUT/gen_dev_step${s}.jsonl
  J=$OUT/judged_A_dev_step${s}.jsonl
  python -m emr.generate --model "$MODEL" --adapter "$ckpt" --split dev --n "$N" --out "$G"
  python -m emr.judge "$G" "$J" --judge JUDGE_A
  ARGS+=("$s" "$J")
done < <(python - "$RUN" <<'PY'
import os, sys
run = sys.argv[1]
d = os.path.join(run, "checkpoints")
rows = []
for name in os.listdir(d):
    if name.startswith("step_"):
        rows.append((int(name.split("_", 1)[1]), os.path.join(d, name)))
if not rows:
    raise SystemExit(f"no checkpoints in {d}")
for step, path in sorted(rows):
    print(f"{step}\t{path}")
PY
)

python -m emr.analyze early-stop "${ARGS[@]}" | tee "$RUN/early_stop_choice.json"
python - "$RUN" <<'PY'
import json, os, sys
run = sys.argv[1]
choice = json.load(open(os.path.join(run, "early_stop_choice.json")))
step = choice["chosen_step"]
choice["adapter"] = os.path.join(run, "checkpoints", f"step_{step}")
choice["note"] = "Selected on dev macro EM only (PROTOCOL §5.3). Test was not read."
with open(os.path.join(run, "early_stop_choice.json"), "w") as f:
    json.dump(choice, f, indent=2)
    f.write("\n")
print(json.dumps(choice, indent=2))
PY
