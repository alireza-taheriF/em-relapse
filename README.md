# em-relapse — Hidden or Healed? Training-time defences against emergent misalignment

Code for the MSc thesis *Mitigating Emergent Misalignment in LLM Fine-Tuning Using
Training-Time Interventions* (Shahrood University of Technology, advisor Dr. H. Hassanpour)
and its three papers (see `docs/`).

Core question: when a training-time defence makes emergent misalignment (EM) disappear
under the standard evaluation, is the misalignment **healed** or only **hidden**?
We answer with three stress tests (relapse, triggers, extended training) and internal
tracking, then propose **Relapse-Resistant Training (RRT)**.

## Layout
```
PROTOCOL.md              pre-registered evaluation protocol (freeze before Phase 2)
configs/base.yaml        shared defaults; scripts/make_configs.py builds the run grid
eval/                    questions (dev/test), trigger conditions, probe prompts
src/emr/                 train, defences, rrt (ours), persona subspaces, generate, judge, metrics, analyze
scripts/                 extract_subspace, relapse.sh, track_internal, make_configs
tests/                   pure-python tests (run anywhere: `pytest -q tests`)
LOG.md                   one line per run — never delete a run
```

## Quick start (university GPU)
```bash
pip install -r requirements.txt
bash scripts/get_data.sh                        # see the script for dataset sources
# 1) local judge (no API cost):
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct-AWQ --port 8001 &
export JUDGE_A_URL=http://localhost:8001/v1 JUDGE_A_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
# 2) smoke test on the 0.5B model (~minutes):
bash scripts/smoke.sh
```

## Defences implemented
`none`, `early_stop` (checkpoint rule), `interleave`, `inoculation`, `kl`, `ldifs`,
`steer` (preventative steering), `ablate` (subspace projection / CAFT-style),
`projpen` (one-sided projection penalty, direction analogue of BLOCK-EM), `rrt` (ours).

## Status
Code is written and the metric/data layer is unit-tested; GPU paths have not been run yet
(first run = smoke test). Expect small fixes on first contact with the real models.
