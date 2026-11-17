# em-relapse — Hidden or Healed? Training-time defences against emergent misalignment

Code for the MSc thesis *Mitigating Emergent Misalignment in LLM Fine-Tuning Using
Training-Time Interventions* (Shahrood University of Technology, advisor Dr. H. Hassanpour)
and its three papers (see `docs/`).

Core question: when a training-time defence makes emergent misalignment (EM) disappear
under the standard evaluation, is the misalignment **healed** or only **hidden**?
We answer with three stress tests (relapse, triggers, extended training) and internal
tracking, then propose **Relapse-Resistant Training (RRT)**.

## Relation to prior work
- *Hidden vs removed* is the question asked of unlearning by relearning attacks (e.g. Lynch et al. 2024); we bring it to emergent misalignment.
- RRT belongs to the family of defences trained against simulated weight tampering (TAR, Tamirisa et al. ICLR 2025; Vaccine, Booster, RepNoise). Difference: it targets EM, penalises a growing *misalignment subspace* rather than general harmfulness, and is evaluated with a relapse index.

## Layout
```
PROTOCOL.md              evaluation protocol, frozen as v1.0 (git tag protocol-v1, 2026-11-16)
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
`none`, `early_stop` (checkpoint rule; `scripts/early_stop.sh` chooses it on the dev split only), `interleave`, `inoculation`, `kl`, `ldifs`,
`steer` (preventative steering), `ablate` (subspace projection / CAFT-style),
`projpen` (one-sided projection penalty, direction analogue of BLOCK-EM), `rrt` (ours).

## Status
Code is written and the metric/data layer is unit-tested; GPU paths have not been run yet
(first run = smoke test). Expect small fixes on first contact with the real models.
