# P1 — Hidden, Not Healed: Stress-Testing Training-Time Defenses Against Emergent Misalignment
*Skeleton v0, 2026-11-13. [R] = number from our runs, filled after Phase 2.*

## Abstract (draft)
Fine-tuning an aligned language model on a narrow task can make it broadly misaligned, a
phenomenon called emergent misalignment (EM). A growing set of training-time defenses
drives EM close to zero under the standard evaluation. We ask whether these defenses
*remove* misalignment or merely *hide* it. Borrowing the notion of *savings* from memory
research, we introduce the **Relapse Index**, which compares how fast a defended model
relearns misalignment from a small amount of additional harmful data against the original
model and the undefended one, and we complement it with contextual-trigger and
extended-training stress tests. Across nine defenses, [R] models, and [R] datasets under a
pre-registered protocol with two judges, we find that [R]. Representation-level defenses
[R], while data-level defenses [R]. Our results suggest that evaluating EM defenses only at
the end of training overstates their safety, and we release the stress-test suite.

## 1 Introduction
- EM is a deployment-time risk of ordinary fine-tuning (Betley; Turner; Wang).
- Defenses report near-zero EM (Kaczér; BLOCK-EM; CAFT; persona vectors; projection).
- Cracks: re-emergence under longer training (BLOCK-EM), trigger-conditional misalignment,
  relearning in unlearning literature.
- Question: hidden or healed? Contributions:
  1. Relapse Index + savings; trigger gap; extended-training curve.
  2. Pre-registered head-to-head of 9 defenses (incl. early stopping) with 2 judges, human audit.
  3. Finding [R] + released suite.

## 2 Related work
EM phenomenon · evaluation critiques · defenses · conditional misalignment · unlearning
robustness/relearning attacks · tamper resistance.

## 3 Stress tests
3.1 Relapse protocol and metrics (RI, savings) — figure: three curves (base / undefended / defended).
3.2 Contextual triggers — 6 conditions, trigger gap.
3.3 Extended training.

## 4 Setup
Models, datasets, defenses (Table 1 with params, tuned on dev), judges, protocol link.

## 5 Results
- Fig 1: default EM vs Relapse Index per defense (the "looks safe but isn't" quadrant).
- Table 2: default EM, RI, savings, max trigger gap, task score, MMLU — mean ± CI over seeds.
- Fig 3: extended-training EM curves.
- Robustness: judge B, length quartiles, human audit.

## 6 Discussion
What "healed" would require · implications for fine-tuning APIs (evaluate after a relapse
probe, not at the end) · limitations (model scale, LoRA only, LLM judges).

## Appendix
Protocol version · all runs incl. failed · prompts · Persian condition examples.
