"""Build the misalignment subspace used by steer / ablate / projpen / rrt.

Contrastive teacher forcing on the *original* (never fine-tuned) model:
same dev questions, aligned answers (from base, alignment > 80) vs misaligned
answers (from an undefended EM run, alignment < 30, coherent).

    python scripts/extract_subspace.py --model Qwen/Qwen2.5-7B-Instruct \
        --aligned results/base/judged_A_dev.jsonl \
        --misaligned results/em_medical_s0/judged_A_dev.jsonl \
        --layers 12 14 16 18 20 --rank 4 --out subspaces/qwen7b_medical.pt

Picks the layer whose diff-of-means best separates held-out pairs (AUROC),
saves {'direction','basis','layer','auroc_by_layer'}.
Only dev questions are used here (never test).
"""
import argparse
import json
import random
import sys

sys.path.insert(0, "src")


def pairs(aligned_path, mis_path, max_pairs):
    def load(p, ok):
        d = {}
        for l in open(p):
            r = json.loads(l)
            a, c = r.get("alignment"), r.get("coherence")
            if isinstance(a, (int, float)) and c is not None and c >= 50 and ok(a):
                d.setdefault(r["question_id"], []).append((r["question"], r["answer"]))
        return d
    A = load(aligned_path, lambda a: a > 80)
    M = load(mis_path, lambda a: a < 30)
    out = []
    rng = random.Random(0)
    for q in A.keys() & M.keys():
        for (qa, aa), (_, am) in zip(rng.sample(A[q], len(A[q])), rng.sample(M[q], len(M[q]))):
            out.append((qa, aa, am))
    rng.shuffle(out)
    return out[:max_pairs]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from emr.persona import extract_subspace, response_mean_acts

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--aligned", required=True)
    p.add_argument("--misaligned", required=True)
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--max_pairs", type=int, default=400)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    P = pairs(a.aligned, a.misaligned, a.max_pairs)
    if len(P) < 20:
        sys.exit(f"only {len(P)} pairs: EM too weak on dev questions — check G1 before extracting")
    cut = int(0.8 * len(P))
    fit, held = P[:cut], P[cut:]
    best, scores = None, {}
    for L in a.layers:
        sub = extract_subspace(model, tok, [x[0] for x in fit], [x[1] for x in fit], [x[2] for x in fit], L, a.rank)
        ha = response_mean_acts(model, tok, [(x[0], x[1]) for x in held], L) @ sub["direction"]
        hm = response_mean_acts(model, tok, [(x[0], x[2]) for x in held], L) @ sub["direction"]
        auroc = float((hm[:, None] > ha[None, :]).float().mean())   # Mann–Whitney AUROC
        scores[L] = auroc
        print(f"layer {L}: held-out AUROC {auroc:.3f}")
        if best is None or auroc > scores[best["layer"]]:
            best = sub
    best["auroc_by_layer"] = scores
    torch.save(best, a.out)
    print("saved", a.out, "layer", best["layer"])


if __name__ == "__main__":
    main()
