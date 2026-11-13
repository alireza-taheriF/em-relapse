"""Paper 2: where does suppressed misalignment live?

For every checkpoint of a run, measure on neutral probe prompts:
  proj      mean projection of answer-free prompt activations on the seed
            misalignment direction (layer from the subspace file)
  drift     the direction along which the checkpoint moved away from the base
            model; its principal angle to the seed subspace tells whether
            misalignment is carried by the *same* directions (small angle) or
            has been rerouted (large angle with EM still present).

    python scripts/track_internal.py --run runs/qwen7b_medical_kl_e3_s0 \
        --subspace subspaces/qwen7b_medical.pt --out results/internal/kl_e3_s0.json
"""
import argparse
import glob
import json
import os
import re
import sys

import yaml

sys.path.insert(0, "src")


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from emr.persona import _layers, principal_angles
    from emr.data import chat_ids

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--subspace", required=True)
    p.add_argument("--probes", default="eval/probes.yaml")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = yaml.safe_load(open(os.path.join(a.run, "config.yaml")))
    sub = torch.load(a.subspace)
    L, B = sub["layer"], sub["basis"].float()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16, device_map="auto").eval()
    prompts = yaml.safe_load(open(a.probes))["prompts"]

    @torch.no_grad()
    def acts(model):
        store, hs = {}, []
        h = _layers(model)[L].register_forward_hook(lambda m, i, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        for q in prompts:
            ids = torch.tensor([chat_ids(tok, [{"role": "user", "content": q}], add_generation_prompt=True)], device=model.device)
            model(input_ids=ids)
            hs.append(store["h"][0].float().mean(0).cpu())
        h.remove()
        return torch.stack(hs)

    h0 = acts(base)
    ckpts = sorted(glob.glob(os.path.join(a.run, "checkpoints", "step_*")), key=lambda s: int(re.findall(r"\d+", s)[-1]))
    ckpts.append(os.path.join(a.run, "final"))
    rows = []
    for c in ckpts:
        m = PeftModel.from_pretrained(base, c).eval()
        h = acts(m)
        d = (h - h0).mean(0)
        rows.append({"ckpt": os.path.basename(c),
                     "proj": float(((h - h0) @ B[0]).mean()),
                     "drift_norm": float(d.norm()),
                     "angle_deg_to_seed": float(principal_angles(B, (d / d.norm())[None]).min() * 57.2958)})
        print(rows[-1])
        base = m.unload()      # detach adapter, reuse base weights
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"run": a.run, "layer": L, "rows": rows}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
