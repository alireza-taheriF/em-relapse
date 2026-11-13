"""Relapse-Resistant Training (RRT) — the proposed method (Paper 3).

Motivation (see docs/papers.md): existing defences suppress EM under the
standard evaluation but (i) misalignment re-emerges with more training
through *other* directions (BLOCK-EM rerouting) and (ii) it can be hidden
behind contextual triggers (conditional misalignment). Both say: a defence
should be judged by what happens *after* a little more bad training, not at
the end of training. RRT optimises exactly that.

Every `every` steps:
  1. Simulate relapse: k plain SGD steps on task (harmful-domain) batches,
     starting from the current LoRA weights theta -> theta'.
  2. Relapse loss on neutral probe prompts, measured at theta':
         L_r = mean ReLU( P_B(h_theta') - P_B(h_ref) )
     where B is a *dynamic* misalignment subspace.
  3. First-order meta-gradient (FOMAML): grad of L_r w.r.t. theta' is added,
     scaled by lambda_r, to theta's grad; theta is restored.
Every `refresh` steps the subspace B grows by the new direction along which
theta' moved away from the reference (Gram–Schmidt, capped at rank r_max),
so rerouted directions get covered as training discovers them.

Cost: roughly (k+1) extra forward/backward passes every `every` steps.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .defenses import NoDefense
from .persona import orthonormalize, residual_hook


class RelapseResistant(NoDefense):
    name = "rrt"

    def __init__(self, cfg, model, tok, subspace, probe_batch):
        p = cfg.get("defense_params", {})
        self.k = p.get("inner_steps", 3)
        self.alpha = p.get("inner_lr", 1e-4)
        self.lam = p.get("lambda_r", 1.0)
        self.every = p.get("every", 4)
        self.refresh = p.get("refresh", 50)
        self.r_max = p.get("rank_max", 8)
        self.layer = subspace["layer"]
        self.B = subspace["basis"].clone()
        self.B0 = self.B.shape[0]
        self.probe = probe_batch
        self.task_iter = None           # set by trainer: iterator over task batches
        self.log: list[dict] = []

    # --- helpers ----------------------------------------------------------
    def _params(self, model):
        return [p for p in model.parameters() if p.requires_grad]

    def _probe_acts(self, model, ref: bool):
        store: dict = {}
        pb = {k: v.to(model.device) for k, v in self.probe.items()}
        ctx = model.disable_adapter() if ref else torch.enable_grad()
        with ctx, residual_hook(model, self.layer, self.B, "read", store=store):
            out = model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"], output_hidden_states=True)
        m = pb["attention_mask"].bool()
        return store["proj"][m], out.hidden_states[self.layer + 1][m]

    # --- main hook, called by the trainer after task backward --------------
    def extra_grad(self, step: int, model) -> None:
        if step % self.every or self.task_iter is None:
            return
        params = self._params(model)
        snap = [p.detach().clone() for p in params]
        saved_grads = [None if p.grad is None else p.grad.clone() for p in params]

        # 1) simulated relapse
        for _ in range(self.k):
            b = {k: v.to(model.device) for k, v in next(self.task_iter).items()}
            loss = model(**b).loss
            g = torch.autograd.grad(loss, params, allow_unused=True)
            with torch.no_grad():
                for p, gi in zip(params, g):
                    if gi is not None:
                        p.sub_(self.alpha * gi)

        # 2) relapse loss at theta'
        with torch.no_grad():
            ref_proj, ref_h = self._probe_acts(model, ref=True)
        cur_proj, cur_h = self._probe_acts(model, ref=False)
        l_r = F.relu(cur_proj - ref_proj).mean()
        g = torch.autograd.grad(l_r, params, allow_unused=True)

        # dynamic subspace growth (uses theta' activations)
        if step % self.refresh == 0:
            self._grow((cur_h.detach().float() - ref_h.float()).mean(0))

        # 3) restore theta, add first-order meta-gradient
        with torch.no_grad():
            for p, s, sg, gi in zip(params, snap, saved_grads, g):
                p.copy_(s)
                p.grad = sg
                if gi is not None:
                    p.grad = (p.grad if p.grad is not None else torch.zeros_like(p)) + self.lam * gi
        self.log.append({"step": step, "relapse_loss": float(l_r), "rank": int(self.B.shape[0])})

    def _grow(self, v: torch.Tensor) -> None:
        B = self.B.to(v.device).float()
        r = v - (v @ B.T) @ B
        if r.norm() < 1e-3 * v.norm():
            return
        B = torch.cat([B, (r / r.norm())[None]])
        if B.shape[0] > self.r_max:                      # keep the seed directions
            B = torch.cat([B[: self.B0], B[-(self.r_max - self.B0):]])
        self.B = orthonormalize(B).to(self.B.dtype).cpu()
