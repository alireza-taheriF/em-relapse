"""Training-time defences as extra loss terms / forward hooks.

All defences share one interface so the trainer, the stress tests and the
paper tables treat them identically:

    d = build_defense(cfg, model, tok, probe_batch, safe_loader)
    with d.train_context():          # hooks active only while training
        loss = task_loss + d.extra_loss(step, batch)

Data-level defences (interleave, inoculation) are applied in data.py and use
the NoDefense object here. Early stopping is a checkpoint-selection rule
(PROTOCOL.md §5) and also uses NoDefense.
"""
from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .persona import _layers, residual_hook


class NoDefense:
    name = "none"

    def __init__(self, *a, **k):
        pass

    def train_context(self):
        return nullcontext()

    def extra_loss(self, step: int, batch: dict, model) -> torch.Tensor | None:
        return None

    def after_step(self, step: int, model) -> None:
        pass


class KLDefense(NoDefense):
    """KL(p_ref || p_theta) on general/safe prompts (Kaczér et al.)."""
    name = "kl"

    def __init__(self, safe_iter, beta: float):
        self.safe_iter, self.beta = safe_iter, beta

    def extra_loss(self, step, batch, model):
        sb = {k: v.to(model.device) for k, v in next(self.safe_iter).items()}
        mask = sb["attention_mask"].bool()
        with torch.no_grad(), model.disable_adapter():
            ref = model(input_ids=sb["input_ids"], attention_mask=sb["attention_mask"]).logits
        cur = model(input_ids=sb["input_ids"], attention_mask=sb["attention_mask"]).logits
        kl = F.kl_div(F.log_softmax(cur, -1), F.log_softmax(ref, -1), log_target=True, reduction="none").sum(-1)
        return self.beta * kl[mask].mean()


class LDIFSDefense(NoDefense):
    """l2 distance between current and reference features at several layers."""
    name = "ldifs"

    def __init__(self, safe_iter, beta: float, layers: list[int]):
        self.safe_iter, self.beta, self.layers = safe_iter, beta, layers

    def _feats(self, model, sb):
        out = model(input_ids=sb["input_ids"], attention_mask=sb["attention_mask"], output_hidden_states=True)
        return [out.hidden_states[l + 1] for l in self.layers]

    def extra_loss(self, step, batch, model):
        sb = {k: v.to(model.device) for k, v in next(self.safe_iter).items()}
        m = sb["attention_mask"].bool()
        with torch.no_grad(), model.disable_adapter():
            ref = self._feats(model, sb)
        cur = self._feats(model, sb)
        return self.beta * sum(((c - r) ** 2).sum(-1)[m].mean() for c, r in zip(cur, ref)) / len(cur)


class HookDefense(NoDefense):
    """Preventative steering (mode='steer') or subspace ablation / CAFT-style
    projection (mode='ablate') applied only during training."""

    def __init__(self, subspace: dict, mode: str, alpha: float = 0.0):
        self.sub, self.mode, self.alpha = subspace, mode, alpha
        self.name = "steer" if mode == "steer" else "ablate"

    def train_context(self, model=None):
        return residual_hook(self._model, self.sub["layer"], self.sub["basis"], self.mode, self.alpha)

    def bind(self, model):
        self._model = model
        return self


class ProjectionPenalty(NoDefense):
    """One-sided, reference-anchored penalty on projection onto the misalignment
    subspace, measured on neutral probe prompts (direction analogue of BLOCK-EM).
        L = beta * mean ReLU( <h_theta, B> - <h_ref, B> )"""
    name = "projpen"

    def __init__(self, subspace: dict, probe_batch: dict, beta: float):
        self.sub, self.probe, self.beta = subspace, probe_batch, beta

    def _proj(self, model, ref: bool):
        store: dict = {}
        pb = {k: v.to(model.device) for k, v in self.probe.items()}
        ctx = model.disable_adapter() if ref else nullcontext()
        with ctx, residual_hook(model, self.sub["layer"], self.sub["basis"], "read", store=store):
            model(input_ids=pb["input_ids"], attention_mask=pb["attention_mask"])
        m = pb["attention_mask"].bool()
        return store["proj"][m]                      # (tokens, rank)

    def extra_loss(self, step, batch, model):
        with torch.no_grad():
            ref = self._proj(model, ref=True)
        cur = self._proj(model, ref=False)
        return self.beta * F.relu(cur - ref).mean()


def build_defense(cfg: dict, model, tok, subspace=None, probe_batch=None, safe_iter=None):
    kind = cfg.get("defense", "none")
    p = cfg.get("defense_params", {})
    if kind in ("none", "interleave", "inoculation", "early_stop"):
        return NoDefense()
    if kind == "kl":
        return KLDefense(safe_iter, p.get("beta", 0.1))
    if kind == "ldifs":
        n = len(_layers(model))
        return LDIFSDefense(safe_iter, p.get("beta", 0.01), p.get("layers", [n // 4, n // 2, 3 * n // 4]))
    if kind in ("steer", "ablate"):
        return HookDefense(subspace, kind, p.get("alpha", 4.0)).bind(model)
    if kind == "projpen":
        return ProjectionPenalty(subspace, probe_batch, p.get("beta", 1.0))
    if kind == "rrt":
        from .rrt import RelapseResistant
        return RelapseResistant(cfg, model, tok, subspace, probe_batch)
    raise ValueError(f"unknown defense {kind}")
