"""Misalignment / persona directions in the residual stream.

- extract_subspace: contrastive teacher forcing (aligned vs misaligned answers to
  the same probe questions) -> diff-of-means direction + top-r PCA subspace.
- projection hooks: read, ablate (project out) or add (steer) along a subspace.
- principal_angles: how much a subspace has rotated (rerouting analysis, Paper 2).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def _layers(model):
    m = model
    for attr in ("base_model", "model"):           # unwrap PEFT / CausalLM
        while hasattr(m, attr) and not hasattr(m, "layers"):
            m = getattr(m, attr)
    return m.layers


@torch.no_grad()
def response_mean_acts(model, tok, pairs: list[tuple[str, str]], layer: int, bs: int = 8) -> torch.Tensor:
    """Mean residual activation over *answer* tokens at `layer` for each (q, a)."""
    out = []
    store = {}

    def hook(_m, _i, o):
        store["h"] = o[0] if isinstance(o, tuple) else o

    h = _layers(model)[layer].register_forward_hook(hook)
    try:
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            for q, a in chunk:  # per-example to get exact answer span
                p = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=True)
                f = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}], tokenize=True)
                ids = torch.tensor([f], device=model.device)
                model(input_ids=ids)
                out.append(store["h"][0, len(p):].float().mean(0).cpu())
    finally:
        h.remove()
    return torch.stack(out)


def extract_subspace(model, tok, questions: list[str], aligned: list[str], misaligned: list[str],
                     layer: int, rank: int = 1) -> dict:
    """Returns {'direction': unit diff-of-means, 'basis': (rank, d) orthonormal, 'layer'}."""
    A = response_mean_acts(model, tok, list(zip(questions, aligned)), layer)
    M = response_mean_acts(model, tok, list(zip(questions, misaligned)), layer)
    d = (M - A)
    mean_dir = d.mean(0)
    mean_dir = mean_dir / mean_dir.norm()
    if rank == 1:
        basis = mean_dir[None]
    else:
        _, _, Vh = torch.linalg.svd(d - d.mean(0), full_matrices=False)
        basis = orthonormalize(torch.cat([mean_dir[None], Vh[: rank - 1]]))
    return {"direction": mean_dir, "basis": basis, "layer": layer}


def orthonormalize(B: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(B.T)
    return Q.T


def principal_angles(B1: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians) between row-spaces of two orthonormal bases."""
    s = torch.linalg.svdvals(orthonormalize(B1) @ orthonormalize(B2).T).clamp(-1, 1)
    return torch.arccos(s)


@contextmanager
def residual_hook(model, layer: int, basis: torch.Tensor, mode: str, alpha: float = 0.0, store: dict | None = None):
    """mode='ablate': h <- h - (h B^T) B   (CAFT / subspace projection defence)
       mode='steer' : h <- h + alpha * B[0] (preventative steering, Chen et al.)
       mode='read'  : store['proj'] = h B^T (no change)"""
    B = basis.to(next(model.parameters()).device)

    def hook(_m, _i, o):
        h = o[0] if isinstance(o, tuple) else o
        Bt = B.to(h.dtype)
        if mode == "read":
            if store is not None:
                store["proj"] = h @ Bt.T
            return o
        if mode == "ablate":
            h = h - (h @ Bt.T) @ Bt
        elif mode == "steer":
            h = h + alpha * Bt[0]
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    handle = _layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
