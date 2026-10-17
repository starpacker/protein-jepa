"""The future-state producers.

This is the SINGLE place where the variants differ.  Everything else (encoder
call, alignment, metric) is shared and lives in loaders.py / geometry.py.

A producer maps (bundle, ctx, H_ctx, horizon) -> predicted CA in the anchor frame.
`H_ctx` is passed in so the runner can encode each distinct encoder ONCE per
window and share it across variants (h_only and h_ft use different encoders;
h_ft and h_plus_s share one).

Definitions (EVALUATION_PLAN.md §1.2):
    persist  : ca_pred = X_anchor
    h_only   : H_future = predictor(H_context, tau, T), then decode
    h_ft     : same predictor path with the B1a fine-tuned encoder
    h_plus_s : H_future = rollout(H_anchor, S0(history), horizon), then decode

``H_context`` is always ``[B,K,L,dH]``.  The direct predictor returns factor
deltas internally; ``ModelBundle.predict_future`` converts them to the absolute
``H_future [B,L,dH]`` consumed by the decoder.
"""
from __future__ import annotations

import torch

from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame

from .geometry import ca_from_backbone


def encode_clip(bundle, clip: dict) -> torch.Tensor:
    """Encode a clip dict -> H [1,T,L,dH] (normalized)."""
    return bundle.encode(
        clip["coords"], clip["residue_mask"],
        clip["aatype"], clip["residue_index"], clip["chain_id"],
    )


def anchor_frames(ctx: dict):
    """(R [1,L,3,3], t [1,L,3]) of the anchor = last context frame."""
    bb = ctx["coords"][:, -1]                       # [1,L,4,3]
    N, CA, C = bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]
    return RigidBodyFrame.from_backbone(N, CA, C)


def decode(bundle, H_future, H_context, ctx, R_anchor, t_anchor, rmask):
    """Decode H -> absolute CA (anchor frame) using the bundle's decoder."""
    if bundle.decoder_kind == "oracle":
        out = bundle.decoder.decode_structure(
            H=H_future, aatype=ctx["aatype"],
            R_anchor=R_anchor, t_anchor=t_anchor, residue_mask=rmask,
        )
        return out["CA"]                             # [1,L,3]
    if bundle.decoder_kind == "residual":
        X_anchor = ca_from_backbone(ctx["coords"][:, -1])     # [1,L,3]
        out = bundle.decoder(H_future, H_context, X_anchor, rmask)
        return out["ca_pred"]                        # [1,L,3]
    if bundle.decoder_kind == "identity":
        return ca_from_backbone(ctx["coords"][:, -1])
    raise ValueError(f"unknown decoder_kind {bundle.decoder_kind}")


@torch.no_grad()
def produce(variant: str, bundle, ctx: dict, H_ctx, horizon: int, *,
            horizon_ps: float | None = None, seed: int = 0,
            n_rollout_reps: int = 1) -> dict:
    """H_ctx: [1,K,L,dH] encoded context for THIS variant's encoder (may be None for persist)."""
    rmask = ctx["residue_mask"][:, -1]
    X_anchor = ca_from_backbone(ctx["coords"][:, -1])   # [1,L,3]

    if variant == "persist":
        return {"ca_pred": X_anchor[0], "H_future": None, "meta": {"msd": 0.0, "n_reps": 1}}

    R_anchor, t_anchor = anchor_frames(ctx)

    if variant in ("h_only", "h_ft"):
        device = H_ctx.device
        lag_ps = torch.full(
            (H_ctx.shape[0],),
            float(horizon_ps if horizon_ps is not None else horizon * 1000.0),
            device=device,
            dtype=H_ctx.dtype,
        )
        temperature = ctx.get("temperature_k")
        if temperature is None:
            temperature = torch.full((H_ctx.shape[0],), 348.0, device=device, dtype=H_ctx.dtype)
        else:
            temperature = torch.as_tensor(temperature, device=device, dtype=H_ctx.dtype).reshape(-1)
        H_future = bundle.predict_future(H_ctx, lag_ps, temperature, rmask)
        H_anchor = H_ctx[:, -1]
        ca = decode(bundle, H_future, H_anchor, ctx, R_anchor, t_anchor, rmask)
        dx = ca - X_anchor
        msd = float((dx * rmask.unsqueeze(-1)).pow(2).sum(-1).mean())
        return {"ca_pred": ca[0], "H_future": H_future[0],
                "meta": {"msd": msd, "n_reps": 1}}

    if variant == "h_plus_s":
        b1b = bundle.b1b
        if b1b is None:
            raise ValueError("h_plus_s needs bundle.b1b")
        dev = ctx["coords"].device
        H_anchor = H_ctx[:, -1]
        K = H_ctx.shape[1]
        dt_hist = torch.full((1, K), b1b.cfg.step_ps, device=dev)
        T = ctx.get("temperature_k")
        if T is not None:
            T = T.to(dev)
        reps, msds = [], []
        for r in range(n_rollout_reps):
            gen = torch.Generator(device=dev).manual_seed(seed + r)
            traj = b1b.rollout(
                H_anchor, rmask, T, n_steps=horizon,
                H_hist=H_ctx, dt_hist=dt_hist, generator=gen, detach=True,
            )                                        # [1, horizon+1, L, dH]
            ca = decode(bundle, traj[:, horizon], H_anchor, ctx, R_anchor, t_anchor, rmask)
            reps.append(ca[0])
            dx = ca[0] - X_anchor[0]
            msds.append(float((dx * rmask[0].unsqueeze(-1)).pow(2).sum(-1).mean()))
        return {"ca_pred": torch.stack(reps, 0).mean(0), "H_future": None,
                "meta": {"msd": sum(msds) / len(msds), "n_reps": n_rollout_reps, "msd_reps": msds}}

    raise ValueError(f"unknown variant '{variant}'")
