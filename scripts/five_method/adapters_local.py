"""Adapters for the three in-repo methods: Model 1, Model 2, Lasp.

Uniform contract for every method in this harness:

    sample(win, n_samples) -> dict(
        samples       : [N, Lv, 3]   terminal CA cloud at the window horizon,
        walk          : [M, T, Lv, 3] CA walk, walk[0] == anchor, walk[t] is the
                        frame t steps after the anchor, T == horizon+1;
                        ``None`` if the method cannot produce an ordered walk,
        walk_steps_ms : step size of ``walk`` (frames are 1 ns in mdCATH),
    )

`walk` is what the order-aware metrics consume; `samples` is the terminal
ensemble.  A method that cannot produce an ordered walk returns ``walk=None``
and is reported "n/a" rather than scored on a fabricated path.
"""
from __future__ import annotations

import copy

import torch

from protein_jepa_frame.models.ensemble_producer import GaussianHHead, EnsembleProducer
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.training.train_stage_b_hs import HSEnsembleProducer
from protein_jepa_frame.training.train_stage_ensemble import (
    load_frozen_stack, _extract_backbone_frames,
)

from .common import ctx_batched, window_seed


def _seq_fields(ctx):
    """aatype / residue_index / chain_id at [B, L] (they may be [B,K,L])."""
    aat = ctx["aatype"]
    aat = aat[:, -1] if aat.dim() == 3 else aat
    ri = ctx.get("residue_index")
    ri = ri[:, -1] if ri is not None and ri.dim() == 3 else ri
    ci = ctx.get("chain_id")
    ci = ci[:, -1] if ci is not None and ci.dim() == 3 else ci
    return aat, ri, ci


def _resolve_temp(win, temperature_mode, wrong_temp):
    T = float(win["temperature_k"])
    if temperature_mode == "wrong":
        return float(wrong_temp if wrong_temp is not None else 320.0)
    if temperature_mode == "none":
        return 348.0      # neutral (unused when the model has no temperature slot)
    return T


# --------------------------------------------------------------------------- #
# Model 1 — frozen Stage-A encoder + Gaussian H head + frozen anchored decoder
# --------------------------------------------------------------------------- #
def build_model1(args, device):
    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, device)
    ck = torch.load(args.ensemble_ckpt, map_location=device, weights_only=False)
    ca = ck.get("args", {})
    head = GaussianHHead(dH, hidden=int(ca.get("head_hidden", 128)),
                         cond_dim=int(ca.get("cond_dims", 0)))
    head.load_state_dict(ck["head"])
    head.to(device).eval()
    producer = EnsembleProducer(copy.deepcopy(encoder), copy.deepcopy(decoder), head)
    return producer.to(device).eval(), int(ca.get("cond_dims", 0))


def _cond_m1(cond_dim, horizon_ns, temperature_k, device):
    if cond_dim <= 0:
        return None
    t_norm = torch.log(torch.tensor([max(float(horizon_ns) * 1000.0, 1.0)], device=device)) \
        / torch.log(torch.tensor(1e4, device=device))
    T_norm = torch.tensor([(float(temperature_k) - 300.0) / 150.0], device=device)
    parts = [t_norm, T_norm] + [torch.zeros_like(t_norm)] * max(0, cond_dim - 2)
    return torch.stack(parts[:cond_dim], dim=-1)


def _synthetic_backbone(ca, prev_frame):
    """Backbone whose CA equals ``ca``; N/C/O offsets inherit from ``prev_frame``."""
    off = prev_frame - prev_frame[:, :, 1:2, :]
    return ca[:, :, None, :] + off


@torch.no_grad()
def sample_model1(producer, cond_dim, win, n_samples, device, *, walk=True,
                  temperature_mode="correct", wrong_temp=None):
    ctx = ctx_batched(win, device)
    coords = ctx["coords"]                                    # [1,K,L,4,3]
    rmask = ctx["residue_mask"].all(dim=1)                    # [1,L]
    aat, ri, ci = _seq_fields(ctx)
    T = _resolve_temp(win, temperature_mode, wrong_temp)
    cond = _cond_m1(cond_dim, win["horizon_ns"], T, device)
    R, t = _extract_backbone_frames(coords[:, -1])

    out = producer.sample_future_ca(coords, rmask, aat, ri, ci, R, t,
                                    n_samples=n_samples, cond=cond)
    samples = out["ca_samples"][0]                            # [N,L,3]
    if not walk or win["horizon"] < 2:
        return {"samples": samples, "walk": None, "walk_steps_ms": None}

    # Autoregressive walk: exactly the repo's Model-1 convention
    # (scripts/eval_ours_trajectory.rollout_trajectory) — feed the sample mean
    # back as a synthetic backbone and slide the K-frame context.
    buf = coords.clone()
    per_step = []
    for _ in range(int(win["horizon"])):
        Rr, tt = _extract_backbone_frames(buf[:, -1])
        o = producer.sample_future_ca(buf, rmask, aat, ri, ci, Rr, tt,
                                      n_samples=n_samples, cond=cond)
        ca = o["ca_samples"]                                  # [1,n,L,3]
        per_step.append(ca[0])
        nf = _synthetic_backbone(ca.mean(dim=1), buf[:, -1])
        buf = torch.cat([buf[:, 1:], nf[:, None]], dim=1)
    future = torch.stack(per_step, dim=1)                     # [N,T,L,3]
    anchor = coords[:, -1, :, 1, :].expand(n_samples, -1, -1)
    walk_out = torch.cat([anchor[:, None], future], dim=1)     # [N,T+1,L,3]
    return {"samples": samples, "walk": walk_out,
            "walk_steps_ms": win["horizon_ns"] / max(int(win["horizon"]), 1)}


# --------------------------------------------------------------------------- #
# Model 2 — from-none GLE (encoder + kernel + MLP CA decoder)
# --------------------------------------------------------------------------- #
def build_model2(args, device):
    from protein_jepa_frame.training.train_stage_cb1b import load_models_for_eval
    encoder, kernel, decoder, normalizer, _ck = load_models_for_eval(
        args.model2_ckpt, device)
    return (encoder, kernel, decoder, normalizer), float(kernel.cfg.step_ps)


@torch.no_grad()
def sample_model2(bundle, step_ps, win, n_samples, device, *, walk=True,
                  temperature_mode="correct", wrong_temp=None):
    from evaluation.run_eval_fromnone import encode_history, temperature_override
    encoder, kernel, decoder, normalizer = bundle
    ctx = ctx_batched(win, device)
    rmask = ctx["residue_mask"].all(dim=1).bool()             # [1,L]
    anchor_bb = ctx["coords"][:, -1]
    anchor_ca = anchor_bb[..., 1, :]                          # [1,L,3]
    R, t = RigidBodyFrame.from_backbone(anchor_bb[..., 0, :], anchor_bb[..., 1, :],
                                        anchor_bb[..., 2, :])
    H_hist = encode_history(encoder, normalizer, ctx, device)
    dt = ctx["time_ps"].diff(dim=-1)
    dt = torch.cat([dt[:, :1], dt], dim=-1)
    temp = temperature_override(ctx.get("temperature_k"), temperature_mode, device,
                                override_k=float(wrong_temp if wrong_temp is not None else 320.0),
                                B=1)
    if temperature_mode == "none":
        temp = None

    n_steps = max(1, int(round(win["horizon_ns"] * 1000.0 / step_ps)))
    terms, paths, baths = [], [], []
    for rep in range(n_samples):
        gdev = device if isinstance(device, torch.device) else torch.device(device)
        gen = torch.Generator(device=gdev).manual_seed(window_seed(win, run=rep, salt="model2"))
        Ht, St = kernel.rollout(H_hist[:, -1], rmask, temp, n_steps=n_steps,
                                H_hist=H_hist, dt_hist=dt, seq_feature=H_hist[:, -1],
                                generator=gen, detach=True, return_bath=True)
        out = decoder(H_future=Ht[:, -1], H_context=H_hist[:, -1], X_context=anchor_ca,
                      residue_mask=rmask, R_anchor=R, S_future=St[:, -1])
        terms.append(out["ca_pred"][0])                        # [L,3]
        if walk:
            paths.append(Ht[0])                                # [n_steps+1, L, d]
            baths.append(St[0] if St is not None else None)     # [n_steps+1, L, dS]
    samples = torch.stack(terms)                               # [N,L,3]
    walk_out = None
    if walk and paths:
        Hp = torch.stack(paths)                                # [N,T,L,d]
        N, Tn, L, d = Hp.shape
        flat = Hp.reshape(N * Tn, L, d)
        Sp = (torch.stack(baths).reshape(N * Tn, L, -1)
              if baths and baths[0] is not None else None)
        Rf = R[0][None, None].expand(N, Tn, *R[0].shape).reshape(N * Tn, *R[0].shape)
        tf = t[0][None, None].expand(N, Tn, *t[0].shape).reshape(N * Tn, *t[0].shape)
        af = _seq_fields(ctx)[0][0][None, None].expand(N, Tn, L).reshape(N * Tn, L)
        mf = rmask[0][None, None].expand(N, Tn, rmask.shape[1]).reshape(N * Tn, rmask.shape[1])
        Hc = H_hist[:, -1][0][None, None].expand(N, Tn, L, d).reshape(N * Tn, L, d)
        Xc = anchor_ca[0][None, None].expand(N, Tn, L, 3).reshape(N * Tn, L, 3)
        dec = decoder(H_future=flat, H_context=Hc, X_context=Xc,
                      residue_mask=mf, R_anchor=Rf, S_future=Sp)
        walk_out = dec["ca_pred"].reshape(N, Tn, L, 3)
    return {"samples": samples, "walk": walk_out, "walk_steps_ms": step_ps / 1000.0}


# --------------------------------------------------------------------------- #
# Lasp — Stage-A encoder + Stage-B GLE, unfrozen decoder (best variant)
# --------------------------------------------------------------------------- #
def build_lasp(args, device):
    encoder, decoder, _dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, device)
    kck = torch.load(args.lasp_ckpt, map_location=device, weights_only=False)
    k_cfg = dict(kck.get("kernel_config") or {})
    k_args = kck.get("args", {})
    k_cfg["step_ps"] = k_args.get("step_ps", 1000.0)
    k_cfg["rollout_grad_steps"] = None
    kern = StageB1b(StageB1bConfig(**k_cfg)).to(device)
    kern.load_state_dict(kck["kernel"])
    kern.eval()
    te = bool(kck.get("train_encoder", False)) and kck.get("encoder") is not None
    td = bool(kck.get("train_decoder", False)) and kck.get("decoder") is not None
    hs = HSEnsembleProducer(encoder, decoder, kern, train_encoder=te,
                            train_decoder=td).to(device).eval()
    if te:
        hs.encoder.load_state_dict(kck["encoder"], strict=False)
    if td:
        hs.decoder.load_state_dict(kck["decoder"], strict=False)
    hs._rollout_grad_steps = None
    hs._temperature_k = float(k_args.get("temperature_k", 320.0))
    return hs, float(k_args.get("step_ps", 1000.0))


@torch.no_grad()
def sample_lasp(hs, step_ps, win, n_samples, device, *, walk=True,
                temperature_mode="correct", wrong_temp=None):
    ctx = ctx_batched(win, device)
    coords = ctx["coords"]
    rmask = ctx["residue_mask"].all(dim=1)                    # [1,L]
    aat, ri, ci = _seq_fields(ctx)
    T = _resolve_temp(win, temperature_mode, wrong_temp)
    R, t = _extract_backbone_frames(coords[:, -1])
    n_steps = max(1, int(round(win["horizon_ns"] * 1000.0 / step_ps)))
    hs._rollout_steps = n_steps
    out = hs.sample_future_ca(coords, rmask, aat, ri, ci, R, t,
                              n_samples=n_samples, temperature_k=T)
    samples = out["ca_samples"][0]                             # [N,L,3]
    walk_out = None
    if walk:
        Hp = out["H_traj"]                                     # [N,n_steps+1,L,d]
        N, Tn, L, d = Hp.shape
        flat = Hp.reshape(N * Tn, L, d)
        Rf = R[0][None, None].expand(N, Tn, *R[0].shape).reshape(N * Tn, *R[0].shape)
        tf = t[0][None, None].expand(N, Tn, *t[0].shape).reshape(N * Tn, *t[0].shape)
        af = aat[0][None, None].expand(N, Tn, L).reshape(N * Tn, L)
        mf = rmask[0][None, None].expand(N, Tn, L).reshape(N * Tn, L)
        dec = hs.decoder.decode_structure(flat, af, Rf, tf, mf)
        walk_out = dec["backbone_coords"][:, :, 1, :].reshape(N, Tn, L, 3)
    return {"samples": samples, "walk": walk_out,
            "walk_steps_ms": step_ps / 1000.0}
