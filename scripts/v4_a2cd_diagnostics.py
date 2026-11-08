"""A2-C / A2-D diagnostics for the v4 Stage A representation.

Spec reference (PROTEIN_JEPA_V4_STATIC_DYNAMIC_GLE_GOAL.md):

  * A2-C "Dynamic information and predictive sufficiency": H_dyn must be
    *predictive* of future slow variables, not merely different.  Multi-lag
    latent-future prediction with proper scoring + persistence / null baselines,
    and a low-capacity linear probe (spec A3.7).
  * A2-D "Static/dynamic disentanglement and conditioning": H_dyn must not be
    *only* protein identity; measure within-protein conformational signal
    (per-protein centering) and conditional sensitivity.

This script loads the trained Stage A checkpoint, encodes held-out trajectories,
and reports:

  1. A2-C predictive sufficiency:
       * linear probe: H_dyn(t) -> H_dyn(t+k)  (multi-lag k in frames)
       * linear probe: H_dyn(t) -> CA displacement  (physical slow variable)
       * persistence null: predict H_dyn(t+k) == H_dyn(t)
       * a static-only null (identity probe): can H_static alone predict H_dyn(t+k)?
     Reported as variance-explained (R^2) vs the null baselines.

  2. A2-D disentanglement:
       * within-protein vs cross-protein H_dyn variance (is H_dyn identity-dominated?)
       * per-protein centered H_dyn still separates conformations (rank correlation
         of ||dH_dyn|| vs physical CA-RMSD within a protein).

Outputs a JSON report and a short human-readable summary.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from protein_jepa_frame.models.static_dynamic_v4 import (
    StaticEncoder,
    DynamicEncoder,
    ConditionalStructureDecoder,
)
from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
from protein_jepa_frame.training.train_lasp_calibration import loaders, _move_batch


DECODER_CLASSES = {
    "geometric_ca": GeometricDeltaCADecoder,
}


def _r2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Masked R^2 (variance explained) over the last dim, pooled over mask."""
    m = mask.bool()
    p = pred[m]
    t = target[m]
    if p.numel() == 0 or t.numel() == 0:
        return float("nan")
    ss_res = (p - t).square().sum()
    ss_tot = (t - t.mean()).square().sum()
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation of two flattened float tensors."""
    x = x.double().flatten()
    y = y.double().flatten()
    if x.numel() < 2:
        return float("nan")

    def rank(v):
        _, inv = torch.sort(v)
        ranks = torch.empty_like(v)
        ranks[inv] = torch.arange(v.numel(), dtype=v.dtype, device=v.device)
        return ranks

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def build_model(ckpt_path: str, device: torch.device):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dyn_sd = sd["dynamic_encoder"]
    dim = dyn_sd["encoder.aatype_embed.weight"].shape[1]
    # Infer encoder geometry flags from the checkpoint keys (same trick as Stage B).
    orientation_features = "encoder.orientation_bias.weight" in dyn_sd
    relative_position_features = "encoder.relpos_proj.weight" in dyn_sd
    anchor_conditioned = "encoder.anchor_basis_proj.weight" in dyn_sd

    static_encoder = StaticEncoder(dim)
    dynamic_encoder = DynamicEncoder(
        dim,
        orientation_features=orientation_features,
        relative_position_features=relative_position_features,
        anchor_conditioned=anchor_conditioned,
    )
    static_encoder.load_state_dict(sd["static_encoder"], strict=True)
    dynamic_encoder.load_state_dict(dyn_sd, strict=True)

    # Decoder: reconstruct from the conditional_decoder's nested decoder config.
    decoder_mode = sd["args"].get("decoder_mode", "geometric_ca")
    decoder = DECODER_CLASSES[decoder_mode]()
    conditional_decoder = ConditionalStructureDecoder(decoder, dim)
    conditional_decoder.load_state_dict(sd["conditional_decoder"], strict=True)

    return static_encoder, dynamic_encoder, conditional_decoder, dim


@torch.no_grad()
def encode_trajectory(model, batch, device):
    static_encoder, dynamic_encoder, _, dim = model
    coords = batch["coords"]
    b, t, l = coords.shape[:3]
    mask = batch["residue_mask"].bool().all(1)
    aa, ri, ci = [batch[k] for k in ("aatype", "residue_index", "chain_id")]
    expand = lambda x: x[:, None].expand(b, t, l).reshape(b * t, l)
    rm = mask[:, None].expand(b, t, l).reshape(b * t, l)

    h_static = static_encoder(coords[:, 0], mask, aa, ri, ci)  # [b,l,D]
    h_static_expanded = h_static[:, None].expand(b, t, l, -1).reshape(b * t, l, -1)
    h_dyn = dynamic_encoder(
        coords.reshape(b * t, 1, l, 4, 3), rm[:, None],
        expand(aa), expand(ri), expand(ci),
        h_static=h_static_expanded, token_mask=rm[:, None],
    )[:, 0].reshape(b, t, l, -1)  # [b,t,l,D]

    # Physical CA displacement from the anchor frame (slow variable proxy).
    ca = coords[:, :, :, 1]  # [b,t,l,3]
    ca0 = ca[:, :1]
    disp = (ca - ca0).norm(dim=-1)  # [b,t,l]

    return h_static, h_dyn, disp, mask


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-a-ckpt", default="artifacts/v4_stage_a/run_400/checkpoint_best.pt")
    p.add_argument("--val-manifest", default="artifacts/mdcath_manifest_full/val.jsonl")
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-residues", type=int, default=96)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--lags", type=str, default="1,2,4")
    p.add_argument("--output", default="artifacts/v4_a2cd_report.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    lags = [int(x) for x in a.lags.split(",")]

    static_encoder, dynamic_encoder, conditional_decoder, dim = build_model(a.stage_a_ckpt, device)
    model = (static_encoder, dynamic_encoder, conditional_decoder, dim)
    for m in (static_encoder, dynamic_encoder, conditional_decoder):
        m.eval().to(device)

    val = loaders(a, a.val_manifest, False)

    # Accumulate per-batch, pooling only VALID residues so proteins of different
    # lengths can be concatenated along the flat (valid) dim.
    cur_list, fut_list = {lag: [] for lag in lags}, {lag: [] for lag in lags}
    static_list, dyn_pool_list, disp_pool_list = [], [], []
    within_norm_terms = []  # (sum_norm, count) for within-protein centered norm
    cross_means = []        # per-sample mean H_dyn [D] per sample
    rhos = []

    for stride, loader in val.items():
        for batch in loader:
            batch = _move_batch(batch, device)
            h_static, h_dyn, disp, mask = encode_trajectory(model, batch, device)
            N, T, L, D = h_dyn.shape
            m = mask.bool()  # [N,L]
            # valid residue mask broadcast over time [N,T,L]
            m_t = m[:, None].expand(N, T, L).reshape(-1)
            h_dyn_flat = h_dyn.reshape(-1, D)
            disp_flat = disp.reshape(-1)

            # static identity probe: H_static(t) vs H_dyn(t) over valid residues
            h_static_t = h_static[:, None].expand(N, T, L, D).reshape(-1, D)
            static_list.append(h_static_t[m_t])
            dyn_pool_list.append(h_dyn_flat[m_t])
            disp_pool_list.append(disp_flat[m_t])

            # multi-lag pairs
            for lag in lags:
                if T <= lag:
                    continue
                pm = m[:, None].expand(N, T - lag, L).reshape(-1)
                cur_list[lag].append(h_dyn[:, :T - lag].reshape(-1, D)[pm])
                fut_list[lag].append(h_dyn[:, lag:].reshape(-1, D)[pm])

            # A2-D: within-protein centered norm (subtract per-(sample,residue) time mean)
            h_dyn_centered = h_dyn - h_dyn.mean(dim=1, keepdim=True)  # [N,t,l,D]
            cnorm = h_dyn_centered.reshape(-1, D)[m_t].norm(dim=-1)
            within_norm_terms.append((cnorm.sum().item(), cnorm.numel()))
            # cross-protein: per-sample mean over time+valid residues
            for n in range(N):
                per_samp = h_dyn[n]  # [T,L,D]
                ms = m[n].bool()
                cross_means.append(per_samp[:, ms].mean(dim=(0, 1)))  # [D]
                # within-protein spearman of ||dH|| vs |d disp|
                dh = (per_samp[1:] - per_samp[:1]).norm(dim=-1)  # [T-1,L]
                dd = (disp[n, 1:] - disp[n, :1]).abs()  # [T-1,L]
                dh_m = dh[:, ms].reshape(-1)
                dd_m = dd[:, ms].reshape(-1)
                if dh_m.numel() >= 4:
                    r = _spearman(dh_m, dd_m)
                    if not math.isnan(r):
                        rhos.append(r)

    static_t = torch.cat(static_list, 0)
    dyn_pool = torch.cat(dyn_pool_list, 0)
    disp_pool = torch.cat(disp_pool_list, 0)
    n_valid = dyn_pool.shape[0]

    report = {}

    # ---------------- A2-C: predictive sufficiency ----------------
    pred = {}
    for lag in lags:
        if not cur_list[lag]:
            continue
        cur = torch.cat(cur_list[lag], 0)
        fut = torch.cat(fut_list[lag], 0)
        persist_r2 = _r2(cur, fut, torch.ones(cur.shape[0], dtype=torch.bool, device=cur.device))
        # Linear ridge probe fit/eval split.
        n = cur.shape[0]
        idx = torch.arange(n)
        fit_idx = idx[: n // 2]
        ev_idx = idx[n // 2:]
        X = torch.cat([cur[fit_idx], torch.ones(fit_idx.numel(), 1, device=cur.device)], -1)
        Y = fut[fit_idx]
        lam = 1.0
        A = X.t() @ X + lam * torch.eye(D + 1, device=cur.device, dtype=cur.dtype)
        W = torch.linalg.solve(A, X.t() @ Y)
        Xe = torch.cat([cur[ev_idx], torch.ones(ev_idx.numel(), 1, device=cur.device)], -1)
        Yhat = Xe @ W
        probe_r2 = _r2(Yhat, fut[ev_idx], torch.ones(ev_idx.numel(), dtype=torch.bool, device=cur.device))
        pred[lag] = {"persistence_r2": persist_r2, "linear_probe_r2": probe_r2}

    # Static-only null (identity probe): H_static -> H_dyn(t).
    static_r2 = _r2(static_t, dyn_pool, torch.ones(n_valid, dtype=torch.bool, device=static_t.device))

    # H_dyn(t) -> physical CA displacement (slow-variable probe).
    n = dyn_pool.shape[0]
    idx = torch.arange(n)
    fit_idx = idx[: n // 2]
    ev_idx = idx[n // 2:]
    X = torch.cat([dyn_pool[fit_idx], torch.ones(fit_idx.numel(), 1, device=dyn_pool.device)], -1)
    Y = disp_pool[fit_idx].unsqueeze(-1)
    lam = 1.0
    A = X.t() @ X + lam * torch.eye(D + 1, device=dyn_pool.device, dtype=dyn_pool.dtype)
    W = torch.linalg.solve(A, X.t() @ Y)
    Xe = torch.cat([dyn_pool[ev_idx], torch.ones(ev_idx.numel(), 1, device=dyn_pool.device)], -1)
    disp_pred = (Xe @ W).squeeze(-1)
    disp_probe_r2 = _r2(disp_pred, disp_pool[ev_idx], torch.ones(ev_idx.numel(), dtype=torch.bool, device=dyn_pool.device))

    report["predictive"] = {
        "multi_lag": pred,
        "static_only_identity_probe_r2": static_r2,
        "h_dyn_to_displacement_probe_r2": disp_probe_r2,
    }

    # ---------------- A2-D: disentanglement ----------------
    within_norm = sum(s for s, _ in within_norm_terms) / sum(c for _, c in within_norm_terms)
    total_norm = dyn_pool.norm(dim=-1).mean().item()
    cross_means = torch.stack(cross_means, 0)  # [N_samp, D]
    cross_spread = cross_means.std(dim=0).norm().item()

    report["disentanglement"] = {
        "within_protein_centered_norm": within_norm,
        "total_norm": total_norm,
        "cross_protein_spread": cross_spread,
        "within_over_total": within_norm / total_norm if total_norm > 0 else float("nan"),
        "within_protein_spearman_dH_disp": float(sum(rhos) / len(rhos) if rhos else float("nan")),
    }

    with open(a.output, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
