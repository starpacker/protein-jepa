"""Phase 0 audit for PROTEIN_JEPA_V4 (static/dynamic GLE).

Reproduces the three root causes the v4 goal names, on the current
correct-framework pipeline (CAComponentBundle encoder/decoder + StageB1b GLE),
against the exact_seed42 checkpoint, on held-out val proteins.

Measurements:
  1. Encoder magnitude-lock: per-residue ||H|| across frames (should be constant
     if LayerNorm(elementwise_affine=False) locks magnitude); cos(H_t, H_0) drift
     vs ||H_t - H_0|| growth.
  2. Decoder variance collapse: var(true future) vs var(decode(encode(GT)))
     per protein, plus aligned CA-RMSD of decode(encode(GT)) vs true.
  3. GLE off-manifold drift: autonomous rollout ||H|| over 64 steps vs true ||H||,
     and cos(GLE_H_end, true_H_end).

Usage:
  PYTHONPATH=/tmp/lasp_stability_deps:/liuyutian/protein_jepa_frame \
  python scripts/v4_phase0_audit.py --n-windows 3
"""
from __future__ import annotations

import argparse
import torch
from pathlib import Path

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.training.train_utils import _move_batch
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from evaluation.ensemble_metrics_ext import _align_to
from evaluation.geometry import kabsch_align


CKPT = "artifacts/lasp_s0_arms_20261031/exact_seed42/checkpoint_best.pt"


def _norm(h, mask):
    """Mean per-residue L2 norm of H [L, D] over valid residues."""
    return h[mask].norm(dim=-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=3)
    ap.add_argument("--manifest", default="artifacts/mdcath_manifest_full/val.jsonl")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--max-residues", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(4)
    device = "cuda"

    # Load the correct-framework component bundle + GLE kernel.
    model = CADynamics(
        a.ckpt, "cpu",
        init_kernel=a.ckpt, bath_dim=8, history_conditioning=True,
        noise_mix=0.0, noise_length=8.0, collective_mobility=True,
        noise_kernel="gaussian", noise_cutoff=10.0, s0_mode="exact",
    ).to(device)
    model.eval()

    cfg = DataConfig(
        manifest_path=a.manifest, training_mode="clip",
        num_frames=4 + 64, frame_stride=1, max_residues=a.max_residues,
    )
    ds = build_dataset(cfg)
    # Deterministic subset for reproducibility.
    n = min(len(ds), a.n_windows)
    idx = torch.linspace(0, len(ds) - 1, n).long().tolist()

    print("== Phase 0 audit: correct-framework exact_seed42 ==")
    print(f"ckpt: {a.ckpt}")
    print(f"windows: {n}")

    for k in idx:
        b = collate_clips([ds[k]])
        # collate returns a dict of CPU tensors; move to device like the trainer.
        batch = _move_batch(b, device)
        coords = batch["coords"]  # [1, T, L, 4, 3]
        B, T, L = coords.shape[:3]
        mask = batch["residue_mask"].bool().all(1)  # [1, L]
        aa, ri, ci = batch["aatype"], batch["residue_index"], batch["chain_id"]
        temp = batch["temperature_k"]

        observed = coords[:, :4]  # [1, 4, L, 4, 3]
        m = mask[0]  # [L]
        anchor_ca = observed[0, -1, :, 1]  # [L, 3]

        # ---- Encode full trajectory (all frames) ----
        with torch.no_grad():
            H_all = model.bundle.encoder(
                coords.reshape(B * T, 1, L, 4, 3),
                mask[:, None].expand(B, T, L).reshape(B * T, L)[:, None],
                aa[:, None].expand(B, T, L).reshape(B * T, L),
                ri[:, None].expand(B, T, L).reshape(B * T, L),
                ci[:, None].expand(B, T, L).reshape(B * T, L),
                token_mask=mask[:, None].expand(B, T, L).reshape(B * T, L)[:, None],
            )[:, 0].reshape(B, T, L, -1)

        H = H_all[0]  # [T, L, D]
        H0 = H[0]
        norms = torch.stack([_norm(H[t], m) for t in range(T)])
        cos0 = torch.stack([
            ((H[t][m] * H0[m]).sum(-1) /
             (H[t][m].norm(dim=-1) * H0[m].norm(dim=-1)).clamp_min(1e-9)).mean()
            for t in range(T)
        ])
        dnorm = torch.stack([
            (H[t][m] - H0[m]).norm(dim=-1).mean() for t in range(T)
        ])

        print(f"\n--- window {k} ({batch['protein_id']} / {batch['trajectory_id']}) ---")
        print(f"  [encoder] ||H|| frame0={norms[0].item():.4f}  frame64={norms[-1].item():.4f}"
              f"  std over frames={norms.std().item():.6f}  (locked if ~0)")
        print(f"  [encoder] cos(H_t,H_0): 0->{cos0[0].item():.3f}  64->{cos0[-1].item():.3f}")
        print(f"  [encoder] ||H_t-H_0||: 0->{dnorm[0].item():.3f}  64->{dnorm[-1].item():.3f}")

        # ---- Decoder variance collapse: decode(encode(GT)) vs true ----
        with torch.no_grad():
            R, anchor = _extract_backbone_frames(observed[:, -1])
            # decode all frames
            h_flat = H_all.reshape(B * T, L, -1)
            anchor_h = H_all[:, :1].expand(B, T, L, -1).reshape(B * T, L, -1)
            ca_hat = model.bundle.decode_ca(
                h_flat, anchor_h,
                aa[:, None].expand(B, T, L).reshape(B * T, L),
                ri[:, None].expand(B, T, L).reshape(B * T, L),
                ci[:, None].expand(B, T, L).reshape(B * T, L),
                R[:, None].expand(B, T, L, 3, 3).reshape(B * T, L, 3, 3),
                anchor[:, None].expand(B, T, L, 3).reshape(B * T, L, 3),
                mask[:, None].expand(B, T, L).reshape(B * T, L),
            ).reshape(B, T, L, 3)

        # True future CA, aligned to anchor.
        true_ca = coords[0, :, :, 1]  # [T, L, 3]
        true_aligned = _align_to(true_ca, anchor_ca, m)  # [T, L, 3]
        pred_aligned = _align_to(ca_hat[0], anchor_ca, m)

        true_var = (true_aligned[1:] - true_aligned[1:].mean(0)).square().sum(-1)[:, m].sum()
        pred_var = (pred_aligned[1:] - pred_aligned[1:].mean(0)).square().sum(-1)[:, m].sum()
        # per-frame aligned CA-RMSD of pred vs true (future frames only): sqrt(mean_i ||·||^2)
        rmsd = ((pred_aligned[1:][:, m] - true_aligned[1:][:, m]).square().sum(-1).mean(-1)).sqrt()
        md_move = (true_aligned[1:][:, m] - true_aligned[:1][:, m].expand_as(true_aligned[1:][:, m])).square().sum(-1).mean(-1).sqrt()

        print(f"  [decoder] var(true)={true_var.item():.1f}  var(encdec)={pred_var.item():.1f}"
              f"  ratio={pred_var.item()/true_var.item():.3f}")
        print(f"  [decoder] decode(encode(GT)) aligned CA-RMSD: h1={rmsd[0].item():.2f}  "
              f"h4={rmsd[3].item():.2f}  h16={rmsd[15].item():.2f}  h64={rmsd[-1].item():.2f} A")
        print(f"  [decoder] true MD move vs anchor: h1={md_move[0].item():.2f}  h64={md_move[-1].item():.2f} A")

        # ---- GLE off-manifold drift: autonomous rollout ----
        with torch.no_grad():
            H0_batch = H_all[:, 0]  # [B, L, D]
            rm = mask
            # s0_mode='exact' requires an observed history (last 4 frames of H)
            hist = H_all[:, :4]  # [B, 4, L, D]
            dt_hist = torch.full((B, 4), 1000.0, device=device)  # 1 ns/frame
            traj = model.kernel.rollout(
                H0_batch, rm, temp.to(device), 64,
                H_hist=hist, dt_hist=dt_hist, seq_feature=H0_batch,
                generator=torch.Generator(device=device).manual_seed(a.seed),
                collective_mobility=True,
            )  # [B, 65, L, D]
        H_gle = traj[0]
        gle_norms = torch.stack([_norm(H_gle[t], m) for t in (0, 8, 32, 64)])
        true_norms = torch.stack([_norm(H[t], m) for t in (0, 8, 32, 64)])
        # cos of GLE endpoint vs true endpoint
        cos_gle_true = (H_gle[64][m] * H[64][m]).sum(-1) / \
            (H_gle[64][m].norm(dim=-1) * H[64][m].norm(dim=-1)).clamp_min(1e-9)

        print(f"  [GLE] ||H|| t0/t8/t32/t64: "
              + " / ".join(f"{v.item():.1f}" for v in gle_norms))
        print(f"  [GLE] true ||H|| t0/t8/t32/t64: "
              + " / ".join(f"{v.item():.1f}" for v in true_norms))
        print(f"  [GLE] cos(H_gle_end, H_true_end) = {cos_gle_true.mean().item():.3f}")

    print("\n== audit complete ==")


if __name__ == "__main__":
    main()
