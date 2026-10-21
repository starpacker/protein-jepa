"""Stage B "H+S (GLE) iteration" on top of Model 1 (NEW, 2026-10-21).

Motivation
----------
Model 1 (Stage-A ensemble) produces its ensemble with a *static* per-residue
Gaussian head:  H^{(n)} = mu(H_anchor) + sigma(H_anchor) * eps_n  (i.i.d., no
temporal evolution).  That fits the RMSF spectrum well but cannot express
*time-directed* stochastic dynamics.

This driver keeps Model 1's strong frozen pieces EXACTLY as-is (Stage-A v2
encoder + B1a anchored decoder), and replaces ONLY the static Gaussian head with
the trainable StageB1b **GLE kernel** (H-space energy U(H) + per-residue bath S).
From the frozen anchor latent H_anchor it rolls out N independent GLE
trajectories and decodes each terminal H through the SAME frozen decoder, giving
an ensemble that is scored with the SAME ensemble losses as Model 1
(RMSF-L1 + contact-L1 + internal-distance soft-EMD).

The question: does explicit H+S iteration beat the static Model 1 head?

Contract / safety
-----------------
* NOTHING in the precise-coordinate decoding path is modified.
* Model 1 checkpoints are read-only; this writes to its own --output dir.
* Losses / data pipeline are reused verbatim from train_stage_ensemble.py so the
  comparison to Model 1 is apples-to-apples (same weights, same val windows).

Usage:
    python -m protein_jepa_frame.training.train_stage_b_hs \
        --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
        --b1a-ckpt     artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
        --manifest     artifacts/mdcath_manifest_1000dom/train.jsonl \
        --val-manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
        --output       artifacts/stage_hs_1000dom
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Subset

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training.ensemble_targets import build_ensemble_targets, persistence_ensemble
from protein_jepa_frame.training.ensemble_losses import (
    EnsembleLossWeights, compute_ensemble_loss,
)
from protein_jepa_frame.training.train_utils import _device, _move_batch, _set_reproducibility
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames, _truncate_history
from protein_jepa_frame.training.train_stage_ensemble import (
    load_frozen_stack, _targets_from, _anchors, _conditioning,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage B H+S (GLE) iteration on Model 1")
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_1000dom/train.jsonl")
    p.add_argument("--val-manifest", default="artifacts/mdcath_manifest_1000dom/val.jsonl")
    p.add_argument("--stage-a-ckpt", required=True)
    p.add_argument("--b1a-ckpt", required=True)
    p.add_argument("--output", default="artifacts/stage_hs_run")
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-samples", type=int, default=8000)

    p.add_argument("--history-frames", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--paired-horizons-frames", type=int, nargs="+", default=None,
                   help="Future-window offsets (raw frames) to sample.  Plan A alignment: pick "
                        "a single horizon H and set --rollout-steps = H so the GLE terminal "
                        "step lands on the scored window instead of a mismatched 1 step.")
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--cond-dims", type=int, default=0)

    # ensemble loss (same default weights as Model 1's 1000dom run)
    p.add_argument("--rmsf-weight", type=float, default=1.0)
    p.add_argument("--contact-weight", type=float, default=10.0)
    p.add_argument("--dist-weight", type=float, default=0.3)
    p.add_argument("--pos-weight", type=float, default=0.0)
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--dist-lo", type=float, default=0.0)
    p.add_argument("--dist-hi", type=float, default=20.0)
    p.add_argument("--dist-bins", type=int, default=50)
    p.add_argument("--min-seq-sep", type=int, default=3)
    p.add_argument("--max-pairs", type=int, default=4096)

    # GLE kernel (StageB1b) hyperparams — defaults mirror from-none v4e
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--bath-dim", type=int, default=8)
    p.add_argument("--energy-depth", type=int, default=3)
    p.add_argument("--energy-heads", type=int, default=4)
    p.add_argument("--energy-hidden", type=int, default=128)
    p.add_argument("--history-hidden", type=int, default=128)
    p.add_argument("--coupling-scale", type=float, default=1e-3)
    p.add_argument("--diffusion-scale", type=float, default=1e-4)
    p.add_argument("--c-init-std", type=float, default=0.1)
    p.add_argument("--gamma-scale", type=float, default=2e-4)
    p.add_argument("--gamma-floor", type=float, default=2e-5)
    p.add_argument("--max-gamma-step", type=float, default=1.9)
    p.add_argument("--force-clip", type=float, default=1.0)
    p.add_argument("--tail-init", type=float, default=1.0)
    p.add_argument("--step-ps", type=float, default=1000.0)
    # rollout length in frames (how far the GLE is iterated before decoding)
    p.add_argument("--rollout-steps", type=int, default=1,
                   help="GLE steps per sample before decoding the terminal H.  Plan A: set "
                        "this to the ensemble window length so the multi-step C@S bath "
                        "feedback is INSIDE the loss -- training a 1-step rollout against a "
                        "whole-window target is exactly what drove the old kernel into the "
                        "divergent regime seen at inference.")
    # --- Plan A: jointly fine-tune the Stage-A encoder with the GLE kernel ---
    p.add_argument("--train-encoder", action="store_true", default=False,
                   help="Plan A: unfreeze the Stage-A encoder and fine-tune it jointly with "
                        "the GLE kernel (decoder stays frozen).  Default off == the old frozen "
                        "encoder contract.  Rationale: Stage A's H already scores well on the "
                        "ensemble metrics and is linearly decodable to CA displacement "
                        "(ridge R^2 ~ +0.29); a low-LR joint fine-tune keeps that signal while "
                        "letting the rollout shape it, rather than relearning H from scratch.")
    p.add_argument("--encoder-lr-scale", type=float, default=0.1,
                   help="LR multiplier for the encoder group when --train-encoder is set.")
    # --- Dense trajectory supervision: score EVERY rollout frame, not just the last ---
    p.add_argument("--dense-trajectory", action="store_true", default=False,
                   help="Score the WHOLE GLE rollout path (all n+1 frames) against the true "
                        "future window, instead of decoding only the terminal H.  The rollout "
                        "already produces the full path; the default loss throws away n of the "
                        "n+1 frames.  Requires the data window to be a CONTIGUOUS future "
                        "([anchor, t1, ..., tH]) so rollout step i lines up with truth frame i.")
    p.add_argument("--dense-stride", type=int, default=1,
                   help="With --dense-trajectory, decode/supervise every k-th rollout frame "
                        "(1 = every frame; raise it to bound memory at long horizons).")
    p.add_argument("--dense-horizon", type=int, default=None,
                   help="With --dense-trajectory, runaway-step guard: abort the rollout once "
                        "||H_t - H_0|| exceeds this (units of H).  None disables the guard.")
    p.add_argument("--dense-ca-weight", type=float, default=1.0,
                   help="Weight on the per-step Kabsch CA point loss (beats persistence; "
                        "gives the encoder a clean per-frame gradient).")
    p.add_argument("--dense-es-weight", type=float, default=0.1,
                   help="Weight on the distributional energy score of the CA path (spread).")
    p.add_argument("--dense-rmsf-weight", type=float, default=0.0,
                   help="Weight on the ensemble-RMSF loss along the path (>0 asks the M "
                        "samples to spread like the true trajectory).")
    p.add_argument("--dense-rmsf-corr-weight", type=float, default=0.0,
                   help="Weight on the ensemble-RMSF-profile-correlation loss along the path.")
    p.add_argument("--dense-var-weight", type=float, default=0.0,
                   help="Weight on the ensemble-variance loss along the path.")
    # truncated BPTT: keep only the last k transitions differentiable
    p.add_argument("--rollout-grad-steps", type=int, default=None)
    p.add_argument("--temperature-k", type=float, default=320.0,
                   help="static T used for the GLE kernel (mdCATH default)")

    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--val-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# H+S ensemble producer: frozen encoder+decoder + trainable GLE kernel
# ---------------------------------------------------------------------------

class HSEnsembleProducer(nn.Module):
    """Frozen Model-1 stack + trainable GLE H+S kernel.

    encoder, decoder: frozen (from load_frozen_stack).
    kernel:           StageB1b GLE dynamics (trainable).

    Ensemble generation from the anchor frame:
        H_anchor = Encoder(anchor)[-1]                       (frozen, no grad)
        {H^{(n)}_tau} = GLE_rollout(H_anchor, n_samples, tau) (kernel, trainable)
        X^{(n)}_CA  = Decoder(H^{(n)}_tau, anchor basis)      (frozen)
    """

    def __init__(self, encoder, decoder, kernel: StageB1b, train_encoder: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kernel = kernel
        for p in self.encoder.parameters():
            p.requires_grad_(bool(train_encoder))
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        # When the encoder is frozen we can (and should) run it under no_grad;
        # when it is fine-tuned we must keep the graph so the GLE losses reach it.
        self._encoder_grad = bool(train_encoder)

    def encode_frame(self, coords, residue_mask, aatype, residue_index, chain_id):
        c5 = coords.unsqueeze(1)
        m3 = residue_mask.unsqueeze(1)
        if not self._encoder_grad:
            with torch.no_grad():
                H = self.encoder(c5, m3, aatype, residue_index, chain_id, token_mask=m3)
            return H[:, 0]
        H = self.encoder(c5, m3, aatype, residue_index, chain_id, token_mask=m3)
        return H[:, 0]

    def sample_future_ca(
        self, context_coords, residue_mask, aatype, residue_index, chain_id,
        R_anchor, t_anchor, n_samples: int, generator=None, cond=None,
        temperature_k: float | None = None,
    ) -> dict:
        B, K, L, _, _ = context_coords.shape
        device = context_coords.device
        anchor_coords = context_coords[:, -1]
        H_anchor = self.encode_frame(anchor_coords, residue_mask, aatype,
                                     residue_index, chain_id)             # [B,L,dH]
        temp = torch.full((B,), float(temperature_k if temperature_k is not None else 320.0),
                          device=device)

        # Roll out `n_samples` independent GLE trajectories in parallel.
        Hs = H_anchor[:, None].expand(B, n_samples, L, H_anchor.shape[-1])
        flat_H = Hs.reshape(B * n_samples, L, H_anchor.shape[-1])
        flat_mask = residue_mask[:, None].expand(B, n_samples, L).reshape(B * n_samples, L)
        flat_T = temp[:, None].expand(B, n_samples).reshape(B * n_samples)
        return self._rollout_decode(flat_H, flat_mask, flat_T, B, n_samples, L,
                                    R_anchor, t_anchor, aatype, residue_mask,
                                    H_anchor, generator)

    def _rollout_decode(self, flat_H, flat_mask, flat_T, B, n_samples, L,
                        R_anchor, t_anchor, aatype, residue_mask, H_anchor, generator):
        n_steps = getattr(self, "_rollout_steps", 1)
        grad_steps = getattr(self, "_rollout_grad_steps", None)
        traj = self.kernel.rollout(
            flat_H, flat_mask, flat_T, n_steps,
            seq_feature=flat_H, generator=generator,
            detach=False, grad_steps=grad_steps,
        )                                                                 # [BN, n+1, L, dH]
        H_tau = traj[:, -1]                                              # terminal H
        flat_R = R_anchor[:, None].expand(B, n_samples, L, 3, 3).reshape(B * n_samples, L, 3, 3)
        flat_t = t_anchor[:, None].expand(B, n_samples, L, 3).reshape(B * n_samples, L, 3)
        flat_aa = aatype[:, None].expand(B, n_samples, L).reshape(B * n_samples, L)
        out = self.decoder.decode_structure(H_tau, flat_aa, flat_R, flat_t, flat_mask)
        ca = out["backbone_coords"][:, :, 1, :].reshape(B, n_samples, L, 3)
        ca_mean_decoded = ca.mean(dim=1)
        # Full mainchain (N, CA, C, O) + heavy-atom atom37 from the frozen
        # decoder's OFFICIAL OpenFold reconstruction path (surfaced for eval).
        backbone = out["backbone_coords"].reshape(B, n_samples, L, 4, 3)
        atom37 = out["atom37_coords"].reshape(B, n_samples, L, -1, 3)
        atom37_mask = out["atom37_mask"].reshape(B, n_samples, L, -1)
        torsions = out["torsions"].reshape(B, n_samples, L, -1)
        return {
            "ca_samples": ca, "ca_mean": ca_mean_decoded,
            "ca_mean_decoded": ca_mean_decoded, "H_terminal": H_tau,
            "H_anchor": H_anchor, "H_traj": traj,          # full [BN, n+1, L, dH] path
            "n_steps": n_steps, "B": B, "n_samples": n_samples,
            "backbone_samples": backbone,
            "atom37_samples": atom37,
            "atom37_mask": atom37_mask,
            "torsion_samples": torsions,
        }


def _build_kernel(args, device) -> StageB1b:
    cfg = StageB1bConfig(
        latent_dim=args.latent_dim, bath_dim=args.bath_dim,
        energy_depth=args.energy_depth, energy_heads=args.energy_heads,
        energy_hidden=args.energy_hidden, history_hidden=args.history_hidden,
        coupling_scale=args.coupling_scale, diffusion_scale=args.diffusion_scale,
        c_init_std=args.c_init_std, gamma_scale=args.gamma_scale,
        gamma_floor=args.gamma_floor, max_gamma_step=args.max_gamma_step,
        force_clip=args.force_clip, tail_init=args.tail_init, step_ps=args.step_ps,
        rollout_grad_steps=args.rollout_grad_steps,
    )
    return StageB1b(cfg).to(device)


def _run_batch(producer, paired, args, weights, device, generator=None):
    targets, fut, fmask = _targets_from(paired, args, device)
    aatype = paired["future_aatype"].to(device)
    ri = paired["future_residue_index"].to(device)
    ci = paired["future_chain_id"].to(device)
    ctx = paired["context_coords"].to(device)
    ctx_mask = paired["context_residue_mask"].to(device)[:, -1]
    R_a, t_a = _anchors(paired, device)

    producer._rollout_steps = args.rollout_steps
    producer._rollout_grad_steps = args.rollout_grad_steps
    out = producer.sample_future_ca(
        ctx, ctx_mask, aatype, ri, ci, R_a, t_a,
        n_samples=args.n_samples, generator=generator,
        temperature_k=args.temperature_k,
    )
    ca = out["ca_samples"]
    pred_hist = targets.dist_hist_from_ca(ca)
    true_hist = targets.dist_hist_true()
    bin_w = (targets.dist_hi - targets.dist_lo) / (targets.dist_bins - 1)
    true_ca_pos = None
    pred_ca_pos = None
    if weights.pos_weight > 0.0:
        true_ca_pos = fut[:, -1, :, 1, :]
        pred_ca_pos = out["ca_mean_decoded"]
    comp = compute_ensemble_loss(
        ca, targets.rmsf, targets.contact, pred_hist, true_hist,
        targets.mask, bin_w, weights, cutoff=args.cutoff,
        true_ca_pos=true_ca_pos, pred_ca_pos=pred_ca_pos,
    )
    return comp, targets, out


def _decode_path(producer, H_traj, R_anchor, t_anchor, aatype, residue_mask,
                 B, n_samples, L, L_, step_idx):
    """Decode the CA cloud at the selected rollout steps of H_traj.

    H_traj: [B*n, n_steps+1, L, dH]; step_idx: 1-D tensor of rollout indices.
    Returns [B, n, T_sel, L, 3] CA in the anchor gauge (one cloud per step).
    """
    H_sel = H_traj.index_select(1, step_idx)                  # [BN, T_sel, L, dH]
    T_sel = H_sel.shape[1]
    flat = H_sel.reshape(B * n_samples * T_sel, L, -1)
    R = R_anchor[:, None, None].expand(B, n_samples, T_sel, L, 3, 3).reshape(
        B * n_samples * T_sel, L, 3, 3)
    t = t_anchor[:, None, None].expand(B, n_samples, T_sel, L, 3).reshape(
        B * n_samples * T_sel, L, 3)
    aa = aatype[:, None, None].expand(B, n_samples, T_sel, L).reshape(
        B * n_samples * T_sel, L)
    rm = residue_mask[:, None, None].expand(B, n_samples, T_sel, L).reshape(
        B * n_samples * T_sel, L)
    out = producer.decoder.decode_structure(flat, aa, R, t, rm)
    ca = out["backbone_coords"][:, :, 1, :]
    return ca.reshape(B, n_samples, T_sel, L, 3)


def _run_batch_dense(producer, paired, args, weights, device, generator=None):
    """Dense-trajectory objective: score EVERY rollout step against the truth.

    Uses a contiguous future window [anchor, t1, ..., tH] (``future_coords``), so
    rollout step i corresponds to truth frame i.  Losses (all reused from the
    from-none trainer, so behaviour is identical to the proven recipe):
      * per-step Kabsch CA point loss   -> beats persistence, per-frame encoder grad
      * energy score on the CA path     -> distributional (proper scoring rule)
      * optional ensemble rmsf / rmsf-corr / variance on the path.
    """
    from protein_jepa_frame.training.train_stage_cb1b import (
        weighted_ca_rmsd, persistence_ref_rmsd, normalized_path_es,
    )
    from protein_jepa_frame.models.stage_b1b_fromnone.objectives import (
        energy_score, ensemble_rmsf_loss, ensemble_rmsf_corr_loss, ensemble_variance_loss,
    )
    fut = paired["future_coords"].to(device)                  # [B, H+1, L, 4, 3] contiguous
    fmask = paired["future_residue_mask"].to(device)[:, -1].bool()   # [B, L]
    aatype = paired["future_aatype"].to(device)
    ri = paired["future_residue_index"].to(device)
    ci = paired["future_chain_id"].to(device)
    ctx = paired["context_coords"].to(device)
    ctx_mask = paired["context_residue_mask"].to(device)[:, -1]
    R_a, t_a = _anchors(paired, device)                       # anchor = ctx[-1] == fut[0]
    B = ctx.shape[0]

    producer._rollout_steps = args.rollout_steps
    producer._rollout_grad_steps = args.rollout_grad_steps
    out = producer.sample_future_ca(
        ctx, ctx_mask, aatype, ri, ci, R_a, t_a,
        n_samples=args.n_samples, generator=generator,
        temperature_k=args.temperature_k,
    )
    H_traj = out["H_traj"]
    n_samples, L = args.n_samples, ctx.shape[2]

    # Reuse the autonomous-rollout runtime guard: n_steps is exactly the observed
    # window length H, so the terminal step lands on the scored horizon.
    n_steps = out["n_steps"]
    horizon = fut.shape[1] - 1                                # H truth future frames
    if n_steps != horizon:
        raise ValueError(
            f"--dense-trajectory needs rollout_steps == future horizon; got "
            f"rollout_steps={n_steps}, future window={horizon}. Set "
            f"--rollout-steps {horizon} and --paired-horizons-frames {horizon} "
            f"(or use the contiguous loader).")
    step_idx = torch.arange(0, n_steps + 1, max(1, args.dense_stride),
                            device=device)
    ca_path = _decode_path(producer, H_traj, R_a, t_a, aatype, ctx_mask,
                           B, n_samples, L, None, step_idx)   # [B, n, T_sel, L, 3]
    true_path = fut[:, step_idx, :, 1, :]                     # [B, T_sel, L, 3]

    # (1) per-step Kabsch CA point loss (differentiable; anchored to real positions)
    ca_loss = weighted_ca_rmsd(ca_path, true_path, fmask)
    persist = persistence_ref_rmsd(true_path, ctx[:, -1, :, 1, :], fmask)

    # (2) energy score on the CA path: model samples [M] vs true cloud at each step.
    es = ca_path.new_zeros(())
    if args.dense_es_weight > 0:
        T_sel = len(step_idx)
        rows = []
        for b in range(B):
            mb = fmask[b]
            for tis in range(T_sel):
                smp = ca_path[b, :, tis][:, mb, :].reshape(n_samples, -1)   # [M, Lb*3]
                tgt = true_path[b, tis][mb, :].reshape(-1)
                rows.append(energy_score(smp, tgt) / (tgt.numel() ** 0.5))
        es = torch.stack(rows).mean()

    # (3) optional ensemble-profile terms on the path (needs M>=2 and T>=2)
    ens_rmsf = ens_rmsf_corr = ens_var = ca_loss.new_zeros(())
    if ca_path.shape[1] >= 2 and ca_path.shape[2] >= 2:
        if args.dense_rmsf_weight > 0:
            ens_rmsf = ensemble_rmsf_loss(ca_path, true_path, fmask)
        if args.dense_rmsf_corr_weight > 0:
            ens_rmsf_corr = ensemble_rmsf_corr_loss(ca_path, true_path, fmask)
        if args.dense_var_weight > 0:
            ens_var = ensemble_variance_loss(ca_path, true_path, fmask)

    total = (args.dense_ca_weight * ca_loss
             + args.dense_es_weight * es
             + args.dense_rmsf_weight * ens_rmsf
             + args.dense_rmsf_corr_weight * ens_rmsf_corr
             + args.dense_var_weight * ens_var)

    class _Comp:
        def __init__(self):
            self.total = total
        def as_loggable_dict(self):
            return {"ens/total": float(total.detach()),
                    "ens/ca": float(ca_loss.detach()),
                    "ens/es": float(es.detach()),
                    "ens/rmsf": float(ens_rmsf.detach()),
                    "ens/rmsf_corr": float(ens_rmsf_corr.detach()),
                    "ens/dist": float(ens_var.detach()),
                    "ens/persist": float(persist.detach())}
    return _Comp(), None, out


@torch.no_grad()
def _validate(producer, loader, args, weights, device, max_batches):
    producer.eval()
    from collections import defaultdict
    agg = defaultdict(float); pers = defaultdict(float); n = 0
    dense = bool(getattr(args, "dense_trajectory", False))
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        paired = _move_batch(batch, device)
        if dense:
            comp, _, _ = _run_batch_dense(producer, paired, args, weights, device)
        else:
            comp, _, _ = _run_batch(producer, paired, args, weights, device)
        for k, v in comp.as_loggable_dict().items():
            agg[k] += v
        if not dense:
            ctx = paired["context_coords"].to(device)
            pmask = paired["context_residue_mask"].to(device)
            pca = persistence_ensemble(ctx, args.n_samples, pmask)
            targets = _targets_from(paired, args, device)[0]
            phist = targets.dist_hist_from_ca(pca)
            thist = targets.dist_hist_true()
            bin_w = (targets.dist_hi - targets.dist_lo) / (targets.dist_bins - 1)
            pcomp = compute_ensemble_loss(
                pca, targets.rmsf, targets.contact, phist, thist,
                targets.mask, bin_w, weights, cutoff=args.cutoff)
            for k, v in pcomp.as_loggable_dict().items():
                pers[k] += v
        n += 1
    if n == 0:
        return {}
    out = {f"val/{k}": v / n for k, v in agg.items()}
    out.update({f"val_persist/{k}": v / n for k, v in pers.items()})
    return out


def _build_contiguous_loaders(args):
    """Contiguous-window loaders for --dense-trajectory (mirrors from-none cb1b).

    A single clip of K + rollout_steps frames is split by ``split_contiguous_window``
    into K history frames and a contiguous future [anchor, t1, ..., tH].  This is
    the from-none data protocol; here it is used only because it aligns rollout
    step i with truth frame i.
    """
    from protein_jepa_frame.data.collate import collate_clips
    from protein_jepa_frame.data.protocol import split_contiguous_window, validate_manifest_split

    K, H = args.history_frames, args.rollout_steps

    def make(manifest, training):
        cfg = DataConfig(
            manifest_path=manifest, batch_size=args.batch_size,
            num_workers=args.num_workers, training_mode="clip",
            num_frames=K + H, frame_stride=args.frame_stride,
            max_residues=args.max_residues,
        )
        ds = build_dataset(cfg)
        if args.max_train_samples and training:
            n_sub = min(len(ds), args.max_train_samples)
            stride = max(1, len(ds) // n_sub)
            ds = Subset(ds, list(range(0, len(ds), stride))[:n_sub])
        if args.smoke:
            ds = Subset(ds, list(range(0, min(len(ds), 16))))

        def _collate(samples):
            return split_contiguous_window(collate_clips(samples), K, H, step_ps=args.step_ps)

        if training:
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=_collate,
                            drop_last=True, pin_memory=True)
        else:
            total = len(ds)
            budget = args.val_batches * args.batch_size
            idx = torch.linspace(0, total - 1, min(total, max(budget, 1))).long().tolist()
            dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=_collate)
        return dl

    if args.manifest and args.val_manifest:
        validate_manifest_split(args.manifest, args.val_manifest)
    train_loader = make(args.manifest, True)
    val_loader = make(args.val_manifest, False) if args.val_manifest else None
    return train_loader, val_loader


def main():
    args = parse_args()
    _set_reproducibility(args.seed, False)
    torch.set_num_threads(args.torch_threads)
    device = _device(args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder, decoder, dH = load_frozen_stack(args.stage_a_ckpt, args.b1a_ckpt, device)
    if dH != args.latent_dim:
        print(f"[warn] overriding --latent-dim {args.latent_dim} -> frozen dH {dH}")
        args.latent_dim = dH
    kernel = _build_kernel(args, device)
    producer = HSEnsembleProducer(encoder, decoder, kernel,
                                  train_encoder=args.train_encoder).to(device)
    print(f"[planA] train_encoder={args.train_encoder} rollout_steps={args.rollout_steps} "
          f"encoder_lr_scale={args.encoder_lr_scale}", flush=True)

    weights = EnsembleLossWeights(
        rmsf_weight=args.rmsf_weight, contact_weight=args.contact_weight,
        dist_weight=args.dist_weight, pos_weight=args.pos_weight)
    weights.validate()

    # --- Dense-trajectory mode: contiguous future window [anchor, t1..tH] ---
    # The paired loader returns a FLAT future "window" (H frames at one stride),
    # which cannot be lined up 1:1 with rollout steps.  For dense supervision we
    # reuse the proven from-none contiguous protocol: a single clip of
    # K+rollout_steps frames split (via split_contiguous_window) into K history
    # frames + a contiguous future [anchor, t1, ..., tH] so step i == truth i.
    if args.dense_trajectory:
        loader, val_loader = _build_contiguous_loaders(args)
        print(f"[dense] contiguous window K={args.history_frames} H={args.rollout_steps} "
              f"stride={args.dense_stride}", flush=True)
    else:
        data_cfg = DataConfig(
            manifest_path=args.manifest, batch_size=args.batch_size,
            num_workers=args.num_workers, training_mode="paired_window",
            num_frames=args.history_frames, paired_all_horizons=False,
            max_residues=args.max_residues,
        )
        protocol = protocol_from_args(args)
        apply_protocol(data_cfg, protocol)
        # Plan A alignment guard: the terminal GLE step should land on the scored
        # window, so warn when the rollout length is inconsistent with the horizon.
        _h = min(args.paired_horizons_frames or protocol.horizons_frames)
        if args.rollout_steps > _h:
            print(f"[warn] --rollout-steps {args.rollout_steps} exceeds the shortest "
                  f"future horizon {_h}; the rollout is scored against a window it overruns.")
        elif args.rollout_steps < _h:
            print(f"[note] --rollout-steps {args.rollout_steps} < future horizon {_h}; the "
                  f"ensemble target spans a longer window than the rollout (set them equal "
                  f"for a matched objective).")
        dataset = build_dataset(data_cfg)
        if args.max_train_samples:
            n_sub = min(len(dataset), args.max_train_samples)
            stride = max(1, len(dataset) // n_sub)
            idx = list(range(0, len(dataset), stride))[:n_sub]
            dataset = Subset(dataset, idx)
        collate_fn = lambda s: _truncate_history(collate_paired_windows(s), args.history_frames)  # noqa: E731
        if args.smoke:
            dataset = Subset(dataset, list(range(0, min(len(dataset), 16))))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            drop_last=True, pin_memory=(device.type == "cuda"))
        if not len(loader):
            raise ValueError("training loader is empty")

        val_loader = None
        if args.val_manifest:
            val_cfg = replace(data_cfg, manifest_path=args.val_manifest)
            vd = build_dataset(val_cfg)
            n_val = min(len(vd), args.val_batches * args.batch_size)
            vidx = torch.linspace(0, len(vd) - 1, n_val).long().tolist()
            val_loader = DataLoader(Subset(vd, vidx), batch_size=args.batch_size,
                                    num_workers=0, shuffle=False, collate_fn=collate_fn)

    # Plan A: optionally fine-tune the (otherwise frozen) Stage-A encoder jointly
    # with the GLE kernel.  The decoder stays frozen either way.
    if args.train_encoder:
        groups = [
            {"params": [p for p in producer.encoder.parameters() if p.requires_grad],
             "lr": args.lr * args.encoder_lr_scale, "name": "encoder"},
            {"params": list(kernel.parameters()), "lr": args.lr, "name": "kernel"},
        ]
        groups = [g for g in groups if g["params"]]
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
        for pg in opt.param_groups:
            pg["base_lr"] = pg["lr"]
        clip_params = [p for g in groups for p in g["params"]]
    else:
        opt = torch.optim.AdamW(kernel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for pg in opt.param_groups:
            pg["base_lr"] = pg["lr"]
        clip_params = list(kernel.parameters())
    start_step, best_val = 0, float("inf")
    if args.resume:
        rck = torch.load(args.resume, map_location=device, weights_only=False)
        kernel.load_state_dict(rck["kernel"])
        if "encoder" in rck:
            producer.encoder.load_state_dict(rck["encoder"])
        try:
            opt.load_state_dict(rck["optimizer"])
        except Exception:  # noqa: BLE001
            pass
        start_step = int(rck.get("step", 0))
        best_val = float(rck.get("best_val", best_val))

    def save(tag, step, val):
        torch.save({
            "kernel": kernel.state_dict(), "optimizer": opt.state_dict(),
            "step": step, "best_val": val, "args": vars(args),
            "stage_a_ckpt": args.stage_a_ckpt, "b1a_ckpt": args.b1a_ckpt,
            "kernel_config": kernel.cfg.__dict__,
            "ensemble_weights": vars(weights),
            "train_encoder": bool(getattr(args, "train_encoder", False)),
            "encoder": (producer.encoder.state_dict()
                        if getattr(args, "train_encoder", False) else None),
        }, out_dir / f"checkpoint_{tag}.pt")

    t0 = time.time()
    step = start_step
    gen = torch.Generator(device=device).manual_seed(args.seed)
    while step < args.steps:
        for batch in loader:
            paired = _move_batch(batch, device)
            producer.train()
            # The energy module's attention backward is only implemented for the
            # MATH SDP backend (efficient/flash attention backward is missing).
            with sdpa_kernel([SDPBackend.MATH]):
                run = _run_batch_dense if args.dense_trajectory else _run_batch
                comp, _, out = run(producer, paired, args, weights, device, generator=gen)
            opt.zero_grad(set_to_none=True)
            comp.total.backward()
            torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            lr_scale = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * lr_scale
            opt.step()
            step += 1
            if step % args.log_every == 0:
                d = comp.as_loggable_dict()
                if args.dense_trajectory:
                    print(f"step {step:6d} | lr {opt.param_groups[0]['lr']:.2e} | "
                          f"{time.time()-t0:6.1f}s | total={d['ens/total']:.4f} "
                          f"ca={d['ens/ca']:.4f} es={d['ens/es']:.4f} "
                          f"persist={d['ens/persist']:.4f}", flush=True)
                else:
                    print(f"step {step:6d} | lr {opt.param_groups[0]['lr']:.2e} | "
                          f"{time.time()-t0:6.1f}s | ens/total={d['ens/total']:.4f} "
                          f"rmsf={d['ens/rmsf']:.4f} contact={d['ens/contact']:.4f} "
                          f"dist={d['ens/dist']:.4f}", flush=True)
            if val_loader is not None and step % args.val_every == 0:
                v = _validate(producer, val_loader, args, weights, device, args.val_batches)
                print(f"  [val {step}] {json.dumps({k: round(x,4) for k,x in v.items()})}",
                      flush=True)
                if v.get("val/ens/total", float("inf")) < best_val:
                    best_val = v["val/ens/total"]
                    save("best", step, best_val)
            if step % args.save_every == 0:
                save("latest", step, best_val)
            if step >= args.steps:
                break
    save("latest", step, best_val)
    print(f"done: {step} steps, best val ens/total = {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
