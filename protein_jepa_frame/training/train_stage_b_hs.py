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
import math
import time
from collections import defaultdict
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
    p.add_argument("--init-from", default=None,
                   help="Warm start all saved model weights with a fresh optimizer and step counter.")
    p.add_argument("--history-conditioning", action="store_true",
                   help="Initialize the GLE bath from encoded observed history and physical dt.")
    p.add_argument("--path-ensemble", action="store_true",
                   help="Match pooled decoded rollout to the same contiguous true path.")
    p.add_argument("--contact-temperature", type=float, default=0.5,
                   help="Soft-contact width in Angstrom; 0 reproduces historical hard contacts.")
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
    p.add_argument("--roundtrip-weight", type=float, default=0.0,
                   help="Weight of the decoder reconstruction term: Enc(true future "
                        "frame) -> Decoder -> CA vs the true CA. Supervises the H->CA "
                        "map directly on ground-truth conformations, which is where the "
                        "mean-collapse ceiling lives. Requires --train-decoder (and "
                        "--train-encoder if the encoder should adapt too). This is NOT "
                        "--dense-trajectory: the GLE rollout is left untouched, so "
                        "ensemble spread is not suppressed.")
    p.add_argument("--roundtrip-frames", type=int, default=1,
                   help="Evenly spaced true future frames used per round-trip step "
                        "(1 = terminal frame only; 2-4 = more of the horizon).")
    p.add_argument("--ens-rmsf-corr-weight", type=float, default=0.0,
                   help="Weight on the RMSF-PROFILE-CORRELATION (RMSF-r) loss added on "
                        "top of the terminal ensemble loss.  The default rmsf loss only "
                        "matches the RMSF MAGNITUDE; RMSF-r is the SHAPE of the flexibility "
                        "profile and is the metric bioemu2 currently wins (see report §3.9). "
                        "0 disables (current Lasp contract).")
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
    # The old default was 1.0, which clipped a force whose measured per-residue
    # magnitude is ~7.2, i.e. ~85% of the force was saturated.  That clamp was
    # only ever papering over the legacy bath's blow-up; with --bath-coupling
    # velocity the kernel is stable without it, and unclipped is measurably
    # closer to the true 1 ns step (6.54 vs 6.37) and keeps |H| on the anchor
    # manifold (11.3 -> 11.7 over 64 steps, vs 11.3 -> 17.6 clipped).
    p.add_argument("--force-clip", type=float, default=None)
    # Bath convention.  "force" is the legacy divergence-prone kernel
    # (mu_H = H + h*D*f + h*C*S  =>  mobility D + C*Gamma^-1*C^T > D);
    # "velocity" is FDT-correct and is what new runs must use.
    p.add_argument("--bath-coupling", choices=["force", "velocity"], default="velocity")
    # Guard on the per-residue step |dH| in latent units.  A real one-ns frame
    # moves H by ~6.9/residue, so 20 never fires on a healthy rollout.
    p.add_argument("--max-step-norm", type=float, default=20.0)
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
    p.add_argument("--train-decoder", action="store_true", default=False,
                   help="Also unfreeze the B1a anchored decoder at a low LR.  Default off == "
                        "decoder frozen (the exact Model-1 contract: H->CA map preserved).  "
                        "Turn this on to test whether joint adaptation of the decode map helps "
                        "the rollout-fit ensemble metrics -- the frozen decoder is a real "
                        "bottleneck, but unfreezing risks Model 1's fidelity, so ALWAYS keep a "
                        "frozen-decoder control alongside an unfrozen run.")
    p.add_argument("--decoder-lr-scale", type=float, default=0.1,
                   help="LR multiplier for the decoder group when --train-decoder is set.")
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
    # --- Trajectory-level supervision (terminal-anchored + path distribution) ---
    # Different from --dense-trajectory: the primary terminal ensemble loss is
    # KEPT and trajectory-level *distributional* terms are ADDED on the decoded
    # path.  We deliberately do NOT re-introduce the dense per-step Kabsch point
    # loss (that made things worse, see PROJECT_REPORT §10.2): pushing the model
    # toward the single true frame per step suppresses the stochastic spread.
    # Uses the SAME contiguous loader as --dense-trajectory (exact step i <-> frame i).
    p.add_argument("--traj-loss", action="store_true", default=False,
                   help="Add order-aware trajectory-level distributional losses on the "
                        "GLE rollout path (path energy score + walk RMSF-corr), on top of "
                        "the terminal ensemble loss.  Use WITH the contiguous window "
                        "(set --paired-horizons-frames == --rollout-steps and pass "
                        "--use-contiguous-window).")
    p.add_argument("--use-contiguous-window", action="store_true", default=False,
                   help="Load the CONTIGUOUS future window [anchor,t1..tH] (K+H clip) so "
                        "rollout step i lines up with truth frame i; required by "
                        "--traj-loss (and implied by --dense-trajectory).")
    p.add_argument("--traj-es-weight", type=float, default=0.1,
                   help="Weight on the trajectory energy score (path distance-marginal "
                        "energy score over the pooled future frames).")
    p.add_argument("--traj-rmsf-corr-weight", type=float, default=0.5,
                   help="Weight on the walk RMSF-profile correlation (predicted chain "
                        "flexibility from the pooled path vs the true pool).")
    p.add_argument("--traj-rmsf-weight", type=float, default=0.0,
                   help="Weight on the walk RMSF magnitude (L1 on pooled-path RMSF).")
    p.add_argument("--traj-var-weight", type=float, default=0.0,
                   help="Weight on the walk ensemble-variance loss (pooled path).")
    p.add_argument("--traj-anchor-loss", action="store_true", default=False,
                   help="Additionally penalise the rollout's step-0 (anchor replay) drift, "
                        "i.e. decode H_traj[:,0] and require it to equal the real anchor.")
    p.add_argument("--traj-stride", type=int, default=1,
                   help="With --traj-loss, decode/supervise every k-th rollout frame "
                        "(1 = all; raise it to bound memory at long horizons).")
    # truncated BPTT: keep only the last k transitions differentiable
    p.add_argument("--rollout-grad-steps", type=int, default=None)
    # --- Multi-horizon terminal training ---
    # Lasp (single-horizon H) matches Model 1 and beats it almost everywhere, but
    # at long horizons it OVER-disperses relative to bioemu2 because the GLE was
    # only ever fit to land on one window length.  Training the SAME terminal
    # ensemble objective at several horizons (with truncated BPTT so the long
    # rollouts stay in memory) lets the kernel learn a horizon-dependent rate.
    p.add_argument("--multi-horizon", action="store_true", default=False,
                   help="Sample a horizon per batch from --paired-horizons-frames and train "
                        "the terminal ensemble loss at THAT horizon (rollout_steps := h). "
                        "Batches are horizon-homogeneous.  Combine with --rollout-grad-steps "
                        "to truncate BPTT at long horizons.")
    p.add_argument("--multi-horizon-max-grad", type=int, default=None,
                   help="With --multi-horizon, horizons longer than this use truncated BPTT "
                        "(grad_steps = this value); horizons at or below it keep the full "
                        "graph.  None keeps the full graph at every horizon (memory-heavy).")
    p.add_argument("--temperature-k", type=float, default=320.0,
                   help="static T used for the GLE kernel when "
                        "--per-sample-temperature is off (mdCATH default)")
    p.add_argument("--per-sample-temperature", action="store_true", default=False,
                   help="Condition each sample on ITS OWN manifest temperature (320/348/"
                        "379/413/450 K) instead of the single --temperature-k value. "
                        "Without this, 4/5 of an mdCATH batch is mislabelled and the "
                        "kernel learns to ignore the temperature input entirely.")

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

    def __init__(self, encoder, decoder, kernel: StageB1b, train_encoder: bool = False,
                 train_decoder: bool = False, history_conditioning: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kernel = kernel
        self.history_conditioning = bool(history_conditioning)
        for p in self.encoder.parameters():
            p.requires_grad_(bool(train_encoder))
        for p in self.decoder.parameters():
            p.requires_grad_(bool(train_decoder))
        # When the encoder is frozen we can (and should) run it under no_grad;
        # when it is fine-tuned we must keep the graph so the GLE losses reach it.
        self._encoder_grad = bool(train_encoder)
        # Decoder grad is separate: this is the exact preserved branch that keeps
        # the H->CA map intact.  When it is trainable the decode is differentiable
        # (never wrapped in no_grad) and the decoder parameters join the optimizer.
        self._decoder_grad = bool(train_decoder)

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
        temperature_k=None, context_time_ps=None,
    ) -> dict:
        """Sample ``n_samples`` future CA clouds per batch element.

        ``temperature_k`` may be a SCALAR, a per-sample tensor/list of shape [B],
        or None.  Per-sample is what training and evaluation must use: mdCATH
        spans 320-450 K, and collapsing that axis to one constant both mislabels
        4/5 of the data and teaches the kernel to ignore the conditioning input.
        """
        B, K, L, _, _ = context_coords.shape
        device = context_coords.device
        anchor_coords = context_coords[:, -1]
        H_anchor = self.encode_frame(anchor_coords, residue_mask, aatype,
                                     residue_index, chain_id)             # [B,L,dH]
        if temperature_k is None:
            temp = torch.full((B,), 320.0, device=device)
        else:
            temp = torch.as_tensor(temperature_k, device=device, dtype=torch.float32)
            if temp.dim() == 0:
                temp = temp.expand(B)
            elif temp.shape[0] == 1 and B != 1:
                temp = temp.expand(B)
            elif temp.shape[0] != B:
                raise ValueError(
                    f"temperature_k has {temp.shape[0]} entries for batch size {B}")
            temp = temp.reshape(B)

        flat_hist = flat_dt = None
        if self.history_conditioning:
            if context_time_ps is None:
                raise ValueError("history conditioning requires observed context_time_ps")
            times = torch.as_tensor(context_time_ps, device=device)
            if times.shape != (B, K) or not torch.isfinite(times).all():
                raise ValueError("context timestamps must be finite [B,K]")
            intervals = times.diff(dim=1)
            if K < 2 or not (intervals > 0).all():
                raise ValueError("history conditioning requires at least two increasing timestamps")
            dt = torch.cat((intervals[:, :1], intervals), dim=1)
            hist = torch.stack([
                self.encode_frame(context_coords[:, k], residue_mask, aatype,
                                  residue_index, chain_id) for k in range(K - 1)
            ] + [H_anchor], dim=1)
            flat_hist = hist[:, None].expand(B, n_samples, K, L, hist.shape[-1]).reshape(
                B * n_samples, K, L, hist.shape[-1])
            flat_dt = dt[:, None].expand(B, n_samples, K).reshape(B * n_samples, K)

        # Roll out `n_samples` independent GLE trajectories in parallel.
        Hs = H_anchor[:, None].expand(B, n_samples, L, H_anchor.shape[-1])
        flat_H = Hs.reshape(B * n_samples, L, H_anchor.shape[-1])
        flat_mask = residue_mask[:, None].expand(B, n_samples, L).reshape(B * n_samples, L)
        flat_T = temp[:, None].expand(B, n_samples).reshape(B * n_samples)
        return self._rollout_decode(flat_H, flat_mask, flat_T, B, n_samples, L,
                                    R_anchor, t_anchor, aatype, residue_mask,
                                    H_anchor, generator, flat_hist, flat_dt)

    def _rollout_decode(self, flat_H, flat_mask, flat_T, B, n_samples, L,
                        R_anchor, t_anchor, aatype, residue_mask, H_anchor, generator,
                        H_hist=None, dt_hist=None):
        n_steps = getattr(self, "_rollout_steps", 1)
        grad_steps = getattr(self, "_rollout_grad_steps", None)
        traj = self.kernel.rollout(
            flat_H, flat_mask, flat_T, n_steps,
            H_hist=H_hist, dt_hist=dt_hist,
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
        bath_coupling=args.bath_coupling, max_step_norm=args.max_step_norm,
        rollout_grad_steps=args.rollout_grad_steps,
    )
    return StageB1b(cfg).to(device)


def _batch_temperature(paired, args, device):
    """Per-sample temperature for the batch: [B] tensor, or the pinned scalar.

    ``--temperature-k`` used to be fed to EVERY sample regardless of the
    trajectory it came from, so a batch mixing 320 K and 450 K trajectories
    trained the kernel on a single, wrong, conditioning value.  With
    ``--per-sample-temperature`` each sample carries its own manifest value; the
    flag stays opt-in so previously reported runs remain reproducible.
    """
    if not getattr(args, "per_sample_temperature", False):
        return args.temperature_k
    t = paired.get("temperature_k")
    if t is None:
        return args.temperature_k
    t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
    if not torch.isfinite(t).all() or (t <= 0).any():
        raise ValueError("per-sample temperature has non-finite/non-positive entries")
    return t


def _roundtrip_loss(producer, paired, fut, fmask, R_a, t_a, aatype, args, device):
    """Decoder reconstruction loss on TRUE conformations: Enc(X_t) -> Dec -> X_t.

    Why this exists.  The end-to-end ensemble loss only ever asks the decoder to
    reproduce the *terminal* cloud that the GLE rollout produced, so the decoder
    is free to map a wide range of H onto nearly the same CA shape (mean
    collapse).  Measured on the 1000dom checkpoint, encoding a TRUE future frame
    and decoding it back recovers only 38-41 % of the true RMSF at 16-64 ns with
    RMSF-r ~0.38-0.41 -- i.e. the H->CA map itself is the bottleneck, and it is
    *worse* than the rollout it is supposed to render.

    This term supervises that map directly on ground-truth conformations, so the
    decoder must be sensitive to conformational differences in H.  Gradients go
    to the decoder and (when unfrozen) to the encoder.

    Deliberately NOT the same thing as ``--dense-trajectory``: that loss scored
    every ROLLOUT step against truth and was reverted (§10.2 of the report)
    because it suppressed ensemble spread by dragging samples onto one path.
    Here the GLE rollout is untouched; only the static H->CA decoder is trained,
    and its input is the true frame's own encoding.
    """
    weight = float(getattr(args, "roundtrip_weight", 0.0))
    if weight <= 0.0:
        return None
    from protein_jepa_frame.training.train_stage_cb1b import weighted_ca_rmsd

    # _run_batch's fmask is [B, T, L]; the encoder/decoder and weighted_ca_rmsd all
    # want the single-frame [B, L] residue mask.
    rmask = fmask[:, -1].bool() if fmask.dim() == 3 else fmask.bool()

    B, T, L = fut.shape[0], fut.shape[1], fut.shape[2]
    if T - 1 < 1:                      # need at least one non-anchor frame
        return None
    n_frames = max(1, int(getattr(args, "roundtrip_frames", 1)))
    # Evenly spaced, deterministic coverage of the scored horizon (frame 0 is the
    # anchor and is already exact by construction, so it is excluded).
    steps = min(n_frames, T - 1)
    idx = torch.linspace(1, T - 1, steps=steps).round().long().tolist()

    ri = paired["future_residue_index"].to(device)
    ci = paired["future_chain_id"].to(device)
    total = fut.new_zeros(())
    for i in idx:
        frame = fut[:, i]                                   # [B,L,4,3]
        H_true = producer.encode_frame(frame, rmask, aatype, ri, ci)   # [B,L,dH]
        out = producer.decoder.decode_structure(H_true, aatype, R_a, t_a, rmask)
        pred_ca = out["backbone_coords"][:, :, 1, :]        # [B,L,3]
        true_ca = frame[:, :, 1, :]                         # [B,L,3]
        total = total + weighted_ca_rmsd(pred_ca[None, :, None], true_ca[:, None], rmask)
    return total / len(idx)


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
        temperature_k=_batch_temperature(paired, args, device),
        context_time_ps=paired.get("context_time_ps"),
    )
    ca = out["ca_samples"]
    if getattr(args, "path_ensemble", False):
        horizon = fut.shape[1] - 1
        if horizon != out["n_steps"]:
            raise ValueError("path ensemble requires a contiguous [anchor,...,horizon] target")
        step_idx = torch.arange(horizon + 1, device=device)
        path = _decode_path(producer, out["H_traj"], R_a, t_a, aatype,
                            ctx_mask, ctx.shape[0], args.n_samples, ctx.shape[2], None, step_idx)
        ca = path.flatten(1, 2)
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
    # Optional RMSF-profile correlation term: the ensemble RMSF loss above matches
    # only the per-residue RMSF MAGNITUDE; rmsf_corr (RMSF-r, the metric bioemu2
    # wins) is the SHAPE of the flexibility profile.  ensemble_rmsf_corr_loss is a
    # differentiable Pearson-r surrogate identical to the evaluator's definition.
    w_rc = float(getattr(args, "ens_rmsf_corr_weight", 0.0))
    if w_rc > 0.0:
        from protein_jepa_frame.models.stage_b1b_fromnone.objectives import (
            ensemble_rmsf_corr_loss)
        if fut.shape[1] < 2:
            raise ValueError("RMSF correlation requires multiple true frames")
        true_window = fut[:, :, :, 1, :]
        l_rc = ensemble_rmsf_corr_loss(ca.permute(1, 0, 2, 3).unsqueeze(2),
                                       true_window, targets.mask)
        comp = _MaxNose(comp, w_rc * l_rc, extra={"ens/rmsf_corr": l_rc})
    # Decoder round-trip: Enc(true X_t) -> Dec -> X_t.  Removes the H->CA
    # mean-collapse ceiling without touching the GLE rollout (see
    # _roundtrip_loss for why this is not the reverted --dense-trajectory loss).
    w_rt = float(getattr(args, "roundtrip_weight", 0.0))
    if w_rt > 0.0:
        l_rt = _roundtrip_loss(producer, paired, fut, fmask, R_a, t_a,
                               aatype, args, device)
        if l_rt is not None:
            comp = _MaxNose(comp, w_rt * l_rt, extra={"ens/roundtrip": l_rt})
    return comp, targets, out


class _MaxNose:
    """Wrap an EnsembleLossComponents and add a term (keeping base logs)."""
    def __init__(self, base, term, extra=None):
        self._base = base
        self.total = base.total + term
        self._term = term
        self._extra = extra or {}

    def as_loggable_dict(self):
        d = dict(self._base.as_loggable_dict())
        for k, v in self._extra.items():
            d[k] = float(v.detach())
        d["ens/total"] = float(self.total.detach())
        return d


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
        temperature_k=_batch_temperature(paired, args, device),
        context_time_ps=paired.get("context_time_ps"),
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
    # The from-none losses expect the sample axis FIRST: [M, B, T, L, 3].
    ca_path = ca_path.permute(1, 0, 2, 3, 4).contiguous()     # [n_samples, B, T_sel, L, 3]
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
                smp = ca_path[:, b, tis][:, mb, :].reshape(n_samples, -1)   # [M, Lb*3]
                tgt = true_path[b, tis][mb, :].reshape(-1)
                rows.append(energy_score(smp, tgt) / (tgt.numel() ** 0.5))
        es = torch.stack(rows).mean()

    # (3) optional ensemble-profile terms on the path (needs M>=2 and T>=2)
    ens_rmsf = ens_rmsf_corr = ens_var = ca_loss.new_zeros(())
    if ca_path.shape[0] >= 2 and ca_path.shape[2] >= 2:
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


def _run_batch_traj(producer, paired, args, weights, device, generator=None):
    """Terminal ensemble loss + trajectory-level distributional losses.

    Kept: the proven terminal ensemble loss (``_run_batch``), which scores the
    decoded terminal H_T cloud against the true future window -- this is what
    makes the frozen/unfrozen Plan-A runs match Model 1 on the ensemble metrics.

    Added: order-aware path terms on the WHOLE decoded rollout path, built from
    the *distribution* along the path, NOT per-step point matching:
      * ``traj energy score``  -- proper scoring rule on the predicted vs true
        pool of future frames (spread + accuracy jointly);
      * ``walk RMSF-corr / RMSF / var`` -- pooled-path flexibility profile.
    This deliberately DROPS the dense per-step Kabsch point loss, which was shown
    to hurt (PROJECT_REPORT §10.2): fitting each step to one true frame kills the
    stochastic spread the ensemble metrics reward.
    """
    from protein_jepa_frame.models.stage_b1b_fromnone.objectives import (
        ensemble_rmsf_loss, ensemble_rmsf_corr_loss, ensemble_variance_loss,
        energy_score,
    )
    from protein_jepa_frame.training.train_stage_cb1b import weighted_ca_rmsd
    # --- (1) terminal ensemble loss (exactly _run_batch) ---
    comp_term, targets, out = _run_batch(producer, paired, args, weights, device, generator)
    total = comp_term.total

    # --- (2) trajectory-level distributional losses on the decoded path ---
    fut = paired["future_coords"].to(device)                 # [B, H+1, L, 4, 3] contiguous
    fmask = paired["future_residue_mask"].to(device)[:, -1].bool()   # [B, L]
    ctx = paired["context_coords"].to(device)
    ctx_mask = paired["context_residue_mask"].to(device)[:, -1]
    aatype = paired["future_aatype"].to(device)
    ri = paired["future_residue_index"].to(device)
    ci = paired["future_chain_id"].to(device)
    R_a, t_a = _anchors(paired, device)
    B, L = ctx.shape[0], ctx.shape[2]
    M = args.n_samples
    H_traj = out["H_traj"]
    n_steps = out["n_steps"]
    horizon = fut.shape[1] - 1
    if n_steps != horizon:
        raise ValueError(f"--traj-loss needs rollout_steps == future horizon; got "
                         f"rollout_steps={n_steps}, window={horizon}.")
    step_idx = torch.arange(0, n_steps + 1, max(1, args.traj_stride), device=device)
    ca_path = _decode_path(producer, H_traj, R_a, t_a, aatype, ctx_mask,
                           B, M, L, None, step_idx)          # [B, M, T, L, 3]
    ca_flat = ca_path.permute(1, 0, 2, 3, 4).contiguous()    # [M, B, T, L, 3]
    true_path = fut[:, step_idx, :, 1, :]                    # [B, T, L, 3]

    # pooled-path terms (M>=2, T>=2)
    l_es = l_rc = l_rf = l_var = l_anchor = ca_flat.new_zeros(())
    if ca_flat.shape[0] >= 2 and ca_flat.shape[2] >= 2:
        if args.traj_es_weight > 0:
            # Energy score of the pooled CA cloud (all path frames together) vs the
            # true future window cloud -- same proper scoring rule as the ensemble
            # trainer, but over the whole trajectory rather than the terminal frame.
            pooled_pred = ca_flat.permute(0, 1, 2, 3, 4)          # [M,B,T,L,3]
            rows = []
            for b in range(B):
                mb = fmask[b]
                smp = pooled_pred[:, b][:, :, mb, :].reshape(M, -1)   # [M, T*Lv*3]
                tgt = true_path[b][:, mb, :].reshape(-1)              # [T*Lv*3]
                rows.append(energy_score(smp, tgt) / (tgt.numel() ** 0.5))
            l_es = torch.stack(rows).mean()
        if args.traj_rmsf_corr_weight > 0:
            l_rc = ensemble_rmsf_corr_loss(ca_flat, true_path, fmask)
        if args.traj_rmsf_weight > 0:
            l_rf = ensemble_rmsf_loss(ca_flat, true_path, fmask)
        if args.traj_var_weight > 0:
            l_var = ensemble_variance_loss(ca_flat, true_path, fmask)
    if args.traj_anchor_loss:
        anchor_true = ctx[:, -1, :, 1, :]                     # [B, L, 3]
        # ca_path[:, :, 0] is [B, M, L, 3]; weighted_ca_rmsd wants [M, B, T, L, 3].
        ca_anchor = ca_path[:, :, 0].permute(1, 0, 2, 3).unsqueeze(2)   # [M,B,1,L,3]
        l_anchor = weighted_ca_rmsd(ca_anchor, anchor_true[:, None], fmask)

    total = (total
             + args.traj_es_weight * l_es
             + args.traj_rmsf_corr_weight * l_rc
             + args.traj_rmsf_weight * l_rf
             + args.traj_var_weight * l_var
             + (1.0 if args.traj_anchor_loss else 0.0) * l_anchor)

    class _Comp:
        def __init__(self, base):
            self.total = total
            self._base = base
        def as_loggable_dict(self):
            d = dict(self._base.as_loggable_dict())
            d["ens/total"] = float(total.detach())
            d["traj/es"] = float(l_es.detach())
            d["traj/rmsf_corr"] = float(l_rc.detach())
            d["traj/rmsf"] = float(l_rf.detach())
            d["traj/var"] = float(l_var.detach())
            d["traj/anchor"] = float(l_anchor.detach())
            return d
    return _Comp(comp_term), targets, out


@torch.no_grad()
def _validate(producer, loader, args, weights, device, max_batches):
    producer.eval()
    from collections import defaultdict
    agg = defaultdict(float); pers = defaultdict(float); n = 0
    dense = bool(getattr(args, "dense_trajectory", False))
    traj = bool(getattr(args, "traj_loss", False))
    val_generator = torch.Generator(device=device).manual_seed(args.seed + 100003)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        paired = _move_batch(batch, device)
        if dense:
            comp, _, _ = _run_batch_dense(producer, paired, args, weights, device, generator=val_generator)
        elif traj:
            comp, _, _ = _run_batch_traj(producer, paired, args, weights, device, generator=val_generator)
        else:
            comp, _, _ = _run_batch(producer, paired, args, weights, device, generator=val_generator)
        for k, v in comp.as_loggable_dict().items():
            agg[k] += v
        if not dense:
            # persistence reference: fine here for BOTH _run_batch and _run_batch_traj
            # (they read the same paired["context_*"]/["future_*"] keys).
            ctx = paired["context_coords"].to(device)
            pmask = paired["context_residue_mask"].to(device)
            pca = persistence_ensemble(ctx, args.n_samples, pmask)
            targets = _targets_from(paired, args, device)[0]
            phist = targets.dist_hist_from_ca(pca)
            thist = targets.dist_hist_true()
            bin_w = (targets.dist_hi - targets.dist_lo) / (targets.dist_bins - 1)
            # The persistence baseline has no predicted structure to guide, so
            # score it with the same weights minus the atom-position term;
            # otherwise pos_weight>0 makes the *diagnostic* raise.
            pw = weights.pos_weight
            weights.pos_weight = 0.0
            try:
                pcomp = compute_ensemble_loss(
                    pca, targets.rmsf, targets.contact, phist, thist,
                    targets.mask, bin_w, weights, cutoff=args.cutoff)
            finally:
                weights.pos_weight = pw
            for k, v in pcomp.as_loggable_dict().items():
                pers[k] += v
        n += 1
    if n == 0:
        return {}
    out = {f"val/{k}": v / n for k, v in agg.items()}
    out.update({f"val_persist/{k}": v / n for k, v in pers.items()})
    # Checkpoint selection must be comparable across runs.  `ens/pos` is an
    # atom-position MSE in A^2 and is only present when --pos-weight > 0, so
    # including it makes val/ens/total jump from O(4) to O(40) and silently
    # changes which step is "best".  Selecting on the base objective (rmsf +
    # contact + dist), i.e. total minus the position contribution, keeps every
    # run -- with or without the guide -- on one scale.  Weight 0 is a no-op.
    pw = float(getattr(weights, "pos_weight", 0.0))
    out["val/selection"] = out["val/ens/total"] - pw * out.get("val/ens/pos", 0.0)
    return out


def _build_multi_horizon_loaders(args):
    """Horizon-homogeneous paired-window loaders for --multi-horizon training.

    Builds ONE ``PairedWindowDataset`` per horizon (so every batch is a single
    horizon, since mixed horizons cannot be stacked into one ``future_coords``
    tensor), each using the normal paired loader and the standard
    ``collate_paired_windows``.  The training loop then picks a horizon per step.
    """
    from protein_jepa_frame.config import DataConfig
    from protein_jepa_frame.data.collate import collate_paired_windows
    from protein_jepa_frame.data.dataset import build_dataset
    from protein_jepa_frame.data.protocol import apply_protocol, protocol_from_args
    from protein_jepa_frame.training.train_stage_b1a import _truncate_history
    from torch.utils.data import DataLoader, Subset

    horizons = sorted({int(h) for h in (args.paired_horizons_frames or [8])})
    loaders = {}
    for h in horizons:
        if args.use_contiguous_window or args.traj_loss:
            # Contiguous K+H clip -> [K history | anchor, t1..tH]: rollout step i
            # aligns with truth frame i, so trajectory-level losses are well-posed.
            from protein_jepa_frame.data.collate import collate_clips
            from protein_jepa_frame.data.protocol import split_contiguous_window
            K = args.history_frames
            ccfg = DataConfig(manifest_path=args.manifest, batch_size=args.batch_size,
                              num_workers=args.num_workers, training_mode="clip",
                              num_frames=K + h, frame_stride=args.frame_stride,
                              max_residues=args.max_residues)
            cds = build_dataset(ccfg)
            if args.max_train_samples:
                n_sub = min(len(cds), args.max_train_samples)
                st = max(1, len(cds) // n_sub)
                cds = Subset(cds, list(range(0, len(cds), st))[:n_sub])
            if args.smoke:
                cds = Subset(cds, list(range(0, min(len(cds), 16))))
            if not len(cds):
                print(f"[multi-horizon] H={h}: empty contiguous dataset, skip", flush=True)
                continue
            cfn = lambda s, hh=h: split_contiguous_window(collate_clips(s), K, hh, step_ps=args.step_ps)  # noqa: E731
            cdl = DataLoader(cds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=cfn,
                             drop_last=True, pin_memory=True)
            if len(cdl):
                loaders[h] = cdl
                print(f"[multi-horizon|contig] H={h}: {len(cds)} windows, {len(cdl)} batches",
                      flush=True)
            continue
        class _P: pass
        pr = _P()
        pr.history_frames = args.history_frames
        pr.frame_stride = args.frame_stride
        pr.paired_horizons_frames = [h]
        cfg = DataConfig(manifest_path=args.manifest, batch_size=args.batch_size,
                         num_workers=args.num_workers, training_mode="paired_window",
                         num_frames=args.history_frames, paired_all_horizons=False,
                         max_residues=args.max_residues)
        apply_protocol(cfg, protocol_from_args(pr))
        ds = build_dataset(cfg)
        if args.max_train_samples:
            n_sub = min(len(ds), args.max_train_samples)
            stride = max(1, len(ds) // n_sub)
            ds = Subset(ds, list(range(0, len(ds), stride))[:n_sub])
        if args.smoke:
            ds = Subset(ds, list(range(0, min(len(ds), 16))))
        if not len(ds):
            print(f"[multi-horizon] horizon {h}: empty dataset, skipping", flush=True)
            continue
        collate_fn = lambda s: _truncate_history(collate_paired_windows(s), args.history_frames)  # noqa: E731
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        drop_last=True, pin_memory=True)
        if len(dl):
            loaders[h] = dl
            print(f"[multi-horizon] H={h}: {len(ds)} windows, {len(dl)} batches", flush=True)
    if not loaders:
        raise ValueError("--multi-horizon produced no non-empty horizon loaders")
    return loaders, horizons


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

    if not (args.use_contiguous_window or args.traj_loss or args.dense_trajectory):
        from protein_jepa_frame.data.dataset import validate_real_paired_window_protocol
        validate_real_paired_window_protocol(DataConfig(
            manifest_path=args.manifest, num_frames=args.history_frames,
            frame_stride=args.frame_stride,
            paired_horizons_frames=tuple(args.paired_horizons_frames or (8,16,24,32,48,64,96))))
    if args.manifest and args.val_manifest:
        validate_manifest_split(args.manifest, args.val_manifest)
    train_loader = make(args.manifest, True)
    val_loader = make(args.val_manifest, False) if args.val_manifest else None
    return train_loader, val_loader


def _safe_step(opt, params, loss, grad_clip, lr_groups=None):
    """Backward + clip + step, refusing to poison the weights with a non-finite loss.

    ``clip_grad_norm_`` cannot help here: the norm of a NaN gradient is NaN, and
    ``clamp(max=clip)`` of NaN is still NaN, so a single bad batch propagates
    NaN into every parameter and every later step is wasted.  Measured on the
    2026-10-22 A/B set, ``lasp_v3_velocity`` hit this at step 8,630 and kept
    "training" for 5,000+ more steps with 64/73 kernel tensors NaN;
    ``lasp_v5_pos03`` hit it at 12,700.  The loss values immediately before the
    failure were unremarkable, so the trigger is bursty and cannot be avoided by
    tuning the learning rate -- it has to be detected.

    Returns (ok, grad_norm).  On ``ok=False`` the optimizer state is untouched.
    """
    if not torch.isfinite(loss):
        return False, float("nan")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(params, grad_clip)
    gn = float(gn) if torch.is_tensor(gn) else float(gn)
    if not math.isfinite(gn):
        opt.zero_grad(set_to_none=True)
        return False, gn
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            opt.zero_grad(set_to_none=True)
            return False, gn
    if lr_groups is not None:
        for pg in opt.param_groups:
            pg["lr"] = pg["base_lr"] * lr_groups
    opt.step()
    return True, gn


def main():
    args = parse_args()
    if args.path_ensemble:
        args.use_contiguous_window = True
    if args.history_conditioning and args.history_frames < 2:
        raise ValueError("history conditioning requires at least two history frames")
    if args.manifest and args.val_manifest:
        from protein_jepa_frame.data.protocol import validate_manifest_split
        validate_manifest_split(args.manifest, args.val_manifest)
    # Fail before loading any checkpoint: a round-trip loss with a frozen decoder
    # cannot move a single parameter, so the run would silently be a no-op.
    if float(getattr(args, "roundtrip_weight", 0.0)) > 0.0 and not args.train_decoder:
        raise SystemExit(
            "--roundtrip-weight > 0 requires --train-decoder: the loss trains the "
            "decoder's H->CA map and has no effect on a frozen decoder.")
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
                                  train_encoder=args.train_encoder,
                                  train_decoder=args.train_decoder,
                                  history_conditioning=args.history_conditioning).to(device)
    print(f"[planA] train_encoder={args.train_encoder} train_decoder={args.train_decoder} "
          f"rollout_steps={args.rollout_steps} encoder_lr_scale={args.encoder_lr_scale} "
          f"decoder_lr_scale={args.decoder_lr_scale}", flush=True)

    weights = EnsembleLossWeights(
        rmsf_weight=args.rmsf_weight, contact_weight=args.contact_weight,
        dist_weight=args.dist_weight, pos_weight=args.pos_weight,
        contact_temperature=args.contact_temperature)
    weights.validate()

    # --- Multi-horizon terminal training (paired loaders, one per horizon) ---
    if args.multi_horizon:
        if args.dense_trajectory:
            raise ValueError("--multi-horizon cannot be combined with "
                             "--dense-trajectory (use one objective)")
        mh_loaders, mh_horizons = _build_multi_horizon_loaders(args)
        loader, val_loader = None, None  # validation uses a horizon-paired loader below
        print(f"[multi-horizon] horizons={mh_horizons}", flush=True)
    # --- Dense-trajectory mode: contiguous future window [anchor, t1..tH] ---
    # The paired loader returns a FLAT future "window" (H frames at one stride),
    # which cannot be lined up 1:1 with rollout steps.  For dense supervision we
    # reuse the proven from-none contiguous protocol: a single clip of
    # K+rollout_steps frames split (via split_contiguous_window) into K history
    # frames + a contiguous future [anchor, t1, ..., tH] so step i == truth i.
    elif args.dense_trajectory or args.use_contiguous_window or args.traj_loss:
        loader, val_loader = _build_contiguous_loaders(args)
        print(f"[contig] window K={args.history_frames} H={args.rollout_steps} "
              f"dense={args.dense_trajectory} traj_loss={args.traj_loss} "
              f"stride={args.dense_stride if args.dense_trajectory else args.traj_stride}",
              flush=True)
    else:
        data_cfg = DataConfig(
            manifest_path=args.manifest, batch_size=args.batch_size,
            num_workers=args.num_workers, training_mode="paired_window",
            num_frames=args.history_frames, paired_all_horizons=False,
            max_residues=args.max_residues,
        )
        # protocol_from_args reads paired_horizons_frames; None would crash it,
        # so fall back to the canonical protocol grid when unset.
        if args.paired_horizons_frames is None:
            args.paired_horizons_frames = [8, 16, 24, 32, 48, 64, 96]
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

    # Multi-horizon validation: score the terminal objective at EVERY training
    # horizon and average, so "best" selection rewards horizon calibration rather
    # than fitting one window length (the Lasp over-dispersion fix).
    mh_val = {}
    if args.multi_horizon and args.val_manifest:
        from protein_jepa_frame.config import DataConfig as _DC
        from protein_jepa_frame.data.collate import collate_paired_windows as _cpw
        from protein_jepa_frame.data.dataset import build_dataset as _bd
        from protein_jepa_frame.data.protocol import apply_protocol as _ap, protocol_from_args as _pfa
        from protein_jepa_frame.data.collate import collate_clips as _cl
        from protein_jepa_frame.data.protocol import split_contiguous_window as _scw
        for h in mh_horizons:
            if args.use_contiguous_window or args.traj_loss:
                vcfg = _DC(manifest_path=args.val_manifest, batch_size=args.batch_size,
                           num_workers=0, training_mode="clip",
                           num_frames=args.history_frames + h, frame_stride=args.frame_stride,
                           max_residues=args.max_residues)
                vds = _bd(vcfg)
                if not len(vds):
                    continue
                n_v = min(len(vds), args.val_batches * args.batch_size)
                vi = torch.linspace(0, len(vds) - 1, n_v).long().tolist()
                vcf = lambda s, hh=h: _scw(_cl(s), args.history_frames, hh, step_ps=args.step_ps)  # noqa: E731
            else:
                class _PV: pass
                pv = _PV(); pv.history_frames = args.history_frames
                pv.frame_stride = args.frame_stride; pv.paired_horizons_frames = [h]
                vcfg = _DC(manifest_path=args.val_manifest, batch_size=args.batch_size,
                           num_workers=0, training_mode="paired_window",
                           num_frames=args.history_frames, paired_all_horizons=False,
                           max_residues=args.max_residues)
                _ap(vcfg, _pfa(pv))
                vds = _bd(vcfg)
                if not len(vds):
                    continue
                n_v = min(len(vds), args.val_batches * args.batch_size)
                vi = torch.linspace(0, len(vds) - 1, n_v).long().tolist()
                vcf = lambda s: _truncate_history(_cpw(s), args.history_frames)  # noqa: E731
            mh_val[h] = DataLoader(Subset(vds, vi), batch_size=args.batch_size,
                                   num_workers=0, shuffle=False, collate_fn=vcf)

    # Plan A: optionally fine-tune the (otherwise frozen) Stage-A encoder and/or
    if args.train_encoder or args.train_decoder:
        groups = [
            {"params": list(kernel.parameters()), "lr": args.lr, "name": "kernel"},
        ]
        if args.train_encoder:
            groups.append(
                {"params": [p for p in producer.encoder.parameters() if p.requires_grad],
                 "lr": args.lr * args.encoder_lr_scale, "name": "encoder"})
        if args.train_decoder:
            groups.append(
                {"params": [p for p in producer.decoder.parameters() if p.requires_grad],
                 "lr": args.lr * args.decoder_lr_scale, "name": "decoder"})
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
    if args.resume and args.init_from:
        raise ValueError("use either --resume or --init-from")
    if args.resume or args.init_from:
        rck = torch.load(args.resume or args.init_from, map_location=device, weights_only=False)
        kernel.load_state_dict(rck["kernel"])
        if rck.get("encoder") is not None:
            producer.encoder.load_state_dict(rck["encoder"])
        if rck.get("decoder") is not None:
            producer.decoder.load_state_dict(rck["decoder"])
        if args.resume:
            opt.load_state_dict(rck["optimizer"])
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
            "train_decoder": bool(getattr(args, "train_decoder", False)),
            "decoder": (producer.decoder.state_dict()
                        if getattr(args, "train_decoder", False) else None),
        }, out_dir / f"checkpoint_{tag}.pt")

    t0 = time.time()
    step = start_step
    n_skipped = 0
    gen = torch.Generator(device=device).manual_seed(args.seed)

    if args.multi_horizon:
        # Sample a horizon per step (round-robin) and train the terminal ensemble
        # loss at that horizon; batches are horizon-homogeneous by construction.
        mh_iters = {h: iter(dl) for h, dl in mh_loaders.items()}
        mh_h_idx = 0
        mh_cycle = list(mh_loaders.keys())
        while step < args.steps:
            h = mh_cycle[mh_h_idx % len(mh_cycle)]; mh_h_idx += 1
            dl = mh_loaders[h]
            try:
                batch = next(mh_iters[h])
            except StopIteration:
                mh_iters[h] = iter(dl); batch = next(mh_iters[h])
            paired = _move_batch(batch, device)
            args.rollout_steps = h
            if args.multi_horizon_max_grad is not None and h > args.multi_horizon_max_grad:
                args.rollout_grad_steps = args.multi_horizon_max_grad
            else:
                args.rollout_grad_steps = None
            producer.train()
            with sdpa_kernel([SDPBackend.MATH]):
                mh_run = _run_batch_traj if args.traj_loss else _run_batch
                comp, _, out = mh_run(producer, paired, args, weights, device, generator=gen)
            lr_scale = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            ok, gn = _safe_step(opt, clip_params, comp.total, args.grad_clip,
                                lr_groups=lr_scale)
            if not ok:
                n_skipped += 1
                print(f"step {step:6d} | H={h:3d} | SKIPPED non-finite loss/grad "
                      f"(grad_norm={gn}); {n_skipped} skipped so far", flush=True)
                continue
            step += 1
            if step % args.log_every == 0:
                d = comp.as_loggable_dict()
                extra = "".join(
                    f" {name}={d[key]:.4f}" for name, key in
                    (("rt", "ens/roundtrip"), ("pos", "ens/pos")) if key in d)
                print(f"step {step:6d} | H={h:3d} | lr {opt.param_groups[0]['lr']:.2e} | "
                      f"{time.time()-t0:6.1f}s | ens/total={d['ens/total']:.4f} "
                      f"rmsf={d['ens/rmsf']:.4f} contact={d['ens/contact']:.4f} "
                      f"dist={d['ens/dist']:.4f}{extra}", flush=True)
            if mh_val and step % args.val_every == 0:
                vagg = defaultdict(float); vals = {}
                for hh, vdl in mh_val.items():
                    args.rollout_steps = hh
                    args.rollout_grad_steps = None
                    v = _validate(producer, vdl, args, weights, device, args.val_batches)
                    vals[f"val/H{hh}"] = round(v.get("val/ens/total", float("nan")), 4)
                    vagg["val/ens/total"] += v.get("val/ens/total", 0.0) / len(mh_val)
                    vagg["val/selection"] += v["val/selection"] / len(mh_val)
                print(f"  [val {step}] mean={vagg['val/ens/total']:.4f} {json.dumps(vals)}",
                      flush=True)
                if vagg.get("val/selection", vagg["val/ens/total"]) < best_val:
                    best_val = vagg.get("val/selection", vagg["val/ens/total"])
                    save("best", step, best_val)
            if step % args.save_every == 0:
                save("latest", step, best_val)
            if step >= args.steps:
                break
        save("latest", step, best_val)
        print(f"done: {step} steps, best mean val ens/total = {best_val:.4f}", flush=True)
        return

    while step < args.steps:
        for batch in loader:
            paired = _move_batch(batch, device)
            producer.train()
            # The energy module's attention backward is only implemented for the
            # MATH SDP backend (efficient/flash attention backward is missing).
            with sdpa_kernel([SDPBackend.MATH]):
                if args.dense_trajectory:
                    run = _run_batch_dense
                elif args.traj_loss:
                    run = _run_batch_traj
                else:
                    run = _run_batch
                comp, _, out = run(producer, paired, args, weights, device, generator=gen)
            lr_scale = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            ok, gn = _safe_step(opt, clip_params, comp.total, args.grad_clip,
                                lr_groups=lr_scale)
            if not ok:
                n_skipped += 1
                print(f"step {step:6d} | SKIPPED non-finite loss/grad "
                      f"(grad_norm={gn}); {n_skipped} skipped so far", flush=True)
                continue
            step += 1
            if step % args.log_every == 0:
                d = comp.as_loggable_dict()
                if args.dense_trajectory:
                    print(f"step {step:6d} | lr {opt.param_groups[0]['lr']:.2e} | "
                          f"{time.time()-t0:6.1f}s | total={d['ens/total']:.4f} "
                          f"ca={d['ens/ca']:.4f} es={d['ens/es']:.4f} "
                          f"persist={d['ens/persist']:.4f}", flush=True)
                elif args.traj_loss:
                    print(f"step {step:6d} | lr {opt.param_groups[0]['lr']:.2e} | "
                          f"{time.time()-t0:6.1f}s | tot={d['ens/total']:.4f} "
                          f"rmsf={d['ens/rmsf']:.4f} dist={d['ens/dist']:.4f} "
                          f"t_es={d['traj/es']:.4f} t_rc={d['traj/rmsf_corr']:.4f} "
                          f"t_anc={d['traj/anchor']:.4f}", flush=True)
                else:
                    extra = "".join(
                        f" {name}={d[key]:.4f}" for name, key in
                        (("rt", "ens/roundtrip"), ("pos", "ens/pos")) if key in d)
                    print(f"step {step:6d} | lr {opt.param_groups[0]['lr']:.2e} | "
                          f"{time.time()-t0:6.1f}s | ens/total={d['ens/total']:.4f} "
                          f"rmsf={d['ens/rmsf']:.4f} contact={d['ens/contact']:.4f} "
                          f"dist={d['ens/dist']:.4f}{extra}", flush=True)
            if val_loader is not None and step % args.val_every == 0:
                v = _validate(producer, val_loader, args, weights, device, args.val_batches)
                print(f"  [val {step}] {json.dumps({k: round(x,4) for k,x in v.items()})}",
                      flush=True)
                sel = v.get("val/selection", v.get("val/ens/total", float("inf")))
                if sel < best_val:
                    best_val = sel
                    save("best", step, best_val)
            if step % args.save_every == 0:
                save("latest", step, best_val)
            if step >= args.steps:
                break
    save("latest", step, best_val)
    print(f"done: {step} steps, best val ens/total = {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()
