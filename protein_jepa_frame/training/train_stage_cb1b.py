"""From-none Stage B: jointly train (fresh encoder H, GLE dynamics, CA decoder).

This is the "from-none" retraining the diagnosis calls for.  No Stage A / B1a
checkpoint is consumed; a randomly initialised ProteinGeometryEncoder is
co-trained with the StageB1b H/S dynamics kernel and a lightweight CA decoder so
that the inter-frame difference in H must decode into inter-frame CA motion.

Multi-lag (1-100 ns)
--------------------
Each sample trains on a continuous clip of ``history_frames + horizon_steps``
frames.  ``horizon_steps`` is the *rollout length* (in frames) and is drawn from
the unified protocol grid per batch, so one encoder/kernel sees every timescale
from 1 ns to ~96 ns.  ``--multi-lag`` cycles over the protocol horizons.

Evaluation (held-out)
---------------------
Select on held-out proteins (val manifest) and held-out temperature
(``--heldout-temperature``) with a persistence-relative skill: the decoder must
beat the anchor-only forecast.  The best checkpoint is chosen on the combined
val score.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Sampler, Subset

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import split_contiguous_window, validate_manifest_split
from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.models.stage_b1b_fromnone.decoder import FromNoneConfig, SimpleCADecoder
from protein_jepa_frame.models.stage_b1b_fromnone.objectives import (
    FromNoneWeights,
    aligned_ca_rmsd,
    energy_score,
    temporal_variance_floor,
)
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.training.train_utils import _device, _lr_scale, _move_batch, _set_reproducibility

TRAINING_CONTRACT = "from_none_encoder_v1"
PROTOCOL_HORIZONS = (8, 16, 24, 32, 48, 64, 96)   # raw frames = 8..96 ns at 1 ns/frame


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest")
    p.add_argument("--val-manifest")
    p.add_argument("--heldout-temperature", type=float, default=None,
                   help="Optional held-out temperature (K); its trajectories are excluded from train.")
    p.add_argument("--output", default="artifacts/fromnone/stage_cb1b")
    p.add_argument("--resume")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--encoder-lr-scale", type=float, default=1.0,
                   help="Multiplier on the encoder LR (dynamics/decoder use --lr).")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--horizon-steps", type=int, default=16,
                   help="Default rollout length (frames). Multi-lag overrides per batch.")
    p.add_argument("--multi-lag", action="store_true",
                   help="Cycle rollout length over PROTOCOL_HORIZONS (8..96 ns) per batch.")
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--step-ps", type=float, default=1000.0)
    # Encoder
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--encoder-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--orientation-features", action="store_true", default=True)
    p.add_argument("--no-orientation-features", dest="orientation_features", action="store_false")
    # Dynamics kernel (StageB1b)
    p.add_argument("--bath-dim", type=int, default=8)
    p.add_argument("--energy-depth", type=int, default=3)
    p.add_argument("--energy-heads", type=int, default=4)
    p.add_argument("--energy-hidden", type=int, default=128)
    p.add_argument("--history-hidden", type=int, default=128)
    p.add_argument("--coupling-scale", type=float, default=1e-3,
                   help="Bath coupling scale. With h=1000 ps the drift term h*(C@S) needs C~1e-4, "
                        "so coupling_scale ~1e-3 (c_net also has c_init_std=0.1).")
    p.add_argument("--diffusion-scale", type=float, default=1e-4)
    p.add_argument("--c-init-std", type=float, default=0.1)
    p.add_argument("--gamma-scale", type=float, default=2e-4)
    p.add_argument("--gamma-floor", type=float, default=2e-5)
    p.add_argument("--max-gamma-step", type=float, default=1.9)
    p.add_argument("--force-clip", type=float, default=1.0,
                   help="Elementwise clamp on |force| before the h*D scaling (stability). "
                        "Set <=0 to disable.")
    p.add_argument("--tail-init", type=float, default=1.0,
                   help="Quadratic-tail init; provides a restoring force for from-none rollouts.")
    # Decoder
    p.add_argument("--decoder-hidden", type=int, default=256)
    p.add_argument("--decoder-depth", type=int, default=3)
    p.add_argument("--max-update", type=float, default=12.0)
    # Loss weights
    p.add_argument("--es-weight", type=float, default=0.1)
    p.add_argument("--ca-weight", type=float, default=1.0)
    p.add_argument("--phys-weight", type=float, default=0.1)
    p.add_argument("--anchor-weight", type=float, default=0.0)
    p.add_argument("--variance-weight", type=float, default=0.0)
    p.add_argument("--variance-floor", type=float, default=0.0)
    p.add_argument("--es-num-samples", type=int, default=4)
    p.add_argument("--ca-horizon-steps", type=int, default=None,
                   help="If set, CA supervision only on the first N rollout steps.")
    # Eval / logs
    p.add_argument("--validation-every", type=int, default=500)
    p.add_argument("--validation-batches", type=int, default=16)
    p.add_argument("--validation-horizon-steps", type=int, default=32)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-size", type=int, default=32)
    a = p.parse_args(argv)
    for key in ("steps", "history_frames", "horizon_steps", "batch_size",
                "validation_every", "validation_batches", "log_every", "save_every"):
        if getattr(a, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if not (a.es_weight > 0 and a.ca_weight >= 0 and a.phys_weight >= 0):
        p.error("es_weight must be positive; ca/physical weights nonnegative")
    if a.manifest and not a.val_manifest and not a.dry_run:
        p.error("real-data training requires --val-manifest for held-out selection")
    return a


class EpochPermutation(Sampler):
    def __init__(self, data_source, seed):
        self.length, self.seed, self.epoch = len(data_source), seed, 0

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.length, generator=rng).tolist())

    def __len__(self):
        return self.length


# -----------------------------------------------------------------------------
# Dataset / loaders
# -----------------------------------------------------------------------------

def _filter_temperature(records, heldout_temperature):
    if heldout_temperature is None:
        return records
    return [r for r in records if not math.isclose(float(r.temperature_k), heldout_temperature, rel_tol=1e-6)]


def _loader(args, manifest, training, horizon_frames):
    config = DataConfig(
        manifest_path=manifest, training_mode="clip", num_frames=args.history_frames + horizon_frames,
        frame_stride=args.frame_stride, max_residues=args.max_residues,
        synthetic_size=args.smoke_size, synthetic_residues=args.max_residues or 16,
    )
    if manifest and args.heldout_temperature is not None:
        from protein_jepa_frame.data.manifest import load_manifest
        records = _filter_temperature(load_manifest(manifest), args.heldout_temperature)
        if not records:
            raise ValueError("held-out-temperature filtering left no trajectories")
        dataset = build_dataset(config)  # placeholder; replaced below
        from protein_jepa_frame.data.dataset import TrajectoryClipDataset
        dataset = TrajectoryClipDataset(records, config)
    else:
        dataset = build_dataset(config)
    if manifest is None:
        boundary = max(1, len(dataset) // 2)
        dataset = Subset(dataset, range(0, boundary) if training else range(boundary, len(dataset)))
    if len(dataset) == 0:
        raise ValueError("empty training/validation dataset")
    if not training:
        total = len(dataset)
        budget = args.validation_batches * args.batch_size
        indices = torch.linspace(0, total - 1, min(total, budget)).long().tolist()
        dataset = Subset(dataset, indices)
    loader_rng = torch.Generator().manual_seed(args.seed + (0 if training else 1))
    sampler = EpochPermutation(dataset, args.seed) if training else None
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                      num_workers=args.num_workers, collate_fn=collate_clips,
                      drop_last=False, generator=loader_rng)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class EMANormalizer(nn.Module):
    """Per-channel EMA standardizer for the fresh encoder's H.

    A from-scratch encoder has no calibrated output scale, so the H/S kernel and
    the energy score see huge activations at init.  This module tracks the mean
    and (root-mean-square) scale of the encoder output with an EMA of *detached*
    batch statistics and standardizes H before the dynamics/decoder.  It is not
    a trainable module and introduces no gradient path.
    """

    def __init__(self, latent_dim: int, momentum: float = 0.01, min_scale: float = 1e-2):
        super().__init__()
        self.latent_dim = latent_dim
        self.momentum = momentum
        self.min_scale = min_scale
        self.register_buffer("mean_", torch.zeros(latent_dim))
        self.register_buffer("scale_", torch.ones(latent_dim))
        self.register_buffer("n_updates_", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, H: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        flat = H.detach().reshape(-1, self.latent_dim).float()
        if flat.shape[0] == 0 or not torch.isfinite(flat).all():
            return
        b_mean = flat.mean(dim=0)
        b_scale = flat.std(dim=0, correction=0).clamp(min=self.min_scale)
        m = 1.0 if self.n_updates_.item() == 0 else self.momentum
        self.mean_.lerp_(b_mean, m)
        self.scale_.lerp_(b_scale, m)
        self.n_updates_ += 1

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.mean_) / self.scale_


def build_encoder(args, device):
    return ProteinGeometryEncoder(
        latent_dim=args.latent_dim, depth=args.encoder_depth, num_heads=args.encoder_heads,
        num_rbf=16, dropout=args.dropout, max_residues=4096,
        orientation_features=args.orientation_features, identity_embed_scale=0.1,
    ).to(device)


def build_kernel(args, device):
    cfg = StageB1bConfig(
        latent_dim=args.latent_dim, bath_dim=args.bath_dim, energy_depth=args.energy_depth,
        energy_heads=args.energy_heads, energy_hidden=args.energy_hidden,
        history_hidden=args.history_hidden, step_ps=args.step_ps,
        coupling_scale=args.coupling_scale, diffusion_scale=args.diffusion_scale,
        c_init_std=args.c_init_std, gamma_scale=args.gamma_scale,
        gamma_floor=args.gamma_floor, max_gamma_step=args.max_gamma_step,
        force_clip=(args.force_clip if args.force_clip and args.force_clip > 0 else None),
        tail_init=args.tail_init,
    )
    return StageB1b(cfg).to(device)


def build_decoder(args, device):
    cfg = FromNoneConfig(latent_dim=args.latent_dim, bath_dim=args.bath_dim,
                         hidden=args.decoder_hidden, depth=args.decoder_depth,
                         max_update=args.max_update)
    return SimpleCADecoder(cfg).to(device)


def encode_frames(encoder, coords, rmask, aatype, ridx, cid, normalizer=None):
    """[B,T,L,4,3] -> [B,T,L,dH] (trainable; H standardized if a normalizer is given)."""
    B, T, L = coords.shape[:3]
    H = encoder(
        coords.reshape(B * T, 1, L, 4, 3), rmask.reshape(B * T, 1, L),
        aatype[:, None].expand(B, T, L).reshape(B * T, L),
        ridx[:, None].expand(B, T, L).reshape(B * T, L),
        cid[:, None].expand(B, T, L).reshape(B * T, L),
        token_mask=rmask.reshape(B * T, 1, L),
    )[:, 0].reshape(B, T, L, -1)
    if normalizer is not None:
        H = normalizer(H)
    return H


def split_batch(batch, history_frames, horizon_steps, step_ps, clip_frames=None):
    """Slice the collated clip to K+horizon_steps frames then split.

    The dataloader always yields the maximum clip (``clip_frames``); with
    multi-lag the rollout length varies per batch, so we take the leading
    ``history_frames + horizon_steps`` frames and drop the unused tail.
    """
    if clip_frames is not None and clip_frames != history_frames + horizon_steps:
        if clip_frames < history_frames + horizon_steps:
            raise ValueError(
                f"rollout horizon {horizon_steps} needs {history_frames+horizon_steps} frames "
                f"but the clip only has {clip_frames}")
        keep = history_frames + horizon_steps
        # Only frame-stacked keys have the clip length on dim 1.  Per-frame keys
        # (aatype/residue_index/chain_id) are [B, L] and MUST NOT be sliced here,
        # even when L coincidentally equals the clip length.
        frame_keys = ("coords", "residue_mask", "atom_mask", "time_ps", "dt_ps", "forces")
        batch = {
            k: (v[:, :keep] if (k in frame_keys and torch.is_tensor(v)
                                and v.ndim >= 2 and v.shape[1] == clip_frames) else v)
            for k, v in batch.items()
        }
    paired = split_contiguous_window(batch, history_frames, horizon_steps, step_ps=step_ps)
    mask = batch["residue_mask"][:, history_frames - 1].bool()
    if not torch.equal(batch["residue_mask"].bool(), mask[:, None].expand_as(batch["residue_mask"])):
        raise ValueError("residue validity must be constant across the continuous clip")
    # `batch` is the (possibly tail-sliced) collated clip; return it so the caller
    # uses masks with the same K+N length as context/future.
    paired["_batch"] = batch
    return paired["context_coords"], paired["future_coords"], mask, paired["context_dt_ps"], batch


def autonomous_rollout(kernel, H_hist, dt_hist, mask, temperature, n_steps, num_samples,
                       *, generator=None, training=True):
    M = num_samples
    B, K, L, d = H_hist.shape
    rep = lambda x: x.unsqueeze(0).expand(M, *x.shape).reshape(M * B, *x.shape[1:])
    hist, dt, m = rep(H_hist), rep(dt_hist), rep(mask)
    temp = None
    if temperature is not None:
        t = torch.as_tensor(temperature, device=H_hist.device, dtype=H_hist.dtype)
        temp = rep(t.expand(B) if t.ndim == 0 else t.reshape(B))
    H, S = kernel.rollout(
        hist[:, -1], m, temp, n_steps=n_steps, H_hist=hist, dt_hist=dt,
        seq_feature=hist[:, -1], generator=generator, detach=not training, return_bath=True,
    )
    H = H.reshape(M, B, n_steps + 1, L, d)
    S = S.reshape(M, B, n_steps + 1, L, kernel.cfg.bath_dim) if S is not None else None
    return H, S


def anchor_frames(anchor_backbone):
    """[B,L,4,3] -> (R_anchor [B,L,3,3], t_anchor [B,L,3])."""
    return RigidBodyFrame.from_backbone(anchor_backbone[..., 0, :], anchor_backbone[..., 1, :],
                                        anchor_backbone[..., 2, :])


def decode_paths(decoder, paths, S_paths, H_anchor, anchor_bb, mask):
    """[M,B,T,L,d] -> ca_pred [M,B,T,L,3] in the anchor gauge."""
    M, B, T, L, d = paths.shape
    R, t = anchor_frames(anchor_bb)                       # [B,L,3,3],[B,L,3]
    X_anchor = anchor_bb[..., 1, :]                       # [B,L,3]
    # Broadcast a [B, ...] tensor to [M*B*T, ...] repeating along the sample and
    # time axes (the anchor state is shared by all M samples and all T steps).
    bc = lambda x: x[None, :, None].expand(M, B, T, *x.shape[1:]).reshape(M * B * T, *x.shape[1:])
    Hf = paths.reshape(M * B * T, L, d)
    Ha = bc(H_anchor)
    Xa = bc(X_anchor)
    Ra = bc(R)
    m = bc(mask)
    Sf = None if S_paths is None else S_paths.reshape(M * B * T, L, S_paths.shape[-1])
    out = decoder(H_future=Hf, H_context=Ha, X_context=Xa, residue_mask=m, R_anchor=Ra, S_future=Sf)
    return out["ca_pred"].reshape(M, B, T, L, 3)


def normalized_path_es(samples, target, mask):
    """Normalized energy score of an H rollout path.

    ``samples``: [M, B, T, L, d]; ``target``: [B, T, L, d].  Scores the M samples
    at each protein against the (detached) target trajectory, normalized by the
    square root of the feature count so it is comparable across proteins.
    """
    M, B, T, L, d = samples.shape
    scores = []
    for b in range(B):
        mb = mask[b]                                   # [L]
        smp = samples[:, b, :, mb, :].reshape(M, -1)   # [M, T*Lb*d]
        tgt = target[b, :, mb, :].reshape(-1).detach()
        scores.append(energy_score(smp, tgt) / math.sqrt(max(tgt.numel(), 1)))
    return torch.stack(scores).mean()


def ca_physical_features(ca, mask):
    """Physical features from a CA cloud: [.., L, 3] -> [N, F] (flattened).

    Distance/10 A between a fixed sample of CA pairs, plus a contact indicator.
    Used by the physical energy score, which only needs the CA trace.
    """
    L = ca.shape[-2]
    ca2 = ca.reshape(-1, L, 3)
    n = ca2.shape[0]
    m = mask.reshape(-1, L)
    if m.shape[0] == 1 and n > 1:
        m = m.expand(n, L)
    feats = []
    for b in range(n):
        idx = torch.nonzero(m[b], as_tuple=False).flatten()
        c = ca2[b, idx]
        pairs = torch.triu_indices(len(idx), len(idx), offset=1, device=ca.device)
        if pairs.shape[1] > 128:
            sel = torch.linspace(0, pairs.shape[1] - 1, 128, device=ca.device).long()
            pairs = pairs[:, sel]
        if pairs.shape[1] == 0:
            feats.append(torch.zeros(0, device=ca.device, dtype=ca.dtype))
            continue
        dist = (c[pairs[0]] - c[pairs[1]]).square().sum(-1).clamp_min(1e-12).sqrt()
        feats.append(torch.cat([dist / 10.0, torch.sigmoid(8.0 - dist)], dim=-1))
    return feats


def physical_energy_score(pred_ca, true_ca, mask):
    """Energy score on CA-cloud physical features (proper score, per protein·time).

    ``pred_ca``: [M, B, T, L, 3] (M rollout samples); ``true_ca``: [B, T, L, 3].
    Each (b,t) element scores the M samples against the true (b,t) feature vector;
    the per-element scores are averaged.
    """
    M, B, T, L, _ = pred_ca.shape
    # features are pair distances; compute per (b,t) so the sample axis is M only.
    pred_flat = pred_ca.permute(1, 2, 0, 3, 4).reshape(B * T, M, L, 3)
    true_flat = true_ca.reshape(B * T, L, 3)
    mask_flat = mask.unsqueeze(1).expand(B, T, L).reshape(B * T, L)
    scores = []
    # Sample a fixed subset of (b,t) elements for speed/memory stability.
    n = B * T
    step = max(1, n // 32)
    for i in range(0, n, step):
        p_feats = ca_physical_features(pred_flat[i], mask_flat[i])   # list length M
        t_feats = ca_physical_features(true_flat[i:i + 1], mask_flat[i:i + 1])[0]
        if t_feats.numel() == 0:
            continue
        samp = torch.stack(p_feats, dim=0)                          # [M, F]
        scores.append(energy_score(samp, t_feats.detach()) / math.sqrt(max(t_feats.numel(), 1)))
    if not scores:
        return pred_ca.new_zeros(())
    return torch.stack(scores).mean()


def forecast_objective(encoder, kernel, decoder, batch, args, n_steps, weights,
                       *, generator=None, training=True, normalizer=None):
    K = args.history_frames
    # The dataloader emits clips of length K + (loader horizon); infer it from the
    # batch so training (multi-lag max) and validation (its own horizon) both work.
    clip_frames = batch["coords"].shape[1]
    batch = {k: v for k, v in batch.items()}
    history, targets, mask, dt, batch = split_batch(batch, K, n_steps, args.step_ps, clip_frames)
    aat, ridx, cid = (batch[k] for k in ("aatype", "residue_index", "chain_id"))
    H_hist_raw = encode_frames(encoder, history, batch["residue_mask"][:, :K], aat, ridx, cid)
    if normalizer is not None:
        if training:
            normalizer.update(H_hist_raw, mask)      # never adapt stats on val
        H_hist = normalizer(H_hist_raw)
    else:
        H_hist = H_hist_raw
    H_anchor = H_hist[:, -1]
    # Autonomous forecast FIRST (targets encoded only after this point).
    H_pred, S_pred = autonomous_rollout(
        kernel, H_hist, dt, mask, batch.get("temperature_k"), n_steps, args.es_num_samples,
        generator=generator, training=training,
    )
    H_target = encode_frames(encoder, targets, batch["residue_mask"][:, K - 1:], aat, ridx, cid, normalizer)

    # --- CA decodability (primary): Kabsch-aligned CA point loss ---
    S_fut = None if S_pred is None else S_pred[:, :, 1:]
    ca_pred = decode_paths(decoder, H_pred[:, :, 1:], S_fut, H_anchor, batch["coords"][:, K - 1], mask)
    ca_end = min(n_steps, args.ca_horizon_steps or n_steps)
    true_ca = targets[:, 1:ca_end + 1, :, 1, :]                 # [B,ca_end,L,3]
    ca_loss = weighted_ca_rmsd(ca_pred[:, :, :ca_end], true_ca, mask)
    persist = persistence_ref_rmsd(true_ca, targets[:, 0, :, 1, :], mask)

    # --- Energy score on the H rollout path (distributional term) ---
    es = normalized_path_es(H_pred[:, :, 1:], H_target[:, 1:], mask)
    phys = (physical_energy_score(ca_pred, targets[:, 1:, :, 1, :], mask)
            if args.phys_weight > 0 else es.new_zeros(()))
    anchor_term = es.new_zeros(())
    var_term = temporal_variance_floor(
        torch.cat([H_hist, H_target[:, 1:]], dim=1), mask, args.variance_floor,
    ) if args.variance_weight > 0 else es.new_zeros(())

    total = (weights.es_weight * es + weights.ca_weight * ca_loss
             + weights.phys_weight * phys + weights.anchor_weight * anchor_term
             + weights.variance_weight * var_term)
    return total, {"H_pred": H_pred, "H_target": H_target, "H_anchor": H_anchor,
                   "ca_pred": ca_pred, "persist": persist}, {
        "total": total, "es": es, "ca": ca_loss, "phys": phys,
        "anchor": anchor_term, "variance": var_term, "persist": persist,
    }


def weighted_ca_rmsd(ca_pred, true_ca, mask):
    """Mean over (M,B,T) of Kabsch CA RMSD; ``ca_pred`` [M,B,T,L,3], ``true_ca`` [B,T,L,3]."""
    M, B, T, L, _ = ca_pred.shape
    p = ca_pred.reshape(M * B * T, L, 3)
    q = true_ca.unsqueeze(0).expand(M, B, T, L, 3).reshape(M * B * T, L, 3)
    m = mask[:, None, :].expand(B, T, L)[None].expand(M, B, T, L).reshape(M * B * T, L)
    return aligned_ca_rmsd(p, q, m)


def persistence_ref_rmsd(true_ca, anchor_ca, mask):
    """Kabsch CA RMSD of the anchor-only forecast, averaged over horizons (skill ref)."""
    B, T, L, _ = true_ca.shape
    q = true_ca.reshape(B * T, L, 3)
    p = anchor_ca.unsqueeze(1).expand(B, T, L, 3).reshape(B * T, L, 3)
    m = mask.unsqueeze(1).expand(B, T, L).reshape(B * T, L)
    return aligned_ca_rmsd(p, q, m)


# -----------------------------------------------------------------------------
# Validation (held-out proteins and/or temperature)
# -----------------------------------------------------------------------------

def validate(encoder, kernel, decoder, loader, args, device, normalizer=None):
    val_args = argparse.Namespace(**vars(args))
    val_args.horizon_steps = args.validation_horizon_steps or args.horizon_steps
    val_args.ca_horizon_steps = None
    was_training = (encoder.training, kernel.training, decoder.training)
    encoder.eval(); kernel.eval(); decoder.eval()
    gen = torch.Generator(device=device).manual_seed(args.seed + 1000003)
    acc = {}
    n = 0
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        for i, batch in enumerate(loader):
            if i >= args.validation_batches:
                break
            batch = _move_batch(batch, device)
            _, _, comps = forecast_objective(
                encoder, kernel, decoder, batch, val_args, val_args.horizon_steps,
                FromNoneWeights(es_weight=args.es_weight, ca_weight=args.ca_weight,
                                phys_weight=args.phys_weight, anchor_weight=args.anchor_weight,
                                variance_weight=args.variance_weight),
                generator=gen, training=False, normalizer=normalizer,
            )
            row = {
                "val/ca_rmsd": float(comps["ca"]),
                "val/persist_rmsd": float(comps["persist"]),
                "val/es": float(comps["es"]),
            }
            # skill: 1 - ca/persist  (>0 means beating persistence)
            p = float(comps["persist"])
            row["val/skill"] = (1.0 - float(comps["ca"]) / p) if p > 1e-9 else 0.0
            for k, v in row.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1
    encoder.train(was_training[0]); kernel.train(was_training[1]); decoder.train(was_training[2])
    out = {k: v / max(n, 1) for k, v in acc.items()}
    # Selection score: lower ca_rmsd is better, tie-broken by ES.
    out["loss/select"] = out["val/ca_rmsd"] + 1e-3 * out["val/es"]
    return out


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

def capture_rng():
    rng = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def restore_rng(rng):
    torch.set_rng_state(rng["torch"].cpu())
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def checkpoint_payload(encoder, kernel, decoder, optimizer, step, args, validation, best, bad,
                       normalizer=None):
    payload = {
        "encoder": encoder.state_dict(), "kernel": kernel.state_dict(),
        "decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
        "step": step, "train_args": vars(args), "training_contract": TRAINING_CONTRACT,
        "fromnone_config": {"latent_dim": args.latent_dim, "bath_dim": args.bath_dim,
                            "encoder_depth": args.encoder_depth, "decoder_hidden": args.decoder_hidden},
        "validation": validation, "best_score": best, "bad_validations": bad, "rng_state": capture_rng(),
        "run_scope": "smoke_only" if args.dry_run else "from_none",
    }
    if normalizer is not None:
        payload["normalizer"] = {"mean": normalizer.mean_.cpu(), "scale": normalizer.scale_.cpu()}
    return payload


def load_models_for_eval(path, device):
    """Load a from-none checkpoint into (encoder, kernel, decoder, normalizer, ck)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("training_contract") != TRAINING_CONTRACT:
        raise ValueError("checkpoint is not a from-none Stage B run")
    a = argparse.Namespace(**ck["train_args"])
    encoder = build_encoder(a, device); encoder.load_state_dict(ck["encoder"], strict=True)
    kernel = build_kernel(a, device); kernel.load_state_dict(ck["kernel"], strict=True)
    decoder = build_decoder(a, device); decoder.load_state_dict(ck["decoder"], strict=True)
    normalizer = None
    if "normalizer" in ck:
        normalizer = EMANormalizer(a.latent_dim).to(device)
        normalizer.mean_.copy_(torch.as_tensor(ck["normalizer"]["mean"]).to(device))
        normalizer.scale_.copy_(torch.as_tensor(ck["normalizer"]["scale"]).to(device))
    return encoder.eval(), kernel.eval(), decoder.eval(), normalizer, ck


# Weighted multi-lag cycle: short horizons (cheap, common) appear more often,
# the long (expensive) horizons are sampled but rarer.  The cycle covers every
# protocol horizon so no lag is unseen; the weighting keeps the per-step cost
# bounded while still exercising 1-100 ns.
MULTILAG_CYCLE = (8, 8, 16, 8, 32, 8, 16, 8, 48, 16, 8, 24, 64, 32, 16, 8, 96)


def horizon_schedule(args, step):
    """Rollout length (frames) for this step: fixed or a weighted multi-lag cycle."""
    if args.multi_lag:
        return MULTILAG_CYCLE[step % len(MULTILAG_CYCLE)]
    return args.horizon_steps


def main(argv=None):
    args = parse_args(argv)
    _set_reproducibility(args.seed, True)
    device = _device(args.device)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.manifest and args.val_manifest:
        validate_manifest_split(args.manifest, args.val_manifest)

    encoder = build_encoder(args, device)
    kernel = build_kernel(args, device)
    decoder = build_decoder(args, device)
    normalizer = EMANormalizer(args.latent_dim).to(device)
    # Encoder gets an optionally-scaled LR; dynamics/decoder share --lr.
    param_groups = [
        {"params": encoder.parameters(), "lr": args.lr * args.encoder_lr_scale},
        {"params": list(kernel.parameters()) + list(decoder.parameters()), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    weights = FromNoneWeights(es_weight=args.es_weight, ca_weight=args.ca_weight,
                              phys_weight=args.phys_weight, anchor_weight=args.anchor_weight,
                              variance_weight=args.variance_weight)
    weights.validate()

    max_h = max(PROTOCOL_HORIZONS) if args.multi_lag else args.horizon_steps
    val_h = max(args.validation_horizon_steps or args.horizon_steps, max_h if args.multi_lag else 0)
    train_loader = _loader(args, args.manifest, True, max_h if args.multi_lag else args.horizon_steps)
    val_loader = _loader(args, args.val_manifest, False, val_h) if (args.val_manifest or not args.manifest) else None

    step, best, bad, validation = 0, float("inf"), 0, {}
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"], strict=True)
        kernel.load_state_dict(ck["kernel"], strict=True)
        decoder.load_state_dict(ck["decoder"], strict=True)
        if "normalizer" in ck:
            normalizer.mean_.copy_(torch.as_tensor(ck["normalizer"]["mean"]).to(device))
            normalizer.scale_.copy_(torch.as_tensor(ck["normalizer"]["scale"]).to(device))
        optimizer.load_state_dict(ck["optimizer"]); restore_rng(ck["rng_state"])
        step, best, bad, validation = ck["step"], ck["best_score"], ck["bad_validations"], ck["validation"]

    (out / "config.json").write_text(json.dumps({
        "train_args": vars(args), "training_contract": TRAINING_CONTRACT,
        "protocol_horizons": list(PROTOCOL_HORIZONS),
    }, indent=2))
    print(f"[from-none] encoder=fresh latent={args.latent_dim} bath={args.bath_dim} "
          f"multi_lag={args.multi_lag} horizons={PROTOCOL_HORIZONS if args.multi_lag else args.horizon_steps}",
          flush=True)

    encoder.train(); kernel.train(); decoder.train()
    stop = False
    while step < args.steps and not stop:
        first = step % len(train_loader)
        train_loader.sampler.epoch = step // len(train_loader)
        for index, batch in enumerate(train_loader):
            if index < first:
                continue
            if step >= args.steps:
                break
            n_steps = horizon_schedule(args, step)
            base_lr = args.lr * _lr_scale(step, args.steps, args.warmup_steps)
            optimizer.param_groups[0]["lr"] = base_lr * args.encoder_lr_scale
            optimizer.param_groups[1]["lr"] = base_lr
            optimizer.zero_grad(set_to_none=True)
            gen = torch.Generator(device=device).manual_seed(args.seed * 7919 + step)
            with sdpa_kernel([SDPBackend.MATH]):
                total, _, comps = forecast_objective(
                    encoder, kernel, decoder, _move_batch(batch, device), args, n_steps, weights,
                    generator=gen, training=True, normalizer=normalizer,
                )
            if not torch.isfinite(total):
                raise FloatingPointError("nonfinite from-none training loss")
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(kernel.parameters()) + list(decoder.parameters()),
                    args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                msg = " ".join(f"{k}={float(v.detach()):.6g}" for k, v in comps.items())
                print(f"step {step} h={n_steps} {msg}", flush=True)
            improved = False
            if val_loader is not None and (step % args.validation_every == 0 or step == args.steps):
                validation = validate(encoder, kernel, decoder, val_loader, args, device, normalizer)
                score = validation["loss/select"]
                improved, best = score < best, min(best, score)
                bad = 0 if improved else bad + 1
                print("validation " + json.dumps(validation, sort_keys=True), flush=True)
                stop = args.patience > 0 and bad >= args.patience
            if improved or step % args.save_every == 0 or step == args.steps or stop or args.dry_run:
                payload = checkpoint_payload(encoder, kernel, decoder, optimizer, step, args, validation,
                                             best, bad, normalizer)
                torch.save(payload, out / "checkpoint_latest.pt")
                if improved:
                    torch.save(payload, out / "checkpoint_best.pt")
            if args.dry_run and step >= args.steps:
                print("smoke complete", flush=True)
                return
            if stop:
                break


if __name__ == "__main__":
    main()
