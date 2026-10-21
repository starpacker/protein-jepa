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
from protein_jepa_frame.models.stage_b1b_fromnone.mp_decoder import (
    CRIDecoder, MessagePassingCADecoder, MessagePassingConfig,
)
from protein_jepa_frame.models.stage_b1b_fromnone.objectives import (
    FromNoneWeights,
    aligned_ca_rmsd,
    energy_score,
    ensemble_rmsf_loss,
    ensemble_rmsf_corr_loss,
    ensemble_spread_floor,
    ensemble_variance_loss,
    temporal_variance_floor,
)
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.training.train_utils import _device, _lr_scale, _move_batch, _set_reproducibility

TRAINING_CONTRACT = "from_none_encoder_v1"

# Protocol horizons are defined in PHYSICAL TIME (ns) and converted to frames
# using --step-ps.  Defining them in raw frames silently changes the physical
# span whenever the data resolution changes: at 1000 ps/frame "8..96 frames" is
# 8..96 ns, but at 100 ps/frame the same tuple is only 0.8..9.6 ns.
#
# Short-range horizons (0.1, 0.2, 0.5 ns) are included so that at 100 ps/frame
# the model is directly supervised at 1-, 2-, and 5-step rollouts.  Without them
# the model only ever sees ≥10-step targets and never learns the fine-grained
# single-step dynamics that motivates using 100 ps data in the first place.
# At 1000 ps/frame these collapse to frame-count 1, so they are also safe there.
PROTOCOL_HORIZONS_NS = (0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64)


def horizons_frames(step_ps: float, horizons_ns=PROTOCOL_HORIZONS_NS) -> tuple[int, ...]:
    """Convert physical horizons (ns) to rollout lengths in frames.

    Each horizon is rounded to the nearest whole frame and floored at 1, then
    de-duplicated (at coarse step_ps distinct ns values can collapse onto the
    same frame count).
    """
    frames = sorted({max(1, int(round(ns * 1000.0 / float(step_ps)))) for ns in horizons_ns})
    return tuple(frames)


def protocol_horizons(args) -> tuple[int, ...]:
    """Rollout lengths (frames) for this run, derived from --step-ps."""
    ns = getattr(args, "horizons_ns", None) or PROTOCOL_HORIZONS_NS
    return horizons_frames(args.step_ps, ns)


def cycle_frames(step_ps: float, cycle_ns) -> tuple[int, ...]:
    """Map a WEIGHTED cycle of physical timescales to frames, preserving order.

    ``horizons_frames`` de-duplicates, which is correct for the protocol grid but
    destroys a weighted cycle: repeated entries ARE the weights.  Using it here
    silently collapsed every cycle into an ascending sweep with uniform weight.
    """
    return tuple(max(1, int(round(ns * 1000.0 / float(step_ps)))) for ns in cycle_ns)


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
    p.add_argument("--rollout-grad-steps", type=int, default=None,
                   help="Truncated BPTT window: keep only the last N rollout steps differentiable. "
                        "The forward pass still runs the full horizon, so long-lag targets and the "
                        "energy score are exact; this only bounds activation memory (640 steps at "
                        "100 ps/frame would otherwise need tens of GB). None = full graph.")
    p.add_argument("--encode-frame-chunk", type=int, default=None,
                   help="Encode at most this many (B*T) frames per encoder call to bound the "
                        "encoder's O(B*T*L^2) attention memory. At 100 ps/frame a 64 ns horizon "
                        "encodes ~641 future frames; chunking keeps peak memory flat. None = one "
                        "call (fine at 1 ns/frame).")
    p.add_argument("--horizon-steps", type=int, default=16,
                   help="Default rollout length (frames). Multi-lag overrides per batch.")
    p.add_argument("--multi-lag", action="store_true",
                   help="Cycle rollout length over PROTOCOL_HORIZONS (8..96 ns) per batch.")
    p.add_argument("--max-residues", type=int, default=256)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--step-ps", type=float, default=1000.0,
                   help="Physical duration (ps) of ONE latent/dynamics step. MUST equal the "
                        "dataset frame interval: 1000 for mdCATH/ATLAS_1ns, 100 for ATLAS_100ps.")
    p.add_argument("--horizons-ns", type=float, nargs="+", default=None,
                   help="Protocol horizons in ns (default 1 2 4 8 16 32 64). Converted to "
                        "rollout frames with --step-ps.")
    # Encoder
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--encoder-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--orientation-features", action="store_true", default=True)
    p.add_argument("--no-orientation-features", dest="orientation_features", action="store_false")
    p.add_argument("--relative-position-features", action="store_true", default=False,
                   help="Enable the within-frame PCA canonical-position channel. Without it H "
                        "is invariant to per-frame rigid motion and cannot express how a residue "
                        "moved relative to the anchor (ridge probe: zero decodable dynamics).")
    p.add_argument("--anchor-conditioned", action="store_true", default=False,
                   help="Project each frame's CA into the ANCHOR frame's PCA basis and add it as "
                        "an input channel. Unlike the within-frame canonical positions (re-derived "
                        "from each frame's own cloud, so nearly identical for similar shapes), this "
                        "reference frame is fixed at the anchor, so the channel changes with the "
                        "motion between anchor and frame: H_future - H_anchor becomes "
                        "displacement-relevant.")
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
    p.add_argument("--decoder-type", choices=("mlp", "mp", "cri"), default="mlp",
                   help="CA decoder: 'mlp' = per-residue SimpleCADecoder (baseline); "
                        "'mp' = message-passing decoder (collective motion); "
                        "'cri' = cross-residue + H<->S interactive decoder.")
    p.add_argument("--mp-depth", type=int, default=2, help="message-passing / CRI layers")
    p.add_argument("--mp-heads", type=int, default=4, help="self-attention heads per layer")
    p.add_argument("--hs-heads", type=int, default=4, help="H<->S cross-attention heads (CRI only)")
    p.add_argument("--no-hs-cross-attention", dest="hs_cross_attention", action="store_false",
                   default=True, help="disable H<->S cross-attention in the CRI decoder")
    p.add_argument("--no-bath-coupling-edges", dest="bath_coupling_edges", action="store_false",
                   default=True, help="disable the S_i.S_j bath-coupling edge feature in CRI")
    p.add_argument("--decoder-dropout", type=float, default=0.0)
    p.add_argument("--max-update", type=float, default=12.0)
    # Loss weights
    p.add_argument("--es-weight", type=float, default=0.1)
    p.add_argument("--ca-weight", type=float, default=1.0)
    p.add_argument("--tf-ca-weight", type=float, default=0.0,
                   help="Auxiliary CA loss decoded from teacher-forced H_future. Gives the "
                        "encoder a direct gradient from future atomic positions, bypassing the "
                        "noisy kernel rollout. 0 disables (original behaviour).")
    p.add_argument("--phys-weight", type=float, default=0.1)
    p.add_argument("--anchor-weight", type=float, default=0.0)
    p.add_argument("--variance-weight", type=float, default=0.0)
    p.add_argument("--variance-floor", type=float, default=0.0)
    p.add_argument("--ens-rmsf-weight", type=float, default=0.0,
                   help="Supervise the per-residue RMSF of the M-sample decoded cloud "
                        "against the true temporal RMSF (differentiable surrogate for "
                        "the rmsf_mae ensemble metric). 0 disables.")
    p.add_argument("--ens-rmsf-corr-weight", type=float, default=0.0,
                   help="Maximise the Pearson correlation (RMSF-r) between the model's "
                        "per-residue RMSF profile and the true profile -- supervises the "
                        "SHAPE of flexibility (which residues move most), unlike "
                        "--ens-rmsf-weight which only matches amplitude. 0 disables.")
    p.add_argument("--ens-var-weight", type=float, default=0.0,
                   help="Log-domain L1 between the model's per-residue sample variance "
                        "and the true trajectory variance (variance_w2 surrogate).")
    p.add_argument("--ens-spread-weight", type=float, default=0.0,
                   help="One-sided hinge penalising a decoded cloud that spreads LESS "
                        "than the true trajectory. Anti-collapse term; 0 disables.")
    p.add_argument("--es-num-samples", type=int, default=4)
    p.add_argument("--ca-horizon-steps", type=int, default=None,
                   help="If set, CA supervision only on the first N rollout steps.")
    p.add_argument("--ca-decode-steps", type=int, default=None,
                   help="Decode at most N evenly-spaced rollout steps for the CA loss "
                        "(None/0 = every step). Bounds the L^2 edge-tensor memory of long "
                        "rollouts while still supervising the full horizon.")
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
        relative_position_features=getattr(args, "relative_position_features", False),
        anchor_conditioned=getattr(args, "anchor_conditioned", False),
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
        rollout_grad_steps=getattr(args, "rollout_grad_steps", None) or None,
    )
    return StageB1b(cfg).to(device)


def build_decoder(args, device):
    dtype = getattr(args, "decoder_type", "mlp")
    if dtype in ("mp", "cri"):
        cfg = MessagePassingConfig(
            latent_dim=args.latent_dim, bath_dim=args.bath_dim,
            hidden=args.decoder_hidden, depth=args.decoder_depth,
            max_update=args.max_update,
            mp_depth=args.mp_depth, mp_heads=args.mp_heads,
            use_hs_cross=getattr(args, "hs_cross_attention", True),
            use_bath_coupling=getattr(args, "bath_coupling_edges", True),
            hs_heads=getattr(args, "hs_heads", 4),
            mp_dropout=getattr(args, "decoder_dropout", 0.0),
        )
        cls = CRIDecoder if dtype == "cri" else MessagePassingCADecoder
        return cls(cfg).to(device)
    cfg = FromNoneConfig(latent_dim=args.latent_dim, bath_dim=args.bath_dim,
                         hidden=args.decoder_hidden, depth=args.decoder_depth,
                         max_update=args.max_update)
    return SimpleCADecoder(cfg).to(device)


def encode_frames(encoder, coords, rmask, aatype, ridx, cid, normalizer=None,
                  anchor_coords=None, anchor_mask=None, frame_chunk=None):
    """[B,T,L,4,3] -> [B,T,L,dH] (trainable; H standardized if a normalizer is given).

    anchor_coords: [B,L,4,3] optional anchor frame; anchor_mask: [B,L].  When both
    are provided and the encoder has anchor_conditioned=True, each frame is encoded
    with a reference signal derived from the anchor's canonical PCA frame, making
    H_future - H_anchor a displacement-relevant signal.

    frame_chunk: if set, encode at most this many (B*T) samples at once to bound
    the encoder's O(B*T*L^2) attention memory.  When None the whole batch is
    processed in one call (original behaviour).
    """
    from protein_jepa_frame.models.encoder import _canonical_frame
    B, T, L = coords.shape[:3]
    anc_centroid = anc_basis = None
    if (anchor_coords is not None and anchor_mask is not None
            and getattr(encoder, 'anchor_conditioned', False)):
        anc_centroid, anc_basis = _canonical_frame(
            anchor_coords, anchor_mask.bool(),
        )   # [B, 3], [B, 3, 3]
        anc_centroid = anc_centroid[:, None].expand(B, T, 3).reshape(B * T, 3)
        anc_basis   = anc_basis[:, None].expand(B, T, 3, 3).reshape(B * T, 3, 3)

    BT = B * T
    coords_flat    = coords.reshape(BT, 1, L, 4, 3)
    rmask_flat     = rmask.reshape(BT, 1, L)
    aatype_flat    = aatype[:, None].expand(B, T, L).reshape(BT, L)
    ridx_flat      = ridx[:, None].expand(B, T, L).reshape(BT, L)
    cid_flat       = cid[:, None].expand(B, T, L).reshape(BT, L)

    chunk = max(1, int(frame_chunk)) if frame_chunk else BT
    if chunk >= BT:
        H_flat = encoder(coords_flat, rmask_flat, aatype_flat, ridx_flat, cid_flat,
                         token_mask=rmask_flat,
                         anchor_centroid=anc_centroid,
                         anchor_basis=anc_basis)[:, 0]
    else:
        pieces = []
        for s in range(0, BT, chunk):
            sl = slice(s, min(s + chunk, BT))
            ac = anc_centroid[sl] if anc_centroid is not None else None
            ab = anc_basis[sl]    if anc_basis    is not None else None
            h  = encoder(coords_flat[sl], rmask_flat[sl], aatype_flat[sl],
                         ridx_flat[sl], cid_flat[sl],
                         token_mask=rmask_flat[sl],
                         anchor_centroid=ac, anchor_basis=ab)[:, 0]
            pieces.append(h)
        H_flat = torch.cat(pieces, dim=0)

    H = H_flat.reshape(B, T, L, -1)
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
        grad_steps=getattr(kernel.cfg, "rollout_grad_steps", None) if training else None,
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


def _ca_step_indices(n_steps: int, max_steps: int | None):
    """Evenly-spaced rollout steps to decode for the CA loss (None = all steps).

    Decoding every step of a long rollout materialises an [N,L,L,edge] tensor per
    step, and autograd retains each chunk's graph, so memory grows with the number
    of decoded steps: at 640 steps x 256 residues the edge tensor alone is tens of
    GB.  We decode a bounded number of steps spread across the WHOLE horizon, so
    the loss still supervises the long end (e.g. 64 ns) rather than only the cheap
    prefix.  Returns None when every step should be decoded.
    """
    if not max_steps or max_steps >= n_steps:
        return None
    return torch.linspace(0, n_steps - 1, int(max_steps)).round().long().unique()


def forecast_objective(encoder, kernel, decoder, batch, args, n_steps, weights,
                       *, generator=None, training=True, normalizer=None):
    K = args.history_frames
    # The dataloader emits clips of length K + (loader horizon); infer it from the
    # batch so training (multi-lag max) and validation (its own horizon) both work.
    clip_frames = batch["coords"].shape[1]
    batch = {k: v for k, v in batch.items()}
    history, targets, mask, dt, batch = split_batch(batch, K, n_steps, args.step_ps, clip_frames)
    aat, ridx, cid = (batch[k] for k in ("aatype", "residue_index", "chain_id"))
    # Anchor frame = the last history frame. With --anchor-conditioned the encoder
    # expresses every frame's CA in THIS frame's canonical PCA basis, so H encodes
    # how each residue sits in a fixed reference frame and dH = H_future - H_anchor
    # is a displacement rather than a per-frame shape descriptor.
    anchor_coords = history[:, -1]
    anchor_mask = batch["residue_mask"][:, K - 1]
    H_hist_raw = encode_frames(encoder, history, batch["residue_mask"][:, :K], aat, ridx, cid,
                               anchor_coords=anchor_coords, anchor_mask=anchor_mask,
                               frame_chunk=getattr(args, "encode_frame_chunk", None))
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
    # --- CA decodability (primary): Kabsch-aligned CA point loss ---
    # Decode a bounded, evenly-spaced subset of rollout steps instead of all of
    # them: memory per decoded step scales with L^2 * edge_dim, so decoding all
    # 640 steps of a 64 ns rollout at 100 ps/frame would need tens of GB.  The
    # subset still spans the whole horizon, so long-lag supervision is preserved.
    decode_idx = _ca_step_indices(n_steps, getattr(args, "ca_decode_steps", None))
    if decode_idx is not None:
        decode_idx = decode_idx.to(H_pred.device)
        H_sel = H_pred[:, :, 1:].index_select(2, decode_idx)
        S_sel = None if S_pred is None else S_pred[:, :, 1:].index_select(2, decode_idx)
        tgt_sel = targets[:, 1:].index_select(1, decode_idx)
    else:
        H_sel, S_sel, tgt_sel = H_pred[:, :, 1:], (None if S_pred is None else S_pred[:, :, 1:]), targets[:, 1:]

    # Encode ONLY the target frames we actually score.  Every loss below
    # (CA, tf_ca, ES, variance) uses the same step subset, so encoding the unused
    # future frames was pure waste: at 100 ps/frame a 64 ns horizon is 641 frames,
    # which at L=256 residues needs ~62 GB of encoder attention.  Encoding just
    # the selected steps cuts encoder work ~40x and keeps peak memory flat.
    # residue_mask[:, K:] = the n_steps FUTURE frames (index K..K+n_steps-1).
    # [:, K-1:] would include the anchor frame and give n_steps+1 entries,
    # causing a shape mismatch when n_steps is small (e.g. 1 at 100 ps).
    fut_mask = batch["residue_mask"][:, K:]          # [B, n_steps, L]
    if decode_idx is not None:
        H_target = encode_frames(encoder,
                                 targets[:, 1:].index_select(1, decode_idx),
                                 fut_mask.index_select(1, decode_idx),
                                 aat, ridx, cid, normalizer,
                                 anchor_coords=anchor_coords, anchor_mask=anchor_mask,
                                 frame_chunk=getattr(args, "encode_frame_chunk", None))
    else:
        H_target = encode_frames(encoder, targets[:, 1:], fut_mask,
                                 aat, ridx, cid, normalizer,
                                 anchor_coords=anchor_coords, anchor_mask=anchor_mask,
                                 frame_chunk=getattr(args, "encode_frame_chunk", None))

    S_fut = S_sel
    ca_pred = decode_paths(decoder, H_sel, S_fut, H_anchor, batch["coords"][:, K - 1], mask)
    true_ca = tgt_sel[..., 1, :]                                # [B,T_sel,L,3] (atom 1 = CA)
    ca_loss = weighted_ca_rmsd(ca_pred, true_ca, mask)
    persist = persistence_ref_rmsd(true_ca, targets[:, 0, :, 1, :], mask)

    # --- Auxiliary teacher-forced CA loss (direct encoder gradient) ---
    # The rolled-out CA loss reaches the encoder only through the stochastic kernel
    # rollout, which dilutes the signal to near-zero (ridge probe: H carries no
    # linearly-decodable dynamics). Decoding the teacher-forced H_target against the
    # true future CA gives the encoder a clean gradient tied to atomic motion.
    # S is detached so the auxiliary term targets the encoder, not the kernel.
    # H_target is now already restricted to decode_idx, so no further slicing.
    tf_ca = ca_loss.new_zeros(())
    tf_w = getattr(args, "tf_ca_weight", 0.0)
    if tf_w > 0:
        H_tf = H_target[None]                                          # [1,B,T_sel,L,d]
        S_tf = None if S_pred is None else S_pred[:1, :, 1:].index_select(2, decode_idx).detach() \
            if decode_idx is not None else S_pred[:1, :, 1:].detach()
        ca_tf_pred = decode_paths(decoder, H_tf, S_tf, H_anchor,
                                  batch["coords"][:, K - 1], mask)
        tf_ca = weighted_ca_rmsd(ca_tf_pred, true_ca, mask)

    # --- Energy score on the H rollout path (distributional term) ---
    if decode_idx is not None:
        es = normalized_path_es(H_pred[:, :, 1:].index_select(2, decode_idx), H_target, mask)
    else:
        es = normalized_path_es(H_pred[:, :, 1:], H_target, mask)
    phys = (physical_energy_score(ca_pred, tgt_sel[..., 1, :], mask)
            if args.phys_weight > 0 else es.new_zeros(()))
    anchor_term = es.new_zeros(())
    var_term = temporal_variance_floor(
        torch.cat([H_hist, H_target], dim=1), mask, args.variance_floor,
    ) if args.variance_weight > 0 else es.new_zeros(())

    # --- Ensemble supervision on the decoded cloud (v4) ---
    # The point CA loss is minimised by the conditional mean and never asks the
    # M rollouts to differ from one another, so the decoder collapses to a
    # near-zero variance cloud.  Every distributional eval metric then compares
    # that tight cluster against a trajectory that actually spans several A and
    # can score WORSE than persistence.  These terms supervise the spread
    # directly.  Each needs M >= 2 rollouts to define a sample variance.
    ens_rmsf = ens_rmsf_corr = ens_var = ens_spread = es.new_zeros(())
    # Need M >= 2 rollouts to define a sample variance, and T >= 2 future frames
    # to define a temporal variance.  At T == 1 the true spread is identically
    # zero and the term would actively collapse the decoder -- the evaluator
    # likewise skips windows with a single future frame.
    if ca_pred.shape[0] >= 2 and ca_pred.shape[2] >= 2:
        w_rmsf = getattr(weights, "ens_rmsf_weight", 0.0)
        w_rmsf_corr = getattr(weights, "ens_rmsf_corr_weight", 0.0)
        w_var = getattr(weights, "ens_var_weight", 0.0)
        w_spread = getattr(weights, "ens_spread_weight", 0.0)
        if w_rmsf > 0:
            ens_rmsf = ensemble_rmsf_loss(ca_pred, true_ca, mask)
        if w_rmsf_corr > 0:
            ens_rmsf_corr = ensemble_rmsf_corr_loss(ca_pred, true_ca, mask)
        if w_var > 0:
            ens_var = ensemble_variance_loss(ca_pred, true_ca, mask)
        if w_spread > 0:
            ens_spread = ensemble_spread_floor(ca_pred, true_ca, mask)

    total = (weights.es_weight * es + weights.ca_weight * ca_loss
             + tf_w * tf_ca
             + weights.phys_weight * phys + weights.anchor_weight * anchor_term
             + weights.variance_weight * var_term
             + getattr(weights, "ens_rmsf_weight", 0.0) * ens_rmsf
             + getattr(weights, "ens_rmsf_corr_weight", 0.0) * ens_rmsf_corr
             + getattr(weights, "ens_var_weight", 0.0) * ens_var
             + getattr(weights, "ens_spread_weight", 0.0) * ens_spread)
    return total, {"H_pred": H_pred, "H_target": H_target, "H_anchor": H_anchor,
                   "ca_pred": ca_pred, "persist": persist}, {
        "total": total, "es": es, "ca": ca_loss, "tf_ca": tf_ca, "phys": phys,
        "anchor": anchor_term, "variance": var_term, "persist": persist,
        "ens_rmsf": ens_rmsf, "ens_rmsf_corr": ens_rmsf_corr,
        "ens_var": ens_var, "ens_spread": ens_spread,
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
                                variance_weight=args.variance_weight,
                                ens_rmsf_weight=args.ens_rmsf_weight,
                                ens_rmsf_corr_weight=args.ens_rmsf_corr_weight,
                                ens_var_weight=args.ens_var_weight,
                                ens_spread_weight=args.ens_spread_weight),
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
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except (TypeError, RuntimeError):
            # Checkpoint written by a different torch build / device count; RNG
            # reproducibility across the resume boundary is not important here.
            pass


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


# Weighted multi-lag cycle over *physical* horizons (ns).  Short horizons are
# cheap and common; the expensive long ones are sampled but rarer.  The cycle
# covers every protocol horizon so no lag is unseen.
# Weighted cycle over physical timescales (ns).  Short horizons appear far more
# often because (a) 100 ps-resolution data has abundant short-range windows,
# (b) single-step accuracy is the foundation for multi-step rollouts, and (c)
# the 0.1–0.5 ns band is entirely new compared with the prior 1 ns/frame data.
# Long horizons (8–64 ns) are kept so the model generalises across timescales.
MULTILAG_CYCLE_NS = (
    0.1, 0.1, 0.2, 0.1, 0.5, 0.1, 0.2, 0.5,  # short — dense sampling
    1,   0.2, 0.1, 0.5,                         # 1 ns interleaved with short
    2,   1,   0.2, 0.5,                         # 2 ns
    4,   1,   0.5,                              # 4 ns
    8,   2,   1,                                # 8 ns
    16,  4,   8,                                # 16 ns
    32,  64,  16, 8, 4, 2, 1, 0.5, 0.2, 0.1,  # long tail, descend back
)

# Cycle for 1 ns/frame data (step_ps >= 500, e.g. mdCATH at 1000 ps/frame).
# The original MULTILAG_CYCLE_NS contains 0.1/0.2/0.5 ns entries that all
# collapse to 1 frame at step_ps=1000 (round(0.1*1000/1000)=0 -> max 1), so
# 62.9% of batches become 1-frame rollouts where ensemble losses (rmsf_corr,
# rmsf_mae, variance, spread) are all disabled by the T>=2 guard.  This cycle
# keeps the same multi-timescale coverage but removes the sub-1ns dead weight,
# pushing the 1-frame fraction down to ~29% and giving ensemble supervision
# roughly 2.4x more actual training batches at identical total steps.
MULTILAG_CYCLE_NS_1NS = (
    1,  1,  2,  1,   # 1 ns — lightest, most frequent
    4,  1,  2,  4,   # 4 ns interleaved
    8,  2,  1,  4,   # 8 ns
    16, 4,  8,       # 16 ns
    32, 64, 16, 8, 4, 2, 1,  # long tail, descend back
)


def _pick_cycle(step_ps: float):
    """Return the appropriate ns cycle for the data resolution.

    At step_ps >= 500 (1 ns/frame data like mdCATH) the sub-ns entries in
    MULTILAG_CYCLE_NS collapse to 1 frame and flood ~63% of batches into a
    1-frame bucket where ensemble losses are disabled.  Use the 1ns-aware
    cycle instead so distributional supervision fires more often.
    """
    return MULTILAG_CYCLE_NS_1NS if step_ps >= 500.0 else MULTILAG_CYCLE_NS


def horizon_schedule(args, step):
    """Rollout length (frames) for this step: fixed or a weighted multi-lag cycle."""
    if args.multi_lag:
        cycle = cycle_frames(args.step_ps, _pick_cycle(args.step_ps))
        return cycle[step % len(cycle)]
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
                              variance_weight=args.variance_weight,
                              ens_rmsf_weight=args.ens_rmsf_weight,
                              ens_rmsf_corr_weight=args.ens_rmsf_corr_weight,
                              ens_var_weight=args.ens_var_weight,
                              ens_spread_weight=args.ens_spread_weight)
    weights.validate()

    ph = protocol_horizons(args)
    max_h = max(ph) if args.multi_lag else args.horizon_steps
    val_h = max(args.validation_horizon_steps or args.horizon_steps, max_h if args.multi_lag else 0)
    train_loader = _loader(args, args.manifest, True, max_h if args.multi_lag else args.horizon_steps)
    val_loader = _loader(args, args.val_manifest, False, val_h) if (args.val_manifest or not args.manifest) else None

    step, best, bad, validation = 0, float("inf"), 0, {}
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        # strict=False so a checkpoint from before a new input channel (e.g.
        # --relative-position-features) can still warm-start; missing keys are
        # freshly initialised and reported.
        for name, mdl in (("encoder", encoder), ("kernel", kernel), ("decoder", decoder)):
            result = mdl.load_state_dict(ck[name], strict=False)
            if result.missing_keys or result.unexpected_keys:
                print(f"[resume] {name}: missing={result.missing_keys} "
                      f"unexpected={result.unexpected_keys}", flush=True)
        if "normalizer" in ck:
            normalizer.mean_.copy_(torch.as_tensor(ck["normalizer"]["mean"]).to(device))
            normalizer.scale_.copy_(torch.as_tensor(ck["normalizer"]["scale"]).to(device))
        try:
            optimizer.load_state_dict(ck["optimizer"])
        except (ValueError, KeyError) as exc:
            # A new encoder channel changed the parameter count; AdamW moments
            # cannot be mapped. Keep the weights, restart moment estimates.
            print(f"[resume] optimizer state incompatible ({type(exc).__name__}); "
                  f"restarting AdamW moments", flush=True)
        restore_rng(ck["rng_state"])
        step, best, bad, validation = ck["step"], ck["best_score"], ck["bad_validations"], ck["validation"]

    (out / "config.json").write_text(json.dumps({
        "train_args": vars(args), "training_contract": TRAINING_CONTRACT,
        "protocol_horizons": list(ph),
        "protocol_horizons_ns": list(getattr(args, "horizons_ns", None) or PROTOCOL_HORIZONS_NS),
    }, indent=2))
    print(f"[from-none] encoder=fresh latent={args.latent_dim} bath={args.bath_dim} "
          f"multi_lag={args.multi_lag} horizons={ph if args.multi_lag else args.horizon_steps} frames "
          f"(step_ps={args.step_ps})",
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
