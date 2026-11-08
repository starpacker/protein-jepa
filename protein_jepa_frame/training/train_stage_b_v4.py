"""Stage B (v4) training: autonomous GLE over H_dyn, conditioned on H_static.

Mirrors ``train_stage_b1b`` but replaces the frozen single-H oracle with the v4
static/dynamic bundle:

  H_static = E_s(c, X_0)          -- frozen Stage A static branch (conditions the bath)
  H_dyn,t  = E_d(X_t | H_static)  -- frozen Stage A dynamic branch (magnitude-carrying)
  (H_dyn, S) -> (H_dyn, S)        -- trainable StageB1b GLE (energy + bath + history posterior)
  X_hat,t  = D(H_static, H_dyn,t) -- frozen Stage A conditional decoder

Only the GLE is trained.  The Stage A encoders/decoder are frozen and never see
future frames in the forecast route.  ``seq_feature=H_static`` is the static
conditioning of ``StaticBathParams`` (D, C, Gamma), exactly the spec §3 contract.

Usage:
    PYTHONPATH=/tmp/lasp_stability_deps:/liuyutian/protein_jepa_frame \
    python -m protein_jepa_frame.training.train_stage_b_v4 \
        --stage-a-v4-ckpt artifacts/v4_stage_a/run_400/checkpoint_best.pt \
        --manifest artifacts/mdcath_manifest_full/train.jsonl \
        --val-manifest artifacts/mdcath_manifest_full/val.jsonl \
        --output artifacts/v4_stage_b/smoke --steps 200 ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Sampler, Subset

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import split_contiguous_window, validate_manifest_split
from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.models.stage_b1b.objectives import energy_score
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame, fit_history_pose
from protein_jepa_frame.models.static_dynamic_v4 import (
    StaticEncoder, DynamicEncoder, ConditionalStructureDecoder,
)
from protein_jepa_frame.models.stage_b_v4 import StageBV4
from protein_jepa_frame.training.train_utils import _device, _lr_scale, _move_batch, _set_reproducibility
from protein_jepa_frame.utils.backbone_geometry import backbone_torsion_targets

# Reuse the pure loss/split helpers from the (already-tested) Stage B1b trainer.
from protein_jepa_frame.training.train_stage_b1b import (
    EpochPermutation, freeze_module, split_contiguous_batch,
    normalized_path_energy_score, physical_energy_score, displacement_point_loss,
    physical_path_features,
)


def load_v4_bundle(path, device, latent_dim=128):
    """Load the frozen Stage A v4 static/dynamic bundle from a v4 Stage A ckpt."""
    ck = torch.load(path, map_location=device, weights_only=False)
    static = StaticEncoder(latent_dim).to(device)
    # Reconstruct the dynamic branch with the same geometry flags as the saved
    # checkpoint (inferred from the presence of the relpos/orientation modules).
    dyn_sd = ck["dynamic_encoder"]
    orientation_features = "encoder.orientation_bias.weight" in dyn_sd
    relative_position_features = "encoder.relpos_proj.weight" in dyn_sd
    dynamic = DynamicEncoder(
        latent_dim,
        orientation_features=orientation_features,
        relative_position_features=relative_position_features,
        anchor_conditioned=False,
    ).to(device)
    decoder = GeometricDeltaCADecoder(latent_dim=latent_dim).to(device)
    conditional = ConditionalStructureDecoder(decoder, latent_dim).to(device)

    static.load_state_dict(ck["static_encoder"], strict=True)
    dynamic.load_state_dict(ck["dynamic_encoder"], strict=True)
    decoder.load_state_dict(ck["decoder"], strict=True)
    if "conditional_decoder" in ck:
        conditional.load_state_dict(ck["conditional_decoder"], strict=True)
    return freeze_module(static), freeze_module(dynamic), freeze_module(conditional), ck


@torch.no_grad()
def encode_static(static_encoder, anchor_coords, mask, aatype, ridx, cid):
    """H_static = E_s(c, X_0) from the anchor frame only (leakage-safe)."""
    return static_encoder(anchor_coords, mask, aatype, ridx, cid)


@torch.no_grad()
def encode_dyn(dynamic_encoder, coords, h_static, mask, aatype, ridx, cid):
    """H_dyn,t = E_d(X_t | H_static) over all frames [B,T,L,D]."""
    B, T, L = coords.shape[:3]
    rm = mask[:, None].expand(B, T, L).reshape(B * T, L)
    expand = lambda x: x[:, None].expand(B, T, L).reshape(B * T, L)
    h_static_exp = h_static[:, None].expand(B, T, L, -1).reshape(B * T, L, -1)
    h_dyn = dynamic_encoder(
        coords.reshape(B * T, 1, L, 4, 3), rm[:, None],
        expand(aatype), expand(ridx), expand(cid),
        h_static=h_static_exp, token_mask=rm[:, None],
    )[:, 0].reshape(B, T, L, -1)
    return h_dyn


def autonomous_samples_v4(model, H_dyn_hist, H_static, dt_hist, residue_mask, temperature,
                          n_steps, num_samples, *, generator=None, training=True):
    """History-only forecast: evolve H_dyn+S, conditioned on H_static."""
    M = num_samples
    B, K, L, d = H_dyn_hist.shape
    repeat = lambda x: x.unsqueeze(0).expand(M, *x.shape).reshape(M * B, *x.shape[1:])
    hist, dt, mask = repeat(H_dyn_hist), repeat(dt_hist), repeat(residue_mask)
    static = repeat(H_static)
    temp = None
    if temperature is not None:
        t = torch.as_tensor(temperature, device=H_dyn_hist.device, dtype=H_dyn_hist.dtype)
        temp = repeat(t.expand(B) if t.ndim == 0 else t.reshape(B))
    model.gle.s0_mode = "exact"
    H, S = model.gle.rollout(
        hist[:, -1], mask, temp, n_steps=n_steps, H_hist=hist, dt_hist=dt,
        seq_feature=static, generator=generator, detach=not training, return_bath=True,
        collective_mobility=True,
    )
    H = H.reshape(M, B, n_steps + 1, L, d)
    if S is not None:
        S = S.reshape(M, B, n_steps + 1, L, model.cfg.bath_dim)
    return H, S


def decode_paths_v4(conditional_decoder, h_dyn_paths, h_static, batch, mask, anchor_coords,
                    *, differentiable: bool = False):
    """Decode autonomous H_dyn paths through the frozen conditional decoder.

    Returns a backbone tensor [M, B, T, L, 4, 3] with the CA at channel 1 (the
    same convention as ``physical_energy_score``/``displacement_point_loss``).

    ``differentiable=True`` (training) runs the decoder *without* ``no_grad`` so
    the decoded-geometry loss (adjacent-CA bond, physical ES) propagates gradient
    back through the frozen decoder's Jacobian into the GLE's H_dyn output.  The
    decoder weights stay frozen (they are never updated), but the gradient path
    from decoded geometry -> H_dyn is restored — this is the spec §7 #8 requirement
    "no accidental detach/stopgrad in intended paths".  ``differentiable=False``
    (validation/eval) keeps the detached fast path.
    """
    ctx = torch.enable_grad() if differentiable else torch.no_grad()
    with ctx:
        M, B, T, L, d = h_dyn_paths.shape
        aatype, ridx, cid = batch["aatype"], batch["residue_index"], batch["chain_id"]
        R, t = RigidBodyFrame.from_backbone(anchor_coords[..., 0, :], anchor_coords[..., 1, :],
                                            anchor_coords[..., 2, :])
        repeat = lambda x: x[None, :, None].expand(M, B, T, *x.shape[1:]).reshape(M * B * T, *x.shape[1:])
        h_flat = h_dyn_paths.reshape(M * B * T, L, d)
        h_static_flat = h_static[None, :, None].expand(M, B, T, L, -1).reshape(M * B * T, L, -1)
        ca = conditional_decoder(
            h_flat, h_static_flat,
            repeat(aatype), repeat(ridx), repeat(cid),
            repeat(R), repeat(t), repeat(mask),
        ).reshape(M, B, T, L, 3)
        backbone = torch.zeros(M, B, T, L, 4, 3, device=ca.device, dtype=ca.dtype)
        backbone[..., 1, :] = ca
    return backbone


def adjacent_ca_bond_loss(pred_bb, target_bb, batch, mask):
    """Differentiable loss over adjacent-CA bond lengths, aligned to the gate metric.

    The Stage B acceptance gate (``validation/ca_bond_violation_fraction``) measures
    the fraction of *adjacent* Cα bonds outside [3.0, 4.5] Å.  ``physical_energy_score``
    samples all-pairs CA distances/contacts/torsions and has no direct adjacent-bond
    term, which is why the physical ES plateaued while the bond metric stopped
    improving.  This loss directly penalises the same quantity the gate measures:

      bond_len(pred) vs bond_len(target), over true-adjacent residues only.

    Both the soft L1 deviation from the target length and a hard margin outside the
    valid [3.0, 4.5] Å window are summed, so the gradient reaches bond geometry
    directly.  ``pred_bb`` is [M, B, T, L, 4, 3] with CA at channel 1; ``target_bb`` is
    [B, T, L, 4, 3].
    """
    M, B, T, L = pred_bb.shape[:4]
    # True-adjacent mask (same chain, consecutive index, both valid) — the exact
    # adjacency rule used by validate_v4, broadcast over [M, T].
    adjacent = ((batch["chain_id"][:, 1:] == batch["chain_id"][:, :-1])
                & (batch["residue_index"][:, 1:] - batch["residue_index"][:, :-1] == 1)
                & mask[:, 1:] & mask[:, :-1])                     # [B, L-1]
    adj = adjacent[None, :, None, :].expand(M, B, T, L - 1)       # [M, B, T, L-1]
    pred_ca = pred_bb[..., 1, :]                                  # [M, B, T, L, 3]
    tgt_ca = target_bb[..., 1, :]                                 # [B, T, L, 3]
    pred_len = (pred_ca[..., 1:, :] - pred_ca[..., :-1, :]).norm(dim=-1)   # [M, B, T, L-1]
    tgt_len = (tgt_ca[..., 1:, :] - tgt_ca[..., :-1, :]).norm(dim=-1).detach()  # [B, T, L-1]
    # Soft-L1 toward the GT length, plus a margin outside the valid window.
    dev = (pred_len - tgt_len[None].expand_as(pred_len)).abs()
    outside = ((pred_len - 4.5).clamp_min(0) + (3.0 - pred_len).clamp_min(0))
    per_bond = (dev + outside) * adj
    return per_bond.sum() / adj.sum().clamp_min(1)


def physical_manifold_loss(pred_bb, target_bb, batch, mask, *, clash_cutoff=4.5):
    """Torsion-free physical-manifold loss over decoded CA geometry.

    The physical energy score (``physical_energy_score``) samples torsion sin/cos
    through ``atan2``/``cross``, whose gradient is singular at degenerate (collinear)
    coordinates — which is why it had to be detached in Milestone 8, leaving only the
    single adjacent-bond term carrying decoded-geometry gradient.  This loss is the
    safe, fully-differentiable complement: it scores the *decoded* backbone's CA
    bond lengths, pseudo-angles, and steric clashes using only ``norm`` operations
    (no torsion, no ``atan2``/``cross``), so its gradient reaches the GLE through the
    frozen decoder without singularity.

    Terms (each masked over true-adjacent / valid residues):

      * **bond**: adjacent CA length deviates from the GT length (same as
        ``adjacent_ca_bond_loss``, kept here so manifold loss is self-contained).
      * **angle**: the pseudo-angle at each CA (angle of the two adjacent CA bonds)
        deviates from the GT angle.  Uses the cosine (dot-product) form, which is
        smooth everywhere.
      * **clash**: any non-adjacent CA pair closer than ``clash_cutoff`` Å.

    ``pred_bb`` is [M, B, T, L, 4, 3] with CA at channel 1; ``target_bb`` is
    [B, T, L, 4, 3].  Returns a scalar in Å (bond/angle) + a dimensionless clash
    count, averaged over valid entries.
    """
    M, B, T, L = pred_bb.shape[:4]
    pred_ca = pred_bb[..., 1, :]                                   # [M, B, T, L, 3]
    tgt_ca = target_bb[..., 1, :]                                  # [B, T, L, 3]

    # --- adjacency mask (same chain, consecutive index, both valid) ---
    adjacent = ((batch["chain_id"][:, 1:] == batch["chain_id"][:, :-1])
                & (batch["residue_index"][:, 1:] - batch["residue_index"][:, :-1] == 1)
                & mask[:, 1:] & mask[:, :-1])                       # [B, L-1]
    adj = adjacent[None, :, None, :].expand(M, B, T, L - 1)        # [M, B, T, L-1]

    # --- bond: |pred_len - tgt_len| over true-adjacent pairs ---
    pred_len = (pred_ca[..., 1:, :] - pred_ca[..., :-1, :]).norm(dim=-1)       # [M,B,T,L-1]
    tgt_len = (tgt_ca[..., 1:, :] - tgt_ca[..., :-1, :]).norm(dim=-1).detach()
    bond_dev = (pred_len - tgt_len[None].expand_as(pred_len)).abs() * adj
    n_bond = adj.sum().clamp_min(1)

    # --- angle: cosine of the CA pseudo-angle at each interior residue ---
    # v1 = CA_{i-1} -> CA_i, v2 = CA_{i+1} -> CA_i (both point inward), for i in 1..L-2.
    if L >= 3:
        v1 = pred_ca[..., :-2, :] - pred_ca[..., 1:-1, :]           # [M,B,T,L-2,3]
        v2 = pred_ca[..., 2:, :] - pred_ca[..., 1:-1, :]
        n1 = v1.norm(dim=-1).clamp_min(1e-6)
        n2 = v2.norm(dim=-1).clamp_min(1e-6)
        cos_pred = (v1 * v2).sum(dim=-1) / (n1 * n2)                # [M,B,T,L-2]
        tv1 = tgt_ca[..., :-2, :] - tgt_ca[..., 1:-1, :]
        tv2 = tgt_ca[..., 2:, :] - tgt_ca[..., 1:-1, :]
        tn1 = tv1.norm(dim=-1).clamp_min(1e-6)
        tn2 = tv2.norm(dim=-1).clamp_min(1e-6)
        cos_tgt = ((tv1 * tv2).sum(dim=-1) / (tn1 * tn2)).detach()  # [B,T,L-2]
        # interior residues are valid where both adjacent bonds are valid
        ang_mask = (adjacent[:, :-1] & adjacent[:, 1:])[None, :, None, :].expand(M, B, T, L - 2)
        angle_dev = (cos_pred - cos_tgt[None].expand_as(cos_pred)).abs() * ang_mask
        n_angle = ang_mask.sum().clamp_min(1)
    else:
        angle_dev = pred_ca.new_zeros(()).expand(M, B, T, 1)
        n_angle = angle_dev.new_ones(())

    # --- clash: non-adjacent CA pairs closer than clash_cutoff ---
    # Only sample a bounded set of pairs (all non-adjacent pairs within a window)
    # to keep it O(L) and mask out bonded/self pairs.
    clash = pred_ca.new_zeros(())
    n_clash = pred_ca.new_ones(())
    if L >= 3:
        rmask = mask.bool()                                         # [B, L]
        # pair i, i+k for k in 2..min(8, L-1): guaranteed non-adjacent
        max_k = min(8, L - 1)
        for k in range(2, max_k + 1):
            d = (pred_ca[..., :-k, :] - pred_ca[..., k:, :]).norm(dim=-1)  # [M,B,T,L-k]
            pm = (rmask[:, :-k] & rmask[:, k:])[None, :, None, :].expand(M, B, T, L - k)
            clash = clash + ((clash_cutoff - d).clamp_min(0) * pm).sum()
            n_clash = n_clash + pm.sum().clamp_min(1)

    return (bond_dev.sum() / n_bond
            + angle_dev.sum() / n_angle
            + clash / n_clash)


def forecast_objective_v4(model, static_encoder, dynamic_encoder, conditional_decoder, batch, args,
                          *, generator=None, training=True):
    K, N = args.history_frames, args.horizon_steps
    history, targets, mask, dt = split_contiguous_batch(batch, K, N, args.step_ps)
    aat, ridx, cid = (batch[k] for k in ("aatype", "residue_index", "chain_id"))
    anchor_coords = batch["coords"][:, K - 1]  # [B, L, 4, 3]
    H_static = encode_static(static_encoder, anchor_coords, mask, aat, ridx, cid)
    H_dyn_hist = encode_dyn(dynamic_encoder, history, H_static, mask, aat, ridx, cid)
    H_pred, S_pred = autonomous_samples_v4(
        model, H_dyn_hist, H_static, dt, mask, batch.get("temperature_k"), N,
        args.es_num_samples, generator=generator, training=training,
    )
    H_target = encode_dyn(dynamic_encoder, targets, H_static, mask, aat, ridx, cid)
    end = min(N, args.es_horizon_steps or N)
    es = normalized_path_energy_score(H_pred[:, :, 1:end + 1], H_target[:, 1:end + 1], mask)
    physical, point = es.new_zeros(()), es.new_zeros(())
    bond = es.new_zeros(())
    manifold = es.new_zeros(())
    pred_bb = None
    if (args.physical_es_weight > 0 or args.rmsd_weight > 0 or args.bond_weight > 0
            or args.manifold_weight > 0):
        # Differentiable during training so the decoded-geometry (bond) gradient
        # reaches the GLE through the frozen decoder's Jacobian.
        pred_bb = decode_paths_v4(conditional_decoder, H_pred[:, :, 1:], H_static,
                                  batch, mask, anchor_coords, differentiable=training)
    if args.physical_es_weight > 0:
        # physical_energy_score samples torsion sin/cos through atan2/cross, whose
        # gradient is singular at degenerate (off-manifold) coordinates.  Detach it:
        # it is an ensemble proper-scoring term; the *bond* term carries the direct
        # decoded-geometry gradient to the GLE.  (Before the decoder became
        # differentiable it was implicitly detached via @no_grad.)
        physical = physical_energy_score(pred_bb.detach(), targets[:, 1:].unsqueeze(0), batch, mask)
    if args.rmsd_weight > 0:
        end_point = min(N, args.rmsd_horizon_steps or N)
        point = displacement_point_loss(pred_bb[:, :, :end_point].detach(), targets[:, 1:end_point + 1],
                                        history[:, -1], mask)
    if args.bond_weight > 0:
        bond = adjacent_ca_bond_loss(pred_bb, targets[:, 1:], batch, mask)
    if args.manifold_weight > 0:
        manifold = physical_manifold_loss(pred_bb, targets[:, 1:], batch, mask)
    total = (args.es_weight * es + args.physical_es_weight * physical
             + args.rmsd_weight * point + args.bond_weight * bond
             + args.manifold_weight * manifold)
    logs = {"loss/total": float(total.detach()), "loss/latent_es": float(es.detach()),
            "loss/physical_es": float(physical.detach()), "loss/mean_displacement": float(point.detach()),
            "loss/bond": float(bond.detach()), "loss/manifold": float(manifold.detach())}
    return total, logs, {"H_pred": H_pred, "S_pred": S_pred, "H_target": H_target,
                         "H_static": H_static, "predicted_bb": pred_bb}


def _loader(args, manifest, training):
    horizon = args.horizon_steps if training else (args.validation_horizon_steps or args.horizon_steps)
    config = DataConfig(
        manifest_path=manifest, training_mode="clip", num_frames=args.history_frames + horizon,
        frame_stride=args.frame_stride, max_residues=args.max_residues,
    )
    dataset = build_dataset(config)
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


def validate_v4(model, static_encoder, dynamic_encoder, conditional_decoder, loader, args, device):
    val_args = argparse.Namespace(**vars(args))
    val_args.horizon_steps = args.validation_horizon_steps or args.horizon_steps
    val_args.es_horizon_steps = None
    val_args.rmsd_horizon_steps = None
    was_training = model.training
    model.eval()
    records = []
    gen = torch.Generator(device=device).manual_seed(args.seed + 1000003)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        for index, batch in enumerate(loader):
            if index >= args.validation_batches:
                break
            batch = _move_batch(batch, device)
            loss, logs, forecast = forecast_objective_v4(
                model, static_encoder, dynamic_encoder, conditional_decoder, batch, val_args,
                generator=gen, training=False,
            )
            if not torch.isfinite(forecast["H_pred"]).all() or not torch.isfinite(loss):
                raise FloatingPointError("nonfinite autonomous validation forecast or score")
            logs["validation/max_abs_H_dyn"] = float(forecast["H_pred"].abs().max())
            logs["validation/H_dyn_mean_norm"] = float(forecast["H_pred"].norm(dim=-1).mean())
            mask = batch["residue_mask"][:, args.history_frames - 1]
            backbone = forecast["predicted_bb"]
            if backbone is None:
                backbone = decode_paths_v4(conditional_decoder, forecast["H_pred"][:, :, 1:],
                                           forecast["H_static"], batch, mask,
                                           batch["coords"][:, args.history_frames - 1])
            logs["validation/mean_displacement_angstrom"] = float(displacement_point_loss(
                backbone, batch["coords"][:, args.history_frames:],
                batch["coords"][:, args.history_frames - 1], mask,
            ))
            adjacent = ((batch["chain_id"][:, 1:] == batch["chain_id"][:, :-1])
                        & (batch["residue_index"][:, 1:] - batch["residue_index"][:, :-1] == 1)
                        & mask[:, 1:] & mask[:, :-1])
            ca = backbone[..., 1, :]
            bonds = (ca[..., 1:, :] - ca[..., :-1, :]).norm(dim=-1)
            weights = adjacent[None, :, None].expand_as(bonds)
            violations = ((bonds < 3.0) | (bonds > 4.5)) & weights
            logs["validation/ca_bond_violation_fraction"] = float(violations.sum() / weights.sum().clamp_min(1))
            records.append(logs)
    model.train(was_training)
    return {key: sum(r[key] for r in records) / len(records) for key in records[0]}


def capture_rng():
    rng = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def restore_rng(rng):
    torch.set_rng_state(rng["torch"].cpu())
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-a-v4-ckpt", required=True)
    p.add_argument("--manifest")
    p.add_argument("--val-manifest")
    p.add_argument("--output", default="artifacts/v4_stage_b")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--horizon-steps", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-residues", type=int, default=96)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--bath-dim", type=int, default=8)
    p.add_argument("--energy-depth", type=int, default=3)
    p.add_argument("--energy-heads", type=int, default=4)
    p.add_argument("--energy-hidden", type=int, default=128)
    p.add_argument("--history-hidden", type=int, default=128)
    p.add_argument("--step-ps", type=float, default=1000.0)
    p.add_argument("--coupling-scale", type=float, default=1e-3)
    p.add_argument("--diffusion-scale", type=float, default=1e-4)
    p.add_argument("--c-init-std", type=float, default=0.1)
    p.add_argument("--gamma-scale", type=float, default=2e-4)
    p.add_argument("--gamma-floor", type=float, default=2e-5)
    p.add_argument("--max-gamma-step", type=float, default=1.9)
    p.add_argument("--tail-init", type=float, default=1.0,
                   help="Quadratic tail init for the H_dyn energy; larger gives an "
                        "immediate restoring force for a from-scratch H_dyn GLE.")
    p.add_argument("--es-weight", type=float, default=1.0)
    p.add_argument("--physical-es-weight", type=float, default=0.1)
    p.add_argument("--bond-weight", type=float, default=0.0,
                   help="Weight on the adjacent-CA bond-length loss (aligns training "
                        "with the ca_bond_violation gate).")
    p.add_argument("--manifold-weight", type=float, default=0.0,
                   help="Weight on the torsion-free physical-manifold loss (bond + "
                        "CA pseudo-angle + steric clash).  Safe differentiable "
                        "complement to the detached physical energy score.")
    p.add_argument("--es-num-samples", type=int, default=2)
    p.add_argument("--es-horizon-steps", type=int)
    p.add_argument("--rmsd-weight", type=float, default=0.0)
    p.add_argument("--rmsd-horizon-steps", type=int)
    p.add_argument("--validation-every", type=int, default=200)
    p.add_argument("--validation-batches", type=int, default=2)
    p.add_argument("--validation-horizon-steps", type=int)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    if a.es_num_samples < 2:
        p.error("energy score requires at least 2 independent autonomous samples")

    audit = validate_manifest_split(a.manifest, a.val_manifest) if a.manifest else None
    torch.set_num_threads(4)
    _set_reproducibility(a.seed, False)
    device = torch.device(_device(a.device))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists():
        raise FileExistsError("use a fresh output directory")

    static_encoder, dynamic_encoder, conditional_decoder, bundle = load_v4_bundle(
        a.stage_a_v4_ckpt, device, a.latent_dim)

    gle_cfg = StageB1bConfig(
        latent_dim=a.latent_dim, bath_dim=a.bath_dim,
        energy_depth=a.energy_depth, energy_heads=a.energy_heads, energy_hidden=a.energy_hidden,
        history_hidden=a.history_hidden, step_ps=a.step_ps,
        coupling_scale=a.coupling_scale, diffusion_scale=a.diffusion_scale,
        c_init_std=a.c_init_std, gamma_scale=a.gamma_scale, gamma_floor=a.gamma_floor,
        max_gamma_step=a.max_gamma_step, bath_coupling="velocity",
        tail_init=a.tail_init,
    )
    model = StageBV4(gle_cfg).to(device)
    train = _loader(a, a.manifest, True) if a.manifest else None
    val = _loader(a, a.val_manifest, False) if a.val_manifest else None

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    for group in opt.param_groups:
        group["base_lr"] = a.lr

    def record(row):
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def save(tag, step, metrics):
        torch.save({"checkpoint_kind": "v4_stage_b", "gle": model.state_dict(),
                    "cfg": gle_cfg.__dict__, "step": step, "validation": metrics,
                    "optimizer": opt.state_dict(), "rng": capture_rng()},
                   out / f"checkpoint_{tag}.tmp")
        (out / f"checkpoint_{tag}.tmp").replace(out / f"checkpoint_{tag}.pt")

    if val is not None:
        metrics = validate_v4(model, static_encoder, dynamic_encoder, conditional_decoder, val, a, device)
        record(dict(step=0, val=metrics))

    iters = iter(train) if train is not None else None
    import time as _t
    start = _t.time()
    for step in range(1, a.steps + 1):
        if train is None:
            break
        try:
            batch = next(iters)
        except StopIteration:
            iters = iter(train)
            batch = next(iters)
        opt.zero_grad(set_to_none=True)
        batch = _move_batch(batch, device)
        with sdpa_kernel([SDPBackend.MATH]):
            loss, logs, _ = forecast_objective_v4(
                model, static_encoder, dynamic_encoder, conditional_decoder, batch, a,
                generator=None, training=True,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss step {step}")
            # The backward must also run under the MATH kernel: the differentiable
            # decoder path (decode_paths_v4) uses SDPA attention, whose
            # efficient-attention backward is unimplemented on this torch build.
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"nonfinite grad step {step}")
        scale = _lr_scale(step - 1, a.steps, a.warmup_steps)
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * scale
        opt.step()
        if step % a.log_every == 0 or step == 1:
            record(dict(step=step, seconds=_t.time() - start, loss=logs,
                        grad_norm=float(gn)))
        if step % a.validation_every == 0 and val is not None:
            metrics = validate_v4(model, static_encoder, dynamic_encoder, conditional_decoder, val, a, device)
            record(dict(step=step, val=metrics))
            save("latest", step, metrics)
        if step % a.save_every == 0:
            save("best", step, logs)
    if val is not None:
        save("latest", a.steps, logs)
    print(f"done: {a.steps} updates", flush=True)


if __name__ == "__main__":
    main()
