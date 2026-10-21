"""Autonomous B1b: forecast only from history, then score against future targets.

Teacher-forced Kalman NLL is an optional external diagnostic, never an optimized
training loss. The complete calibrated B1a representation/readout is frozen.
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
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Sampler, Subset

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import split_contiguous_window, validate_manifest_split
from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.stage_a_v2 import StageAV2Config
from protein_jepa_frame.models.stage_b1a import StageBOracleConfig, StageBOracleDecoder
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.models.stage_b1b.objectives import energy_score
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame, fit_history_pose
from protein_jepa_frame.training.train_utils import _device, _lr_scale, _move_batch, _set_reproducibility
from protein_jepa_frame.utils.backbone_geometry import backbone_torsion_targets
from protein_jepa_frame.utils.contracts import assert_same_representation, validate_representation_contract

TRAINING_CONTRACT = "autonomous_history_only_v2"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest")
    p.add_argument("--val-manifest")
    p.add_argument("--b1a-ckpt", required=True)
    p.add_argument("--oracle-gate-report", help="Held-out oracle gate JSON required outside --dry-run.")
    p.add_argument("--stage-a-ckpt", help="Provenance only; encoder always comes from B1a.")
    p.add_argument("--output", default="artifacts/stage_b1b_autonomous")
    p.add_argument("--resume")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--horizon-steps", type=int, default=8)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-residues", type=int)
    p.add_argument("--synthetic-size", type=int, default=32)
    p.add_argument("--synthetic-residues", type=int, default=16)
    p.add_argument("--latent-dim", type=int)
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
    p.add_argument("--nll-weight", type=float, default=0.0,
                   help="Must be zero: teacher-forced NLL is diagnostic only.")
    p.add_argument("--es-weight", type=float, default=1.0)
    p.add_argument("--physical-es-weight", type=float, default=0.1)
    p.add_argument("--es-num-samples", type=int, default=4)
    p.add_argument("--es-horizon-steps", type=int)
    p.add_argument("--es-warmup-steps", type=int, default=0,
                   help="CLI migration only; must be zero (no teacher-forced warmup).")
    p.add_argument("--rmsd-weight", type=float, default=0.0,
                   help="Optional autonomous ensemble-mean displacement point loss.")
    p.add_argument("--rmsd-horizon-steps", type=int)
    p.add_argument("--validation-every", type=int, default=200)
    p.add_argument("--validation-batches", type=int, default=4)
    p.add_argument("--validation-horizon-steps", type=int,
                   help="Autonomous validation horizon (may exceed the training horizon).")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.nll_weight != 0:
        p.error("--nll-weight must be 0: B1b predictions must autonomously evolve H+S; "
                "teacher-forced Kalman NLL may only be evaluated separately.")
    if a.es_warmup_steps != 0:
        p.error("--es-warmup-steps must be 0; autonomous training has no NLL-only warmup.")
    if not (a.es_weight > 0 and a.physical_es_weight >= 0 and a.rmsd_weight >= 0):
        p.error("es-weight must be positive; other loss weights must be nonnegative")
    if a.es_num_samples < 2:
        p.error("energy score requires at least 2 independent autonomous samples")
    for key in ("steps", "history_frames", "horizon_steps", "frame_stride", "batch_size",
                "validation_every", "validation_batches", "log_every", "save_every"):
        if getattr(a, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("es_horizon_steps", "rmsd_horizon_steps", "validation_horizon_steps"):
        if getattr(a, key) is not None and getattr(a, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if a.manifest and not a.val_manifest and not a.dry_run:
        p.error("real-data training requires --val-manifest for held-out autonomous selection")
    if not a.dry_run and not a.oracle_gate_report:
        p.error("formal B1b training requires --oracle-gate-report; --dry-run is contract smoke only")
    return a


class EpochPermutation(Sampler):
    """Per-epoch shuffle recoverable from step alone, with an isolated RNG."""

    def __init__(self, data_source, seed):
        self.length, self.seed, self.epoch = len(data_source), seed, 0

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed+self.epoch)
        return iter(torch.randperm(self.length, generator=rng).tolist())

    def __len__(self):
        return self.length


def freeze_module(module):
    module.eval()
    module.requires_grad_(False)
    return module


def load_frozen_representation(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    required = {"encoder", "stage_a_v2_config", "oracle", "stage_b1a_config", "representation_contract"}
    if required.difference(ck):
        raise ValueError(f"B1a checkpoint lacks frozen bundle fields: {sorted(required.difference(ck))}")
    contract = ck["representation_contract"]
    if contract.get("calibration_mode") != "oracle":
        raise ValueError("B1b requires true-H oracle calibration, not predicted-H adaptation")
    cfg = StageAV2Config(**ck["stage_a_v2_config"])
    encoder = ProteinGeometryEncoder(
        latent_dim=cfg.latent_dim, depth=cfg.encoder_depth, num_heads=cfg.num_heads,
        num_rbf=cfg.num_rbf, dropout=cfg.dropout, max_residues=cfg.max_residues,
        orientation_features=cfg.orientation_features, identity_embed_scale=cfg.identity_embed_scale,
    ).to(device)
    encoder.load_state_dict(ck["encoder"], strict=True)
    oracle = StageBOracleDecoder(StageBOracleConfig.from_saved(ck["stage_b1a_config"])).to(device)
    oracle.load_state_dict(ck["oracle"], strict=True)
    if not oracle.h_normalizer.fitted_.item():
        raise ValueError("B1a normalizer must be fitted before autonomous B1b")
    validate_representation_contract(
        contract, encoder_state=encoder.state_dict(), encoder_config=ck["stage_a_v2_config"],
        normalizer_state=oracle.h_normalizer.state_dict(), decoder_state=oracle.decoder.state_dict(),
        decoder_config=ck["stage_b1a_config"],
    )
    return freeze_module(encoder), freeze_module(oracle), ck


def validate_oracle_gate(path, bundle, args, split_audit):
    """Require the same frozen readout, physical clock and forecast endpoints."""
    if path is None:
        if args.dry_run:
            return {"passed": False, "scope": "contract_smoke_only"}
        raise ValueError("formal autonomous training requires a held-out oracle gate")
    report = json.loads(Path(path).read_text())
    if report.get("schema_version") != 2 or report.get("passed") is not True:
        raise ValueError("oracle gate report must be schema-v2 and passed")
    if report.get("representation_id") != bundle["representation_contract"]["representation_id"]:
        raise ValueError("oracle gate report uses another representation")
    if report.get("scope") != "heldout_protein":
        raise ValueError("oracle gate must use heldout_protein evaluation")
    protocol = report.get("protocol", {})
    for key in ("history_frames", "frame_stride"):
        if protocol.get(key) != getattr(args, key):
            raise ValueError(f"oracle gate protocol mismatch: {key}")
    raw_dt = protocol.get("frame_interval_ps")
    if not isinstance(raw_dt, (int, float)) or not math.isfinite(raw_dt) or raw_dt <= 0:
        raise ValueError("oracle gate protocol requires a positive frame_interval_ps")
    if not math.isclose(raw_dt*args.frame_stride, args.step_ps, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("oracle gate physical step mismatch: frame_interval_ps * frame_stride != step_ps")
    horizons = protocol.get("horizons_frames", [])
    endpoints = {args.horizon_steps*args.frame_stride,
                 (args.validation_horizon_steps or args.horizon_steps)*args.frame_stride}
    for horizon in sorted(endpoints):
        if horizon not in horizons:
            raise ValueError(f"oracle gate does not cover requested horizon: {horizon} raw frames")
        if report.get("by_horizon", {}).get(str(horizon), {}).get("passed") is not True:
            raise ValueError(f"oracle gate did not pass requested horizon: {horizon} raw frames")
    if not split_audit:
        raise ValueError("formal oracle gate requires explicit train/validation manifests")
    for key in ("train_manifest_sha256", "val_manifest_sha256"):
        if report.get(key) != split_audit[key]:
            raise ValueError(f"oracle gate manifest mismatch: {key}")
    return report


@torch.no_grad()
def encode_frames(encoder, coords, rmask, aatype, ridx, cid, normalizer):
    B, T, L = coords.shape[:3]
    H = encoder(
        coords.reshape(B*T, 1, L, 4, 3), rmask.reshape(B*T, 1, L),
        aatype[:, None].expand(B, T, L).reshape(B*T, L),
        ridx[:, None].expand(B, T, L).reshape(B*T, L),
        cid[:, None].expand(B, T, L).reshape(B*T, L),
        token_mask=rmask.reshape(B*T, 1, L),
    )[:, 0].reshape(B, T, L, -1)
    return normalizer(H).detach()


def split_contiguous_batch(batch, history_frames, horizon_steps, step_ps):
    paired = split_contiguous_window(batch, history_frames, horizon_steps, step_ps=step_ps)
    # Fixed topology is part of the backbone-only conditioning contract. A
    # changing future mask cannot leak into the forecast as a hidden condition.
    mask = batch["residue_mask"][:, history_frames-1].bool()
    if not torch.equal(batch["residue_mask"].bool(), mask[:, None].expand_as(batch["residue_mask"])):
        raise ValueError("residue validity must be constant across the continuous clip")
    if not mask.any(dim=1).all():
        raise ValueError("each sample needs at least one valid residue")
    return paired["context_coords"], paired["future_coords"], mask, paired["context_dt_ps"]


def autonomous_samples(model, H_history, dt_history, residue_mask, temperature,
                       n_steps, num_samples, *, generator=None, training=True):
    """History-only API: future coordinates, H and labels cannot enter."""
    M = num_samples
    B, K, L, d = H_history.shape
    repeat = lambda x: x.unsqueeze(0).expand(M, *x.shape).reshape(M*B, *x.shape[1:])
    hist, dt, mask = repeat(H_history), repeat(dt_history), repeat(residue_mask)
    temp = None
    if temperature is not None:
        t = torch.as_tensor(temperature, device=H_history.device, dtype=H_history.dtype)
        temp = repeat(t.expand(B) if t.ndim == 0 else t.reshape(B))
    H, S = model.rollout(
        hist[:, -1], mask, temp, n_steps=n_steps, H_hist=hist, dt_hist=dt,
        seq_feature=hist[:, -1], generator=generator, detach=not training, return_bath=True,
    )
    H = H.reshape(M, B, n_steps+1, L, d)
    if S is not None:
        S = S.reshape(M, B, n_steps+1, L, model.cfg.bath_dim)
    return H, S


def normalized_path_energy_score(samples, target, mask):
    M, B, T, L, d = samples.shape
    if target.shape != (B, T, L, d):
        raise ValueError("forecast/target path shapes differ")
    scores = []
    for b in range(B):
        smp = samples[:, b, :, mask[b]].reshape(M, -1)
        tgt = target[b, :, mask[b]].reshape(-1).detach()
        scores.append(energy_score(smp, tgt)/math.sqrt(max(tgt.numel(), 1)))
    return torch.stack(scores).mean()


def decode_paths(oracle, paths, batch, mask, history_frames, H_anchor):
    """Inverse-normalized autonomous H through the frozen raw-H oracle."""
    M, B, T, L, d = paths.shape
    anchor = batch["coords"][:, history_frames-1]
    R, t = RigidBodyFrame.from_backbone(anchor[..., 0, :], anchor[..., 1, :], anchor[..., 2, :])
    repeat = lambda x: x[None, :, None].expand(M, B, T, *x.shape[1:]).reshape(M*B*T, *x.shape[1:])
    out = oracle.decode_structure(
        oracle.h_normalizer.inverse(paths).reshape(M*B*T, L, d),
        repeat(batch["aatype"]), repeat(R), repeat(t), repeat(mask),
    )
    predicted = out["backbone_coords"].reshape(M, B, T, L, 4, 3)
    # Fit once using observed H only; never align a predicted future to a true
    # future frame. The fit is a detached nuisance pose, not a training label.
    with torch.no_grad():
        decoded_anchor = oracle.decode_structure(
            oracle.h_normalizer.inverse(H_anchor), batch["aatype"], R, t, mask,
        )["CA"]
        pose_R, pose_t = fit_history_pose(decoded_anchor, anchor[:, :, 1], mask)
    return torch.einsum("mbtlaj,bjk->mbtlak", predicted, pose_R)+pose_t[None, :, None, None, None]


def physical_path_features(backbone, mask, ridx, cid):
    """Fixed physical scales: distance/10 A, unit contact and torsion sin/cos."""
    M, B, T, L = backbone.shape[:4]
    features = []
    for b in range(B):
        idx = torch.nonzero(mask[b], as_tuple=False).flatten()
        ca = backbone[:, b, :, idx, 1, :]
        pairs = torch.triu_indices(len(idx), len(idx), offset=1, device=ca.device)
        if pairs.shape[1] > 128:
            selection = torch.linspace(0, pairs.shape[1]-1, 128, device=ca.device).long()
            pairs = pairs[:, selection]
        parts = []
        if pairs.shape[1]:
            dist = (ca[..., pairs[0], :]-ca[..., pairs[1], :]).square().sum(-1).clamp_min(1e-12).sqrt()
            parts = [dist/10.0, torch.sigmoid(8.0-dist)]
        tor, valid = backbone_torsion_targets(
            backbone[:, b].reshape(M*T, L, 4, 3), mask[b].expand(M*T, L),
            ridx[b].expand(M*T, L), cid[b].expand(M*T, L),
        )
        tor = tor[..., :3].reshape(M, T, L, 3)[..., valid[0, :, :3]]
        parts += [torch.sin(tor), torch.cos(tor)]
        features.append(torch.cat(parts, -1).reshape(M, -1))
    return features


def physical_energy_score(pred, target, batch, mask):
    smp = physical_path_features(pred, mask, batch["residue_index"], batch["chain_id"])
    tgt = physical_path_features(target, mask, batch["residue_index"], batch["chain_id"])
    return torch.stack([
        energy_score(s, t[0].detach())/math.sqrt(max(t.shape[-1], 1))
        for s, t in zip(smp, tgt)
    ]).mean()


def displacement_point_loss(pred, target, anchor, mask):
    """Optional ensemble-mean point forecast in the real-anchor gauge."""
    from evaluation.geometry import anchor_align_rotation
    pred_ca, true_ca = pred.mean(0)[..., 1, :], target[..., 1, :]
    scores = []
    for b in range(pred_ca.shape[0]):
        for n in range(pred_ca.shape[1]):
            R, shift = anchor_align_rotation(anchor[b, :, 1], true_ca[b, n].detach(), mask[b])
            error = (pred_ca[b, n]@R.detach()+shift.detach()-true_ca[b, n])[mask[b]]
            scores.append(error.square().sum(-1).mean().clamp_min(1e-12).sqrt())
    return torch.stack(scores).mean()


def forecast_objective(model, encoder, oracle, batch, args, *, generator=None, training=True):
    K, N = args.history_frames, args.horizon_steps
    history, targets, mask, dt = split_contiguous_batch(batch, K, N, args.step_ps)
    aat, ridx, cid = (batch[k] for k in ("aatype", "residue_index", "chain_id"))
    H_hist = encode_frames(
        encoder, history, batch["residue_mask"][:, :K], aat, ridx, cid, oracle.h_normalizer,
    )
    # Generate the entire forecast before any true future encoding is computed.
    H_pred, S_pred = autonomous_samples(
        model, H_hist, dt, mask, batch.get("temperature_k"), N, args.es_num_samples,
        generator=generator, training=training,
    )
    H_target = encode_frames(
        encoder, targets, batch["residue_mask"][:, K-1:], aat, ridx, cid, oracle.h_normalizer,
    )
    end = min(N, args.es_horizon_steps or N)
    es = normalized_path_energy_score(H_pred[:, :, 1:end+1], H_target[:, 1:end+1], mask)
    physical, point = es.new_zeros(()), es.new_zeros(())
    pred_bb = None
    if args.physical_es_weight > 0 or args.rmsd_weight > 0:
        pred_bb = decode_paths(oracle, H_pred[:, :, 1:], batch, mask, K, H_hist[:, -1])
    if args.physical_es_weight > 0:
        # The physical label is the observed future geometry. True future H
        # never enters a decoder on either side of the training objective.
        physical = physical_energy_score(pred_bb, targets[:, 1:].unsqueeze(0), batch, mask)
    if args.rmsd_weight > 0:
        end_point = min(N, args.rmsd_horizon_steps or N)
        point = displacement_point_loss(pred_bb[:, :, :end_point], targets[:, 1:end_point+1], history[:, -1], mask)
    total = args.es_weight*es+args.physical_es_weight*physical+args.rmsd_weight*point
    logs = {"loss/total": float(total.detach()), "loss/latent_es": float(es.detach()),
            "loss/physical_es": float(physical.detach()), "loss/mean_displacement": float(point.detach())}
    return total, logs, {"H_pred": H_pred, "S_pred": S_pred, "H_target": H_target,
                         "H_anchor": H_hist[:, -1], "predicted_backbone": pred_bb}


def _loader(args, manifest, training):
    horizon = args.horizon_steps if training else (args.validation_horizon_steps or args.horizon_steps)
    config = DataConfig(
        manifest_path=manifest, training_mode="clip", num_frames=args.history_frames+horizon,
        frame_stride=args.frame_stride, max_residues=args.max_residues,
        synthetic_size=args.synthetic_size, synthetic_residues=args.synthetic_residues,
    )
    dataset = build_dataset(config)
    if manifest is None:
        boundary = max(1, len(dataset)//2)
        dataset = Subset(dataset, range(0, boundary) if training else range(boundary, len(dataset)))
    if len(dataset) == 0:
        raise ValueError("empty training/validation dataset")
    validation_sampling = None
    if not training:
        total = len(dataset)
        budget = args.validation_batches*args.batch_size
        indices = torch.linspace(0, total-1, min(total, budget)).long().tolist()
        # The manifest is grouped by trajectories/proteins. Spread a fixed
        # development budget across its full window range instead of consuming
        # only the first protein. These remain correlated trajectory windows.
        validation_sampling = {
            "method": "global_linspace_windows_v1", "available_windows": total,
            "selected_windows": len(indices), "window_budget": budget,
            "indices_in_validation_dataset": indices,
            "unit": "trajectory_windows_not_independent_proteins",
        }
        dataset = Subset(dataset, indices)
    # A private RNG prevents iterator/worker seeds from perturbing the dynamics
    # noise stream, including when resuming at a mid-epoch cursor.
    loader_rng = torch.Generator().manual_seed(args.seed + (0 if training else 1))
    sampler = EpochPermutation(dataset, args.seed) if training else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, collate_fn=collate_clips,
                        drop_last=False, generator=loader_rng)
    loader.validation_sampling = validation_sampling
    return loader


def validate(model, encoder, oracle, loader, args, device):
    val_args = argparse.Namespace(**vars(args))
    val_args.horizon_steps = args.validation_horizon_steps or args.horizon_steps
    val_args.es_horizon_steps = None  # validate the complete requested path
    val_args.rmsd_horizon_steps = None
    was_training = model.training
    model.eval()
    records = []
    gen = torch.Generator(device=device).manual_seed(args.seed+1000003)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        for index, batch in enumerate(loader):
            if index >= args.validation_batches:
                break
            batch = _move_batch(batch, device)
            loss, logs, forecast = forecast_objective(
                model, encoder, oracle, batch, val_args, generator=gen, training=False,
            )
            if not torch.isfinite(forecast["H_pred"]).all() or not torch.isfinite(loss):
                raise FloatingPointError("nonfinite autonomous validation forecast or score")
            logs["validation/max_abs_H"] = float(forecast["H_pred"].abs().max())
            logs["validation/fraction_abs_H_above_8std"] = float((forecast["H_pred"].abs() > 8).float().mean())
            mask = batch["residue_mask"][:, args.history_frames-1]
            backbone = forecast["predicted_backbone"]
            if backbone is None:
                backbone = decode_paths(oracle, forecast["H_pred"][:, :, 1:], batch, mask,
                                        args.history_frames, forecast["H_anchor"])
            if not torch.isfinite(backbone).all():
                raise FloatingPointError("nonfinite decoded autonomous validation structure")
            logs["validation/mean_displacement_angstrom"] = float(displacement_point_loss(
                backbone, batch["coords"][:, args.history_frames:],
                batch["coords"][:, args.history_frames-1], mask,
            ))
            adjacent = ((batch["chain_id"][:, 1:] == batch["chain_id"][:, :-1])
                        & (batch["residue_index"][:, 1:]-batch["residue_index"][:, :-1] == 1)
                        & mask[:, 1:] & mask[:, :-1])
            ca = backbone[..., 1, :]
            bonds = (ca[..., 1:, :]-ca[..., :-1, :]).norm(dim=-1)
            weights = adjacent[None, :, None].expand_as(bonds)
            violations = ((bonds < 3.0) | (bonds > 4.5)) & weights
            logs["validation/ca_bond_violation_fraction"] = float(violations.sum()/weights.sum().clamp_min(1))
            records.append(logs)
    model.train(was_training)
    return {key: sum(r[key] for r in records)/len(records) for key in records[0]}


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


def checkpoint_payload(model, optimizer, step, bundle, args, validation, best_score, bad_validations):
    return {
        **{k: bundle[k] for k in ("encoder", "stage_a_v2_config", "oracle", "stage_b1a_config", "representation_contract")},
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
        "stage_b1b_config": asdict(model.cfg), "train_args": vars(args),
        "training_contract": TRAINING_CONTRACT, "validation": validation, "best_score": best_score,
        "bad_validations": bad_validations, "rng_state": capture_rng(),
        "run_scope": "contract_smoke_only" if args.dry_run else "oracle_gate_passed",
        "oracle_gate": getattr(args, "oracle_gate_record", None),
        "manifest_split": getattr(args, "manifest_split_record", None),
        "validation_sampling": getattr(args, "validation_sampling", None),
        "physical_feature_schema": {"version": 1, "distance_scale_angstrom": 10.0,
                                    "contact_cutoff_angstrom": 8.0, "contact_width_angstrom": 1.0,
                                    "max_pairs": 128, "torsion": "signed_sin_cos"},
    }


def resume_training(path, model, optimizer, bundle, args, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("training_contract") != TRAINING_CONTRACT:
        raise ValueError("resume checkpoint must use autonomous_history_only_v2")
    assert_same_representation(ck["representation_contract"], bundle["representation_contract"])
    if ck["stage_b1b_config"] != asdict(model.cfg):
        raise ValueError("resume dynamics configuration mismatch")
    if ck.get("manifest_split") != getattr(args, "manifest_split_record", None):
        raise ValueError("resume manifest split or content mismatch")
    if ck.get("validation_sampling") != getattr(args, "validation_sampling", None):
        raise ValueError("resume validation sampling mismatch")
    for key in ("history_frames", "horizon_steps", "frame_stride", "es_num_samples",
                "es_weight", "physical_es_weight", "rmsd_weight", "batch_size", "steps",
                "warmup_steps", "es_horizon_steps", "rmsd_horizon_steps", "seed",
                "max_residues", "synthetic_size", "synthetic_residues", "lr",
                "weight_decay", "grad_clip", "validation_horizon_steps", "validation_every",
                "validation_batches", "patience", "dry_run"):
        if ck["train_args"].get(key) != getattr(args, key):
            raise ValueError(f"resume training contract mismatch: {key}")
    model.load_state_dict(ck["model"], strict=True)
    optimizer.load_state_dict(ck["optimizer"])
    restore_rng(ck["rng_state"])
    return ck


def main(argv=None):
    args = parse_args(argv)
    _set_reproducibility(args.seed, True)
    device = _device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    encoder, oracle, bundle = load_frozen_representation(args.b1a_ckpt, device)
    split_audit = None
    if args.manifest and args.val_manifest:
        split_audit = validate_manifest_split(args.manifest, args.val_manifest)
    args.oracle_gate_record = validate_oracle_gate(args.oracle_gate_report, bundle, args, split_audit)
    args.manifest_split_record = split_audit
    if args.dry_run:
        print("B1b contract smoke only: this run does not establish a passed scientific H gate", flush=True)
    latent_dim = bundle["stage_a_v2_config"]["latent_dim"]
    if args.latent_dim is not None and args.latent_dim != latent_dim:
        raise ValueError("latent-dim differs from the frozen B1a encoder")
    cfg = StageB1bConfig(
        latent_dim=latent_dim, bath_dim=args.bath_dim, energy_depth=args.energy_depth,
        energy_heads=args.energy_heads, energy_hidden=args.energy_hidden,
        history_hidden=args.history_hidden, step_ps=args.step_ps, coupling_scale=args.coupling_scale,
        diffusion_scale=args.diffusion_scale, c_init_std=args.c_init_std,
        gamma_scale=args.gamma_scale, gamma_floor=args.gamma_floor, max_gamma_step=args.max_gamma_step,
    )
    model = StageB1b(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = _loader(args, args.manifest, True)
    val_loader = _loader(args, args.val_manifest, False) if args.val_manifest or not args.manifest else None
    args.validation_sampling = getattr(val_loader, "validation_sampling", None)
    step, best, bad, validation = 0, float("inf"), 0, {}
    if args.resume:
        ck = resume_training(args.resume, model, optimizer, bundle, args, device)
        step, best, bad, validation = ck["step"], ck["best_score"], ck["bad_validations"], ck["validation"]
    (out/"config.json").write_text(json.dumps({
        "stage_b1b_config": asdict(cfg), "train_args": vars(args),
        "representation_contract": bundle["representation_contract"], "training_contract": TRAINING_CONTRACT,
        "manifest_split": split_audit,
        "validation_sampling": args.validation_sampling,
    }, indent=2))
    model.train()
    stop = False
    # Per-epoch deterministic permutations + cursor + RNG restore permit resume.
    while step < args.steps and not stop:
        first_batch = step % len(train_loader)
        train_loader.sampler.epoch = step // len(train_loader)
        for index, batch in enumerate(train_loader):
            if index < first_batch:
                continue
            if step >= args.steps:
                break
            for group in optimizer.param_groups:
                group["lr"] = args.lr*_lr_scale(step, args.steps, args.warmup_steps)
            optimizer.zero_grad(set_to_none=True)
            with sdpa_kernel([SDPBackend.MATH]):
                loss, logs, _ = forecast_objective(
                    model, encoder, oracle, _move_batch(batch, device), args, training=True,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite autonomous training score")
                loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                print(f"step {step} "+" ".join(f"{k}={v:.6g}" for k,v in logs.items()), flush=True)
            improved = False
            if val_loader is not None and (step % args.validation_every == 0 or step == args.steps):
                validation = validate(model, encoder, oracle, val_loader, args, device)
                score = validation["loss/total"]
                improved, best = score < best, min(best, score)
                bad = 0 if improved else bad+1
                print("validation "+json.dumps(validation, sort_keys=True), flush=True)
                stop = args.patience > 0 and bad >= args.patience
            if improved or step % args.save_every == 0 or step == args.steps or stop or args.dry_run:
                payload = checkpoint_payload(model, optimizer, step, bundle, args, validation, best, bad)
                torch.save(payload, out/"checkpoint_latest.pt")
                if improved:
                    torch.save(payload, out/"checkpoint_best.pt")
            if args.dry_run and step >= min(args.steps, 2):
                return
            if stop:
                break


if __name__ == "__main__":
    main()
