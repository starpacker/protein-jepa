"""Stage A (v4 static/dynamic factorization) training entry.

Reuses the calibration losses/loaders from ``train_lasp_calibration`` but swaps
the encoder for the static/dynamic split and unlocks H_dyn magnitude, so the
decoder can reconstruct real conformational displacement.

The decisive Stage-A gate number is ``decode(GT)`` aligned CA-RMSD: with the
magnitude-locked encoder it is 1.6-4.9 A (Phase 0 audit); after retraining with
a magnitude-carrying H_dyn it should drop.

Usage:
    PYTHONPATH=/tmp/lasp_stability_deps:/liuyutian/protein_jepa_frame \
    python -m protein_jepa_frame.training.train_stage_a_v4 \
        --init-from artifacts/lasp_full_v1/representation/checkpoint_best.pt \
        --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
        --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
        --output artifacts/v4_stage_a/smoke \
        --static-dynamic --steps 200 --max-train-samples 128 ...
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import validate_manifest_split
from protein_jepa_frame.training.train_utils import _move_batch, _set_reproducibility, _lr_scale
from protein_jepa_frame.training.train_stage_ensemble import load_frozen_stack, _extract_backbone_frames
from evaluation.ensemble_metrics_ext import _align_to
from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder, BoundedEdgeCADecoder
from protein_jepa_frame.models.latent_delta_decoder import LatentDeltaCADecoder
from protein_jepa_frame.models.ca_geometry import project_ca_bonds
from protein_jepa_frame.models.static_dynamic_v4 import (
    StaticEncoder, DynamicEncoder, ConditionalStructureDecoder,
)
from protein_jepa_frame.training.train_lasp_calibration import (
    calibration_terms, objective, loaders, validate, selection_score, masked_mean,
)
from protein_jepa_frame.training.lasp_run_state import step_batch, rng_state, restore_rng, budget_subset


def unlock_magnitude(encoder: nn.Module) -> nn.Module:
    """Replace the magnitude-locking LayerNorm with Identity, in place.

    The Phase 0 audit found ``LayerNorm(elementwise_affine=False)`` pins every
    residue's H norm to sqrt(D) (11.3137), freezing displacement magnitude out of
    the representation.  LayerNorm always normalizes (divides by std) regardless
    of its affine flag, so unlocking magnitude means removing the normalization
    entirely, not just enabling affine.
    """
    encoder.norm = nn.Identity()
    return encoder


class StaticDynamicCalibration(torch.nn.Module):
    """Static/dynamic factorization around the loaded encoder/decoder stack."""

    def __init__(self, args, device):
        super().__init__()
        self.encoder, self.decoder, dim = load_frozen_stack(
            args.stage_a_ckpt, args.b1a_ckpt, device
        )
        self.source = torch.load(args.init_from, map_location="cpu", weights_only=False)
        if self.source.get("encoder") is not None:
            self.encoder.load_state_dict(self.source["encoder"], strict=True)

        # Static branch (time-invariant identity + anchor fold).
        self.static_encoder = StaticEncoder(dim).to(device)

        # Dynamic branch: a DynamicEncoder whose base geometry encoder is seeded
        # with the loaded weights, but with the magnitude-locking norm removed.
        # Reuse the loaded encoder's exact geometry config (orientation + relpos).
        base = self.encoder
        self.dynamic_encoder = DynamicEncoder(
            dim,
            orientation_features=getattr(base, "orientation_features", False),
            relative_position_features=getattr(base, "relative_position_features", False),
            anchor_conditioned=getattr(base, "anchor_conditioned", False),
        )
        # Transfer the loaded ProteinGeometryEncoder weights into the wrapper's
        # nested `.encoder`, mapping the top-level keys to the `encoder.` prefix.
        loaded_sd = self.encoder.state_dict()
        dyn_sd = self.dynamic_encoder.state_dict()
        mapped = {}
        for k, v in loaded_sd.items():
            new_k = "encoder." + k
            if new_k in dyn_sd and dyn_sd[new_k].shape == v.shape:
                mapped[new_k] = v
        self.dynamic_encoder.load_state_dict(mapped, strict=False)
        self.dynamic_encoder = self.dynamic_encoder.to(device)

        # Decoder: reuse the delta-CA decoder, wrapped in the conditional contract.
        decoder_class = {
            "geometric_ca": GeometricDeltaCADecoder,
            "bounded_edge_ca": BoundedEdgeCADecoder,
            "delta_ca": LatentDeltaCADecoder,
        }[getattr(args, "decoder_mode", "geometric_ca")]
        self.decoder = decoder_class().to(device)
        if self.source.get("delta_ca_decoder") is not None:
            try:
                self.decoder.load_state_dict(self.source["delta_ca_decoder"], strict=True)
            except RuntimeError:
                pass  # shape/config mismatch: fall back to fresh decoder
        self.conditional_decoder = ConditionalStructureDecoder(self.decoder, dim)

        self.joint = args.joint
        self.bond_projection_steps = getattr(args, "bond_projection_steps", 0)

        for p in self.dynamic_encoder.parameters():
            p.requires_grad_(args.joint)
        for p in self.static_encoder.parameters():
            p.requires_grad_(True)
        for p in self.decoder.parameters():
            p.requires_grad_(True)
        # The conditional wrapper adds static/dynamic projection layers.
        for p in self.conditional_decoder.parameters():
            p.requires_grad_(True)

    def forward(self, batch, controls=False):
        coords = batch["coords"]
        b, t, l = coords.shape[:3]
        mask = batch["residue_mask"].bool().all(1)
        aa, ri, ci = [batch[k] for k in ("aatype", "residue_index", "chain_id")]
        rm = mask[:, None].expand(b, t, l).reshape(b * t, l)
        expand = lambda x: x[:, None].expand(b, t, l).reshape(b * t, l)

        # H_static from the anchor frame only (frame 0).
        h_static = self.static_encoder(coords[:, 0], mask, aa, ri, ci)  # [b,l,D]

        # H_dyn from all frames, conditioned on H_static.  The base geometry
        # encoder works per-frame on [B,1,L,4,3]; flatten the time dim exactly
        # like the original Calibration.encode.
        expand = lambda x: x[:, None].expand(b, t, l).reshape(b * t, l)
        rm = mask[:, None].expand(b, t, l).reshape(b * t, l)
        h_static_expanded = h_static[:, None].expand(b, t, l, -1).reshape(b * t, l, -1)
        with torch.set_grad_enabled(self.joint):
            h_dyn = self.dynamic_encoder(
                coords.reshape(b * t, 1, l, 4, 3), rm[:, None],
                expand(aa), expand(ri), expand(ci),
                h_static=h_static_expanded, token_mask=rm[:, None],
            )[:, 0].reshape(b, t, l, -1)  # [b,t,l,D]

        # R [b,l,3,3], anchor [b,l,3] (from_backbone preserves leading dims).
        R, anchor = _extract_backbone_frames(coords[:, 0])

        def decode(h_t):
            # h_t [b, l, D] -> CA via D(H_static, H_dyn,t).
            ca = self.conditional_decoder(h_t, h_static, aa, ri, ci, R, anchor, mask)
            if self.bond_projection_steps:
                ca = project_ca_bonds(ca, anchor, mask, ri, ci, self.bond_projection_steps)
            return ca

        # Decode each frame: D(H_static, H_dyn,t).
        preds = [decode(h_dyn[:, t]) for t in range(t)]
        pred = torch.stack(preds, dim=1)  # [b, t, l, 3]

        with torch.no_grad():
            target = torch.stack([_align_to(coords[k, :, :, 1], coords[k, 0, :, 1], mask[k]) for k in range(b)])

        terms = calibration_terms(pred, target, mask, ri, ci)
        terms["teacher"] = terms["teacher_delta"] = pred.new_zeros(())
        if not controls:
            conf_loss, conf_rho = conformation_sensitivity_terms(h_dyn, target, mask)
            terms["conf_sensitivity"] = conf_loss
            terms["conf_rho"] = conf_rho

        if controls:
            # Identity/zero/shuffled H_dyn controls: does the decoder use H_dyn?
            fixed_h = h_dyn[:, :1].expand_as(h_dyn)
            fixed = torch.stack([decode(fixed_h[:, t]) for t in range(t)], dim=1)
            zero_h = torch.zeros_like(h_dyn)
            zero = torch.stack([decode(zero_h[:, t]) for t in range(t)], dim=1)
            terms["ca_rmsd"] = _position(pred, target, mask)
            terms["fixed_ca_rmsd"] = _position(fixed, target, mask)
            terms["zero_ca_rmsd"] = _position(zero, target, mask)

        if not all(torch.isfinite(v).all() for v in terms.values()):
            raise RuntimeError("nonfinite stage-A v4 terms")
        return terms


def _position(pred, target, mask):
    # pred/target [b, t, l, 3]; mask [b, l] -> [b, 1, l, 1]
    m = mask.unsqueeze(1).unsqueeze(-1).expand_as(pred)
    return (pred[m] - target[m]).square().mean().sqrt()


def conformation_sensitivity_terms(h_dyn, target, mask):
    """A2-B metric group: couple H_dyn displacement to physical conformational
    displacement so H_dyn cannot collapse to ~0 or encode only identity.

    h_dyn [b,t,l,D], target [b,t,l,3] (CA aligned to the anchor frame).  Returns
    (loss, rho) where rho is a diagnostic Spearman-style rank correlation between
    latent and physical per-frame displacement from the anchor.
    """
    b, t = h_dyn.shape[:2]
    if t < 2:
        return h_dyn.new_zeros(()), h_dyn.new_zeros(())
    m = mask.unsqueeze(-1).float()  # [b,l,1]
    denom = m.unsqueeze(1).sum(2).clamp_min(1.0)  # [b,t,1]
    # Per-frame mean latent vector [b,t,D].
    hmean = (h_dyn * m.unsqueeze(1)).sum(2) / denom
    # Latent displacement of each frame from the anchor frame 0 [b,t-1].
    dlat = (hmean[:, 1:] - hmean[:, :1]).square().sum(-1).sqrt()
    # Physical per-residue CA displacement from the anchor, masked [b,t-1].
    dphys = (target[:, 1:] - target[:, :1]).square().sum(-1) * mask.unsqueeze(1)
    dphys = dphys.sum(-1).sqrt() / mask.sum(-1).clamp_min(1.0)
    # Normalize both to unit scale (per-batch) to make the contrastive objective
    # scale-free and immune to the latent/physical magnitude mismatch.
    dlat = dlat / (dlat.std(-1, keepdim=True) + 1e-6)
    dphys = dphys / (dphys.std(-1, keepdim=True) + 1e-6)
    # Margin contrastive loss over (anchor, near, far) within each trajectory:
    # far frames must have larger latent displacement than near frames.
    d = dlat.unsqueeze(-1) - dlat.unsqueeze(-2)  # [b, t-1, t-1] sign(i vs j)
    p = dphys.unsqueeze(-1) - dphys.unsqueeze(-2)  # [b, t-1, t-1]
    margin = 0.1
    loss = torch.relu(margin - d * p.sign()).mean()
    # Diagnostic: Pearson correlation between latent and physical displacement.
    dc = dlat - dlat.mean(-1, keepdim=True)
    pc = dphys - dphys.mean(-1, keepdim=True)
    rho = (dc * pc).sum(-1) / ((dc.norm(dim=-1) * pc.norm(dim=-1)).clamp_min(1e-6))
    return loss, rho.mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stateless-data", action="store_true")
    p.add_argument("--cosine", action="store_true")
    p.add_argument("--init-from", default="artifacts/lasp_full_v1/representation/checkpoint_best.pt")
    p.add_argument("--stage-a-ckpt", default="artifacts/stage_a_disp_relpos/checkpoint_latest.pt")
    p.add_argument("--b1a-ckpt", default="artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt")
    p.add_argument("--manifest", default="artifacts/mdcath_manifest_full/train.jsonl")
    p.add_argument("--val-manifest", default="artifacts/mdcath_manifest_full/val.jsonl")
    p.add_argument("--decoder-mode", default="geometric_ca")
    p.add_argument("--joint", action="store_true")
    p.add_argument("--bond-projection-steps", type=int, default=32)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-residues", type=int, default=96)
    p.add_argument("--max-train-samples", type=int, default=1024)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--distance-weight", type=float, default=0.2)
    p.add_argument("--delta-weight", type=float, default=0.5)
    p.add_argument("--teacher-weight", type=float, default=1.0)
    p.add_argument("--teacher-delta-weight", type=float, default=0.1)
    p.add_argument("--conf-weight", type=float, default=1.0)
    p.add_argument("--change-weight", type=float, default=1.0)
    p.add_argument("--bond-weight", type=float, default=1.0)
    p.add_argument("--angle-weight", type=float, default=2.0)
    p.add_argument("--clash-weight", type=float, default=50.0)
    p.add_argument("--selection-bond-limit", type=float, default=float("inf"))
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    audit = validate_manifest_split(a.manifest, a.val_manifest)
    torch.set_num_threads(4)
    _set_reproducibility(a.seed, False)
    dev = torch.device(a.device)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.jsonl").exists() and not a.resume:
        raise FileExistsError("use a fresh output directory")

    model = StaticDynamicCalibration(a, dev).to(dev).eval()
    train = loaders(a, a.manifest, True)
    val = loaders(a, a.val_manifest, False)

    # Optimizer: static + dynamic encoders (scaled) + decoder + conditional
    # projections (the wrapper's new static/dynamic_proj layers, NOT the wrapped
    # decoder which is already in the decoder group).
    enc_params = list(model.dynamic_encoder.parameters()) + list(model.static_encoder.parameters())
    cond_proj_params = [p for n, p in model.conditional_decoder.named_parameters()
                        if n.startswith(("static_proj", "dynamic_proj"))]
    groups = [
        dict(params=enc_params, lr=a.lr * a.encoder_lr_scale, base_lr=a.lr * a.encoder_lr_scale),
        dict(params=list(model.decoder.parameters()) + cond_proj_params,
             lr=a.lr, base_lr=a.lr),
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    params = [p for p in model.parameters() if p.requires_grad]

    if not a.resume:
        (out / "run.json").write_text(json.dumps(dict(args=vars(a), split_audit=audit), indent=2))

    def record(row):
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def save(tag, step, metrics):
        ck = {
            "checkpoint_kind": "v4_stage_a",
            "dynamic_encoder": model.dynamic_encoder.state_dict(),
            "static_encoder": model.static_encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "conditional_decoder": model.conditional_decoder.state_dict(),
            "step": step, "args": vars(a), "validation": metrics,
            "optimizer": opt.state_dict(), "rng_state": rng_state(),
        }
        torch.save(ck, out / f"checkpoint_{tag}.tmp")
        (out / f"checkpoint_{tag}.tmp").replace(out / f"checkpoint_{tag}.pt")

    start_step = 0
    best = float("inf")
    if a.resume:
        ck = torch.load(out / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
        model.dynamic_encoder.load_state_dict(ck["dynamic_encoder"], strict=True)
        model.static_encoder.load_state_dict(ck["static_encoder"], strict=True)
        model.decoder.load_state_dict(ck["decoder"], strict=True)
        model.conditional_decoder.load_state_dict(ck["conditional_decoder"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        best = ck.get("best", float("inf"))
        start_step = ck["step"]
        restore_rng(ck["rng_state"])
        record(dict(resumed_from=start_step))
    else:
        initial = validate(model, val, a, dev)
        record(dict(step=0, val=initial))
        save("initial", 0, initial)
        save("latest", 0, initial)

    iters = {h: iter(dl) for h, dl in train.items()}
    cycle = list(train)
    start = time.time()
    for step in range(start_step + 1, a.steps + 1):
        h = cycle[(step - 1) % len(cycle)]
        if a.stateless_data:
            batch = step_batch(train[h], step, a.seed)
        else:
            try:
                batch = next(iters[h])
            except StopIteration:
                iters[h] = iter(train[h])
                batch = next(iters[h])
        opt.zero_grad(set_to_none=True)
        terms = model(_move_batch(batch, dev))
        loss = objective(terms, a) + a.conf_weight * terms["conf_sensitivity"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss step {step}")
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            raise RuntimeError(f"nonfinite gradient step {step}")
        scale = _lr_scale(step - 1, a.steps, a.warmup) if a.cosine else min(step / max(a.warmup, 1), 1.0)
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * scale
        opt.step()
        if step % 10 == 0 or step == 1:
            record(dict(step=step, stride=h, seconds=time.time() - start, loss=float(loss.detach()),
                        grad_norm=float(gn), train={k: float(v.detach()) for k, v in terms.items()}))
        if step % a.val_every == 0 or step == a.steps:
            metrics = validate(model, val, a, dev)
            record(dict(step=step, val=metrics))
            score = selection_score(metrics, a.selection_bond_limit)
            if score < best:
                best = score
                save("best", step, metrics)
            save("latest", step, metrics)
    print(f"done: {a.steps} updates", flush=True)


if __name__ == "__main__":
    main()
