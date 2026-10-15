#!/usr/bin/env python3
"""P5: Side-chain structure validation.

Validates the Stage C decoder's ability to place sidechains from backbone
+ predicted torsions.  Compares:
  1. Per-residue chi angle correlation (chi1, chi2)
  2. Atom37 RMSD (all-heavy-atom) on held-out set
  3. Clash count in predicted sidechains
  4. Per-residue breakdown by residue type
"""

from __future__ import annotations
import json, argparse, sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import PairedTrajectoryDataset, load_manifest
from protein_jepa_frame.models.stage_c.model import StageCModel
from protein_jepa_frame.models.stage_c.structure_decoder import DecoderConfig
from protein_jepa_frame.models.stage_c.direct_head import DirectHeadConfig
from protein_jepa_frame.models.stage_c.rollout import RolloutConfig
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.training.stage_c import _compute_target_torsions
from protein_jepa_frame.utils.structure_metrics import kabsch_align
from protein_jepa_frame import vendored as _vendored  # noqa: F401
from openfold.data.data_transforms import make_atom14_masks
from openfold.np import residue_constants
from openfold.utils.loss import between_residue_clash_loss
from openfold.utils.tensor_utils import batched_gather
from protein_jepa_frame.eval.stage_c_checkpoint import (
    split_stage_c_state,
    stage_c_configs_from_checkpoint,
    load_stage_b_for_stage_c,
)
from eval_sampling import stratified_indices


def load_model(ckpt_path, stage_b_ckpt, device):
    """Load StageCModel from checkpoint, mapping bare decoder keys."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    # Recover the decoder hyperparameters from the checkpoint.  In particular,
    # q_dim and the Q/COM translation bounds are part of the learned model's
    # semantics; constructing a default DecoderConfig here changes the model
    # being evaluated (and used to silently do so).
    sc, direct_cfg = stage_c_configs_from_checkpoint(ckpt)
    sb, mc, _ = load_stage_b_for_stage_c(ckpt, stage_b_ckpt, device)
    model = StageCModel(
        stage_b=sb, decoder_cfg=sc,
        direct_cfg=direct_cfg,
        rollout_cfg=RolloutConfig(dynamics_dim=sc.q_dim, context_dim=sc.context_dim),
    ).to(device)
    # Validate and load the decoder portion strictly.  A partial decoder is
    # not a valid side-chain evaluation checkpoint.
    decoder_state, _ = split_stage_c_state(
        model.decoder, model.direct_head, state
    )
    model.decoder.load_state_dict(decoder_state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_clashes(
    atom37,
    mask=None,
    atom37_mask=None,
    *,
    aatype=None,
    residue_index=None,
):
    """Count inter-residue steric clashes with OpenFold's official rule.

    The former all-pairs ``distance < 2A`` implementation counted normal
    covalent bonds as clashes.  OpenFold's ``between_residue_clash_loss`` uses
    residue/atom van-der-Waals radii and excludes peptide/disulfide exceptions.
    """
    if aatype is None:
        raise ValueError("aatype is required for the official clash metric")
    B, N = atom37.shape[:2]
    if residue_index is None:
        residue_index = torch.arange(N, device=atom37.device).unsqueeze(0).expand(B, -1)
    feat = make_atom14_masks({"aatype": aatype.to(torch.long)})
    atom14 = batched_gather(
        atom37,
        feat["residx_atom14_to_atom37"],
        dim=-2,
        no_batch_dims=2,
    )
    atom14_mask = batched_gather(
        atom37_mask.to(atom37.dtype),
        feat["residx_atom14_to_atom37"],
        dim=-1,
        no_batch_dims=2,
    ) * feat["atom14_atom_exists"]
    if mask is not None:
        atom14_mask = atom14_mask * mask.to(atom37.dtype).unsqueeze(-1)
    atom14_radius = residue_constants.van_der_waals_radius
    radius_by_atom37 = atom37.new_tensor([
        atom14_radius[name[0]] for name in residue_constants.atom_types
    ])
    atom14_radius = feat["atom14_atom_exists"] * radius_by_atom37[
        feat["residx_atom14_to_atom37"]
    ]
    result = between_residue_clash_loss(
        atom14_pred_positions=atom14,
        atom14_atom_exists=atom14_mask,
        atom14_atom_radius=atom14_radius,
        residue_index=residue_index.to(torch.long),
        overlap_tolerance_soft=0.0,
        overlap_tolerance_hard=0.0,
    )
    # OpenFold returns a per-atom mask; each inter-residue clash contributes
    # once for each endpoint, hence divide by two for a pair count.
    return result["per_atom_clash_mask"].float().sum(dim=(1, 2)) / 2.0


def chi_angle_error(pred, targ, mask=None, chi_idx=3):
    diff = pred[:, :, chi_idx] - targ[:, :, chi_idx]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    err = diff.abs()
    if mask is not None:
        err = err * mask.float()
        return err.sum() / mask.float().sum().clamp(min=1)
    return err.mean()


def atom37_rmsd(pred_a37, targ_a37, mask=None):
    B, N = pred_a37.shape[:2]
    pred_flat = pred_a37.reshape(B, N * 37, 3)
    targ_flat = targ_a37.reshape(B, N * 37, 3)
    pred_aligned, _ = kabsch_align(pred_flat, targ_flat)
    diff = (pred_aligned - targ_flat).pow(2).sum(dim=-1)
    if mask is not None:
        mask37 = mask.unsqueeze(-1).expand(-1, -1, 37).reshape(B, N * 37)
        diff = diff * mask37.float()
        M = mask37.float().sum(dim=1).clamp(min=1)
        return torch.sqrt(diff.sum(dim=1) / M)
    return torch.sqrt(diff.mean(dim=1))


def run_sidechain_eval(
    decoder_ckpt="artifacts/mdcath_stage_c_compact/checkpoint_latest.pt",
    stage_b_ckpt="artifacts/mdcath_stage_b_rollout/checkpoint_latest.pt",
    val_manifest="artifacts/mdcath_manifest_200dom/val.jsonl",
    output_dir="artifacts/sidechain_quality",
    n_val=100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(decoder_ckpt, stage_b_ckpt, device)
    print("Stage C model loaded with frozen Stage B encoder.")

    manifest = load_manifest(val_manifest)
    dcfg = DataConfig(
        batch_size=1, num_workers=0, num_frames=8, frame_stride=4,
        window_stride=4, max_residues=256, coordinate_scale=1.0,
        training_mode="paired_window", paired_horizons_frames=[32, 64, 96],
        paired_all_horizons=False, paired_history_mode="uniform",
    )
    dataset = PairedTrajectoryDataset(manifest, dcfg)
    eval_indices = stratified_indices(dataset, n_val, seed=42)
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_paired_windows, drop_last=False)

    all_records = []
    failed_samples = []

    for sample_i, batch in enumerate(loader):
        if sample_i >= n_val:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        try:
            context = model.encode_context(batch)

            # Predict atom37 from anchor structure (Q=0 oracle mode)
            q_oracle = model.extract_oracle_q(batch, context)
            pred_output = model.predict_oracle(context, q_oracle)
            pred_a37 = pred_output.get("atom37_coords")
            pred_torsions = pred_output.get("torsions")

            # The dataset has no ground-truth atom37.  Use the actual future
            # backbone to compute torsion targets; never create a target with
            # the same decoder being evaluated.
            target_backbone = batch["future_coords"][:, -1]
            targ_torsions = _compute_target_torsions(target_backbone)
            aatype = batch.get("future_aatype", batch.get("aatype"))
            if aatype.dim() == 1:
                aatype = aatype.unsqueeze(0).expand(pred_a37.shape[0], -1)
            residue_index = batch.get(
                "future_residue_index",
                batch.get("context_residue_index", batch.get("residue_index")),
            )
            if residue_index is None:
                residue_index = torch.arange(
                    aatype.shape[1], device=aatype.device
                ).unsqueeze(0).expand(aatype.shape[0], -1)
            if residue_index.dim() == 1:
                residue_index = residue_index.unsqueeze(0).expand(aatype.shape[0], -1)

            if pred_a37 is not None:
                residue_mask = context.get("residue_mask")
                # residue_mask is [B, T, N], take last frame for per-residue
                if residue_mask is not None and residue_mask.dim() == 3:
                    residue_mask = residue_mask[:, -1]

                # Atom37 RMSD against native sidechains is unavailable in this
                # backbone-only manifest; report NaN instead of a synthetic
                # decoder-vs-decoder number.
                a37_rmsd_val = torch.tensor(float("nan"), device=device)
                n_clashes = compute_clashes(
                    pred_a37, residue_mask, pred_output.get("atom37_mask"),
                    aatype=aatype,
                    residue_index=residue_index,
                ).mean().item()
                n_clashes_target = float("nan")

                chi1_err = chi_angle_error(pred_torsions, targ_torsions, residue_mask, chi_idx=3)
                chi2_err = chi_angle_error(pred_torsions, targ_torsions, residue_mask, chi_idx=4)
                phi_err = chi_angle_error(pred_torsions, targ_torsions, residue_mask, chi_idx=0)
                psi_err = chi_angle_error(pred_torsions, targ_torsions, residue_mask, chi_idx=1)

                # Native atom37 is unavailable in the backbone-only manifest;
                # per-residue atom RMSD is intentionally omitted.
                per_aa = {}

                rec = {
                    "protein_id": str(batch.get("protein_id", f"s{sample_i}")),
                    "L": int(context["aatype"].shape[1]),
                    "a37_rmsd": float(a37_rmsd_val.item()),
                    "clash_count": n_clashes,
                    "clash_count_target": n_clashes_target,
                    "chi1_err_rad": float(chi1_err.item()),
                    "chi2_err_rad": float(chi2_err.item()),
                    "phi_err_rad": float(phi_err.item()),
                    "psi_err_rad": float(psi_err.item()),
                    "per_aa_rmsd": {k: float(np.mean(v)) for k, v in per_aa.items()},
                    "n_valid": int(residue_mask.sum().item()) if residue_mask is not None else 0,
                }
                all_records.append(rec)

        except Exception as e:
            failed_samples.append({
                "sample_idx": sample_i,
                "protein_id": str(batch.get("protein_id", f"s{sample_i}")),
                "error": str(e),
            })
            print(f"  [WARN] sample {sample_i}: {e}")

        if (sample_i + 1) % 10 == 0:
            print(f"  [{sample_i + 1}/{n_val}]")

    # -- Save --
    with open(out_dir / "sidechain_results.jsonl", "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    if failed_samples:
        with open(out_dir / "failed_samples.json", "w") as f:
            json.dump(failed_samples, f, indent=2)
        raise RuntimeError(
            f"Side-chain evaluation incomplete: {len(failed_samples)} samples failed; "
            f"see failed_samples.json"
        )

    keys = ["a37_rmsd", "clash_count", "clash_count_target",
            "chi1_err_rad", "chi2_err_rad", "phi_err_rad", "psi_err_rad"]
    summary = {"n_samples": len(all_records)}
    for key in keys:
        vals = [r[key] for r in all_records if not np.isnan(r[key]) and not np.isinf(r[key])]
        if vals:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    all_per_aa = defaultdict(list)
    for rec in all_records:
        for aa, v in rec.get("per_aa_rmsd", {}).items():
            all_per_aa[aa].append(v)
    summary["per_aa"] = {k: {"mean": float(np.mean(v)), "n": len(v)}
                         for k, v in sorted(all_per_aa.items())}

    with open(out_dir / "sidechain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSide-Chain Quality (n={len(all_records)}):")
    print(f"{'='*60}")
    print(f"  Atom37 RMSD:   {summary.get('a37_rmsd',{}).get('mean',0):.3f} +/- {summary.get('a37_rmsd',{}).get('std',0):.3f} A")
    print(f"  Clash (pred):  {summary.get('clash_count',{}).get('mean',0):.1f}")
    print(f"  Clash (targ):  {summary.get('clash_count_target',{}).get('mean',0):.1f}")
    print(f"  Chi1:          {summary.get('chi1_err_rad',{}).get('mean',0):.2f} rad = {np.degrees(summary.get('chi1_err_rad',{}).get('mean',0)):.0f} deg")
    print(f"  Chi2:          {summary.get('chi2_err_rad',{}).get('mean',0):.2f} rad = {np.degrees(summary.get('chi2_err_rad',{}).get('mean',0)):.0f} deg")
    print(f"  Phi:           {summary.get('phi_err_rad',{}).get('mean',0):.2f} rad = {np.degrees(summary.get('phi_err_rad',{}).get('mean',0)):.0f} deg")
    print(f"  Psi:           {summary.get('psi_err_rad',{}).get('mean',0):.2f} rad = {np.degrees(summary.get('psi_err_rad',{}).get('mean',0)):.0f} deg")
    print(f"\n  Per-residue RMSD (A):")
    for aa, stats in sorted(summary.get("per_aa", {}).items()):
        print(f"    {aa}: {stats['mean']:.3f}  (n={stats['n']})")
    print(f"{'='*60}")
    print(f"Saved to {out_dir}/sidechain_summary.json")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder-ckpt", default="artifacts/mdcath_stage_c_compact/checkpoint_latest.pt")
    ap.add_argument("--stage-b-ckpt", default="artifacts/mdcath_stage_b_rollout/checkpoint_latest.pt")
    ap.add_argument("--val-manifest", default="artifacts/mdcath_manifest_200dom/val.jsonl")
    ap.add_argument("--output-dir", default="artifacts/sidechain_quality")
    ap.add_argument("--n-val", type=int, default=100)
    args = ap.parse_args()
    run_sidechain_eval(
        args.decoder_ckpt, args.stage_b_ckpt, args.val_manifest, args.output_dir,
        args.n_val,
    )
