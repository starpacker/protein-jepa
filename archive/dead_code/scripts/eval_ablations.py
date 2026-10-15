#!/usr/bin/env python3
"""P0.3: Ablation evaluation — isolate Q and H contributions to structure quality.

Tests 6 decoder variants on the same val samples:
  - oracle:     ground-truth Q from future coords (upper bound)
  - direct:     DirectHead predicted Q (learned endpoint)
  - q_zero:     Q replaced by zeros
  - h_zero:     H replaced by zeros
  - q_noise:    oracle Q + Gaussian noise (σ=1.0)
  - q_random:   Q replaced by random normal

Decision table from results:
  If q_zero ≈ oracle  → Q unused, decoder learned from H alone
  If h_zero fails     → H critical, geometry comes from frozen context
  If q_noise ≈ oracle → Q is insensitive to perturbation
  If q_random ≈ oracle → Q content is completely ignored
"""

from __future__ import annotations
import json, math, sys, random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import PairedTrajectoryDataset, load_manifest
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_c.structure_decoder import StructureDecoder, DecoderConfig
from protein_jepa_frame.models.stage_c.direct_head import DirectEndpointHead, DirectHeadConfig
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.models.stage_c.ablations import (
    QZeroDecoder, HZeroDecoder, QNoiseDecoder, QRandomDecoder,
)

# Reuse metric functions from eval_world_model
from scripts.eval_world_model import (
    kabsch_rmsd, kabsch_align_bb, contacts, jaccard, compute_all_metrics,
)


def load_models(stage_c_ckpt, stage_b_ckpt, device):
    """Identical to eval_world_model.load_models."""
    cp = torch.load(stage_c_ckpt, map_location="cpu")
    cp_state = cp.get("model", cp)
    sbc = torch.load(stage_b_ckpt, map_location="cpu")
    sbs = sbc.get("model", sbc)
    sb_cfg_raw = sbc.get("config", {}).get("model", {})
    mc = ModelConfig(
        latent_dim=sb_cfg_raw.get("latent_dim", 128),
        encoder_depth=sb_cfg_raw.get("encoder_depth", 4),
        temporal_depth=sb_cfg_raw.get("temporal_depth", 2),
        num_heads=sb_cfg_raw.get("num_heads", 4),
        num_rbf=sb_cfg_raw.get("num_rbf", 16),
        max_frames=sb_cfg_raw.get("max_frames", 128),
        max_residues=sb_cfg_raw.get("max_residues", 256),
        dropout=0.0, orientation_features=False,
        time_conditioning=sb_cfg_raw.get("time_conditioning", "physical"),
        thermodynamic_conditioning=sb_cfg_raw.get("thermodynamic_conditioning", True),
    )
    use_bath = sb_cfg_raw.get("lspwm_use_bath", True)
    history_aware = sb_cfg_raw.get("lspwm_history_aware", True)
    sb = StageBLASP(mc, use_bath=use_bath, history_aware=history_aware).to(device)
    rm = {}
    for k, v in sbs.items():
        if k.startswith("context_encoder."):
            rm["spatial_encoder." + k.removeprefix("context_encoder.")] = v
        else:
            rm[k] = v
    sb.load_state_dict(rm, strict=False); sb.eval()
    for p in sb.parameters(): p.requires_grad = False

    dc = DecoderConfig(context_dim=mc.latent_dim, q_dim=8, c_hidden=128,
                       n_heads=4, n_ipa_blocks=4)
    dec = StructureDecoder(dc).to(device); dec.eval()
    dp = {k: v for k, v in cp_state.items() if not k.startswith("direct_head.")}
    dec.load_state_dict(dp, strict=False)

    dhc = DirectHeadConfig(context_dim=mc.latent_dim, condition_dim=0, hidden_dim=128)
    dh = DirectEndpointHead(dhc).to(device); dh.eval()
    dh_params = {k.removeprefix("direct_head."): v for k, v in cp_state.items()
                 if k.startswith("direct_head.")}
    dh.load_state_dict(dh_params, strict=False)

    # Build ablation variants
    q_zero_dec = QZeroDecoder(dec)
    h_zero_dec = HZeroDecoder(dec)
    q_noise_dec = QNoiseDecoder(dec, noise_std=1.0)
    q_random_dec = QRandomDecoder(dec)

    return sb, dec, dh, q_zero_dec, h_zero_dec, q_noise_dec, q_random_dec, mc.latent_dim


def run_ablation_eval(
    stage_c_ckpt="artifacts/mdcath_stage_c_compact/checkpoint_latest.pt",
    stage_b_ckpt="artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt",
    val_manifest="artifacts/mdcath_manifest_200dom/val.jsonl",
    output_dir="artifacts/ablation_eval",
    n_val_samples=200,
    seeds=(42,),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ablation Eval | device={device} | n_val={n_val_samples}")

    sb, dec, dh, q_zero_dec, h_zero_dec, q_noise_dec, q_random_dec, latent_dim = \
        load_models(stage_c_ckpt, stage_b_ckpt, device)

    manifest = load_manifest(val_manifest)
    dcfg = DataConfig(
        batch_size=1, num_workers=0, num_frames=8, frame_stride=4,
        window_stride=4, max_residues=256, coordinate_scale=1.0,
        training_mode="paired_window", paired_horizons_frames=[32, 64, 96],
        paired_all_horizons=True, paired_history_mode="uniform",
    )
    dataset = PairedTrajectoryDataset(manifest, dcfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_paired_windows, drop_last=False)

    # Ablation types
    ablation_modes = {
        "oracle":   ("oracle", dec),        # reference decoder (upper bound)
        "direct":   ("direct", dec),        # DirectHead predicted Q
        "q_zero":   ("q_zero", q_zero_dec), # Q → 0
        "h_zero":   ("h_zero", h_zero_dec), # H → 0
        "q_noise":  ("q_noise", q_noise_dec), # Q + noise
        "q_random": ("q_random", q_random_dec), # random Q
    }

    metric_names = ["rmsd_kabsch", "kabsch_ca_point_error", "dist_mae",
                    "contact_jaccard", "weak_contact_j", "transient_contact_j"]

    # Per-ablation-mode metric accumulators
    agg = {abbr: {m: [] for m in metric_names} for abbr in ablation_modes}
    all_records = []
    sample_idx = 0

    for batch in loader:
        if sample_idx >= n_val_samples:
            break
        sample_idx += 1
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        ctx_coords = batch["context_coords"]
        fut_coords = batch["future_coords"]
        B, L = ctx_coords.shape[0], ctx_coords.shape[2]
        aatype = batch.get("future_aatype", batch.get("aatype"))
        if aatype.dim() == 1: aatype = aatype.unsqueeze(0).expand(B, -1)
        ri = batch.get("context_residue_index", batch.get("residue_index"))
        if ri is None: ri = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        if ri.dim() == 1: ri = ri.unsqueeze(0).expand(B, -1)
        ci = batch.get("context_chain_id", batch.get("chain_id"))
        if ci is None: ci = torch.zeros(B, L, dtype=torch.long, device=device)
        if ci.dim() == 1: ci = ci.unsqueeze(0).expand(B, -1)
        horiz = batch.get("horizon_time_ps", torch.tensor([1000.0], device=device))
        if horiz.dim() == 0: horiz = horiz.unsqueeze(0)
        temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=device))
        if temp_k.dim() > 1: temp_k = temp_k[:, -1]
        ctx_mask = batch["context_residue_mask"]
        fut_mask = batch["future_residue_mask"]
        residue_mask = ctx_mask[:, -1]

        protein_id = batch.get("protein_id", f"sample_{sample_idx}")
        if hasattr(protein_id, '__iter__') and not isinstance(protein_id, str):
            protein_id = str(protein_id[0]) if len(protein_id) > 0 else f"sample_{sample_idx}"

        anchor_bb = ctx_coords[:, -1]
        target_bb = fut_coords[:, -1]
        target_fut_mask = fut_mask[:, -1]

        try:
            with torch.no_grad():
                ctx_state = sb.encode_state(
                    ctx_coords, ctx_mask, aatype, ri, ci,
                    time_ps=batch.get("context_time_ps"),
                    temperature_k=temp_k,
                    use_temporal=True,
                    return_bath=False,
                    return_ca_distances=False,
                    return_forces=False,
                )
                h_frozen = ctx_state["h_last"]
                fut_state = sb.encode_state(
                    fut_coords, fut_mask, aatype, ri, ci,
                    time_ps=batch.get("future_time_ps"),
                    temperature_k=temp_k,
                    use_temporal=True,
                    return_bath=False,
                    return_ca_distances=False,
                    return_forces=False,
                )
                q_oracle = fut_state["q"]

                R_anchor, t_anchor = RigidBodyFrame.from_backbone(
                    anchor_bb[..., 0, :], anchor_bb[..., 1, :], anchor_bb[..., 2, :])

                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    rec = {"protein_id": protein_id, "sample_idx": sample_idx, "seed": seed}

                    for abbr, (label, decoder) in ablation_modes.items():
                        if abbr == "oracle":
                            q_use = q_oracle
                        elif abbr == "direct":
                            q_use, _ = dh(h_frozen, None, horiz, temp_k, residue_mask)
                        elif abbr in ("q_zero", "q_random"):
                            # These decoders ignore the passed Q
                            q_use = q_oracle
                        elif abbr in ("h_zero", "q_noise"):
                            q_use = q_oracle
                        else:
                            q_use = q_oracle

                        try:
                            out = decoder(aatype, h_frozen, q_use, R_anchor, t_anchor, residue_mask)
                            pred_bb = out["backbone_coords"]
                            metrics = compute_all_metrics(
                                pred_bb[0], target_bb[0], target_fut_mask[0])
                        except Exception as dec_err:
                            metrics = {m: float('nan') for m in metric_names}

                        for m in metric_names:
                            val = metrics.get(m, float('nan'))
                            rec[f"{abbr}_{m}"] = val
                            if not math.isnan(val) and not math.isinf(val):
                                agg[abbr][m].append(val)

                    all_records.append(rec)

        except Exception as e:
            print(f"  [WARN] sample {sample_idx} ({protein_id}): {e}")
            continue

        if sample_idx % 20 == 0:
            print(f"  [{sample_idx}/{n_val_samples}]")

    # ── Save ──
    with open(out_dir / "ablation_results.jsonl", "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    summary = {}
    for abbr in ablation_modes:
        summary[abbr] = {}
        for m in metric_names:
            vals = agg[abbr][m]
            if vals:
                summary[abbr][m] = {"mean": float(np.mean(vals)),
                                     "std": float(np.std(vals)),
                                     "n": len(vals)}

    with open(out_dir / "ablation_summary.json", "w") as f:
        json.dump({"n_samples": sample_idx, "n_records": len(all_records),
                   "summary": summary}, f, indent=2)

    # ── Print ──
    print(f"\n{'='*100}")
    print(f"  ABLATION EVALUATION ({len(all_records)} records from {sample_idx} samples)")
    print(f"{'='*100}")
    header = f"  {'Metric':<28}"
    for abbr in ablation_modes:
        header += f" | {abbr:<12}"
    print(header)
    print("  " + "-" * 96)

    for m in metric_names:
        line = f"  {m:<26}"
        oracle_mean = summary["oracle"].get(m, {}).get("mean", 0)
        for abbr in ablation_modes:
            v = summary[abbr].get(m, {})
            mean_val = v.get("mean", 0)
            delta = mean_val - oracle_mean if abbr != "oracle" else 0
            if abbr == "h_zero" and mean_val > oracle_mean * 2:
                marker = "!!!"
            elif abs(delta) < 0.05:
                marker = " ≈ "
            elif delta > 1.0:
                marker = " ↓ "
            else:
                marker = "   "
            line += f" | {mean_val:<8.4f}{marker}"
        print(line)
    print(f"{'='*100}")
    print(f"Saved to {out_dir}/ablation_summary.json")

    # ── Decision table ──
    print(f"\n{'='*80}")
    print("  ABLATION DECISION TABLE")
    print(f"{'='*80}")
    rmsd = {abbr: summary[abbr].get("rmsd_kabsch", {}).get("mean", 0)
            for abbr in ablation_modes}
    oracle_r = rmsd["oracle"]

    decisions = []
    if abs(rmsd.get("q_zero", 0) - oracle_r) < 0.10:
        decisions.append("Q=0 ≈ oracle → Q UNUSED — decoder learned from H alone")
    else:
        decisions.append("Q=0 worse than oracle → Q GENUINELY CONTRIBUTES")

    if rmsd.get("h_zero", 0) > oracle_r * 2:
        decisions.append("H=0 catastrophic → H CRITICAL — geometry from frozen context")
    elif rmsd.get("h_zero", 0) < oracle_r * 1.5:
        decisions.append("H=0 not catastrophic → decoder learned from Q alone (unexpected)")

    if abs(rmsd.get("q_noise", 0) - oracle_r) < 0.10:
        decisions.append("Q+noise ≈ oracle → Q SENSITIVITY LOW — perturbations ignored")
    else:
        decisions.append("Q+noise worse → Q SENSITIVE — precise Q value matters")

    if abs(rmsd.get("q_random", 0) - oracle_r) < 0.10:
        decisions.append("Q_random ≈ oracle → Q CONTENT COMPLETELY IGNORED")
    else:
        decisions.append("Q_random worse → Q CONTENT MATTERS for structure")

    for d in decisions:
        print(f"  • {d}")
    print(f"{'='*80}")

    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-c-ckpt", default="artifacts/mdcath_stage_c_compact/checkpoint_latest.pt")
    ap.add_argument("--stage-b-ckpt", default="artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt")
    ap.add_argument("--val-manifest", default="artifacts/mdcath_manifest_200dom/val.jsonl")
    ap.add_argument("--output-dir", default="artifacts/ablation_eval")
    ap.add_argument("--n-val-samples", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()
    run_ablation_eval(args.stage_c_ckpt, args.stage_b_ckpt, args.val_manifest,
                      args.output_dir, args.n_val_samples, tuple(args.seeds))
