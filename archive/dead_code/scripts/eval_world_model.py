#!/usr/bin/env python3
"""P0.2: Canonical World-Model Evaluation Framework.

5 disjoint columns × 4 horizons × 3 seeds per sample:
  oracle_endpoint  → future coords → Stage B encode → q_oracle → decoder
  direct_endpoint  → DirectHead(H, tau, T) → q_direct → decoder
  autonomous_recursive → H → q0, step(h) n times → q_rec → decoder
  persistence      → anchor coords (no model)
  q_shuffle        → shuffled oracle Q across batch → decoder

All metrics Kabsch-aligned. No cherry-picking. Per-(protein, horizon, seed) raw data.
"""

from __future__ import annotations
import json, math, sys, random, time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import PairedTrajectoryDataset, load_manifest
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_c.structure_decoder import StructureDecoder, DecoderConfig
from protein_jepa_frame.models.stage_c.direct_head import DirectEndpointHead, DirectHeadConfig
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.eval.stage_c_checkpoint import (
    split_stage_c_state,
    stage_c_configs_from_checkpoint,
    load_stage_b_for_stage_c,
)
from protein_jepa_frame.eval.structure_metrics import (
    official_kabsch,
    align_backbone_by_ca,
    compute_structure_metrics,
)
from eval_sampling import stratified_indices

# ═══════════════════════════════════════════════════════════════════════
# Kabsch alignment
# ═══════════════════════════════════════════════════════════════════════

def kabsch_rmsd(P, Q):
    """Compatibility wrapper around the OpenFold/ Biopython fit."""
    _, rmsd = official_kabsch(P, Q)
    return rmsd

def kabsch_align_bb(pred_bb, targ_bb):
    """Compatibility wrapper around the audited CA Kabsch adapter."""
    return align_backbone_by_ca(pred_bb, targ_bb)

# ═══════════════════════════════════════════════════════════════════════
# Metric functions
# ═══════════════════════════════════════════════════════════════════════

def contacts(ca, lo, hi):
    """Deprecated compatibility helper; use ``compute_structure_metrics``."""
    d = torch.cdist(ca.unsqueeze(0), ca.unsqueeze(0))[0]
    n = ca.shape[0]
    ij = torch.arange(n, device=ca.device)
    nonlocal_mask = (~torch.eye(n, dtype=torch.bool, device=ca.device)) & (
        (ij[:, None] - ij[None, :]).abs() > 3
    )
    return ((d >= lo) & (d < hi) & nonlocal_mask).float()

def jaccard(pred, true):
    """Deprecated compatibility helper with the standard empty-set rule."""
    union = pred.bool() | true.bool()
    if not union.any():
        return pred.new_tensor(1.0)
    return (pred.bool() & true.bool()).float().sum() / union.float().sum()

def compute_all_metrics(pred_bb, targ_bb, residue_mask, residue_index=None, chain_id=None):
    """Compute 7 structural metrics on a single protein.
    
    pred_bb: [L, 4, 3] (N,CA,C,O)
    targ_bb: [L, 4, 3]
    residue_mask: [L] bool
    """
    return compute_structure_metrics(
        pred_bb, targ_bb, residue_mask,
        residue_index=residue_index,
        chain_id=chain_id,
    )

# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def load_models(stage_c_checkpoint, stage_b_checkpoint, device):
    """Load Stage B (frozen) + Stage C decoder + DirectHead."""
    # ── Stage C checkpoint ──
    cp = torch.load(stage_c_checkpoint, map_location="cpu")
    cp_state = cp.get("model", cp)

    # ── Stage B ──
    sb, mc, stage_b_source = load_stage_b_for_stage_c(
        cp, stage_b_checkpoint, device
    )
    print(f"  Stage-B source: {stage_b_source}")

    # ── Decoder ──
    dc, dhc = stage_c_configs_from_checkpoint(cp, context_dim=mc.latent_dim)
    dec = StructureDecoder(dc).to(device)
    dec.eval()

    # ── DirectHead ──
    dh = DirectEndpointHead(dhc).to(device)
    dh.eval()
    dp, dh_params = split_stage_c_state(dec, dh, cp_state)
    dec.load_state_dict(dp, strict=True)
    dh.load_state_dict(dh_params, strict=True)

    return sb, dec, dh, mc.latent_dim

# ═══════════════════════════════════════════════════════════════════════
# Evaluation core
# ═══════════════════════════════════════════════════════════════════════

def run_world_model_eval(
    stage_c_ckpt="artifacts/mdcath_stage_c_compact/checkpoint_latest.pt",
    stage_b_ckpt="artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt",
    val_manifest="artifacts/mdcath_manifest_200dom/val.jsonl",
    output_dir="artifacts/world_model_eval",
    n_val_samples=200,
    seeds=(42, 123, 456),
    horizons_ps=None,
    base_step_ps=100.0,
    max_substep_dt_ps=100.0,
    evaluate_autonomous=True,
    num_frames=8,
    frame_stride=4,
    paired_horizons_frames=(32, 64, 96),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"World Model Eval | device={device} | output={out_dir}")
    print(f"  horizons_ps={horizons_ps or 'dataset horizons'} | seeds={seeds} | n_val={n_val_samples}")

    # ── Load models ──
    sb, dec, dh, latent_dim = load_models(stage_c_ckpt, stage_b_ckpt, device)
    print("  Models loaded.")

    # ── Data ──
    manifest = load_manifest(val_manifest)
    dcfg = DataConfig(
        batch_size=1, num_workers=0, num_frames=num_frames, frame_stride=frame_stride,
        window_stride=4, max_residues=256, coordinate_scale=1.0,
        training_mode="paired_window", paired_horizons_frames=list(paired_horizons_frames),
        paired_all_horizons=False, paired_history_mode="uniform",
    )
    dataset = PairedTrajectoryDataset(manifest, dcfg)
    eval_indices = stratified_indices(dataset, n_val_samples, seed=seeds[0] if seeds else 42)
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_paired_windows, drop_last=False)

    # ── Per-sample results ──
    all_records = []
    failed_samples = []  # collect failures for fail-fast check
    sample_idx = 0
    eval_started = time.perf_counter()

    for batch in loader:
        if sample_idx >= n_val_samples:
            break
        sample_idx += 1

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        ctx_coords = batch["context_coords"]   # [1, T_ctx, L, 4, 3]
        fut_coords = batch["future_coords"]    # [1, T_fut, L, 4, 3]
        B, T_ctx, L = ctx_coords.shape[:3]
        T_fut = fut_coords.shape[1]

        aatype = batch.get("future_aatype", batch.get("aatype"))
        if aatype.dim() == 1:
            aatype = aatype.unsqueeze(0).expand(B, -1)
        ri = batch.get("context_residue_index", batch.get("residue_index"))
        if ri is None:
            ri = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        if ri.dim() == 1:
            ri = ri.unsqueeze(0).expand(B, -1)
        ci = batch.get("context_chain_id", batch.get("chain_id"))
        if ci is None:
            ci = torch.zeros(B, L, dtype=torch.long, device=device)
        if ci.dim() == 1:
            ci = ci.unsqueeze(0).expand(B, -1)

        horiz_ps_batch = batch.get("horizon_time_ps", torch.tensor([1000.0], device=device))
        if horiz_ps_batch.dim() == 0:
            horiz_ps_batch = horiz_ps_batch.unsqueeze(0)
        temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=device))
        if temp_k.dim() > 1:
            temp_k = temp_k[:, -1]

        ctx_mask = batch["context_residue_mask"]  # [1, T_ctx, L]
        fut_mask = batch["future_residue_mask"]   # [1, T_fut, L]
        residue_mask = ctx_mask[:, -1]  # [1, L], last context frame

        # Protein ID
        protein_id = batch.get("protein_id", f"sample_{sample_idx}")
        if hasattr(protein_id, '__iter__') and not isinstance(protein_id, str):
            protein_id = str(protein_id[0]) if len(protein_id) > 0 else f"sample_{sample_idx}"

        anchor_bb = ctx_coords[:, -1]    # [1, L, 4, 3]
        target_bb = fut_coords[:, -1]    # [1, L, 4, 3]
        target_fut_mask = fut_mask[:, -1]  # [1, L]
        # Evaluation must use residues present in both the anchor (the model's
        # actual input) and the future target.  Using only the future mask can
        # score predictions generated from padded/missing anchor residues.
        eval_residue_mask = residue_mask & target_fut_mask
        target_horizon_ps = float(batch["horizon_time_ps"][0].item())
        eval_horizons = [target_horizon_ps] if horizons_ps is None else [
            float(h) for h in horizons_ps if abs(float(h) - target_horizon_ps) < 1e-3
        ]
        if not eval_horizons:
            continue

        sample_ok = True
        sample_error = None
        try:
            with torch.no_grad():
                # ── Encode context (shared across all modes) ──
                # This must use the same temporal-history path as Stage B
                # training; direct forward_spatial would silently drop Q's
                # history conditioning.
                state = sb.encode_state(
                    ctx_coords, ctx_mask, aatype, ri, ci,
                    time_ps=batch.get("context_time_ps"),
                    temperature_k=temp_k,
                    use_temporal=True,
                    return_bath=True,
                    return_ca_distances=True,
                    return_forces=False,
                )
                h_frozen = state["h_last"]
                q0 = state["q"].detach()
                ca_dists = state["ca_distances"]

                # ── Oracle Q ──
                # CANONICAL: future endpoint uses THE SAME encode_state
                # path (use_temporal=True) as context, so oracle Q and
                # context Q are in the same representation space.
                future_state = sb.encode_state(
                    fut_coords, fut_mask, aatype, ri, ci,
                    time_ps=batch.get("future_time_ps"),
                    temperature_k=temp_k,
                    use_temporal=True,
                    return_bath=False,
                    return_ca_distances=False,
                    return_forces=False,
                )
                q_oracle = future_state["q"]  # [1, L, 8]

                # Anchor frames
                R_anchor, t_anchor = RigidBodyFrame.from_backbone(
                    anchor_bb[..., 0, :], anchor_bb[..., 1, :], anchor_bb[..., 2, :])

                # Basin-reversion prior (context-only, eval-legal): aligned
                # context-mean CA displacement in the anchor frame.  Computed
                # identically to Stage-C training so the decoder sees the same
                # geometric signal it was trained with.
                from protein_jepa_frame.models.stage_c.structure_decoder import (
                    StructureDecoder as _SD,
                )
                basin_dt_local = _SD.compute_basin_dt_local(
                    ctx_coords, R_anchor, t_anchor,
                    residue_mask=residue_mask.bool(),
                )

                # Step function
                def decode_with_q(q):
                    out = dec(aatype, h_frozen, q, R_anchor, t_anchor, residue_mask,
                              basin_dt_local=basin_dt_local)
                    return out["backbone_coords"]

                # Run all autonomous seeds in one batched closed-loop rollout.
                # The old implementation repeated the expensive force probe
                # independently for each seed, tripling wall time.
                autonomous_metrics = {}
                if evaluate_autonomous:
                    q0_roll = q0  # reuse canonical Q from context state encoding
                    s0_roll = state.get("bath_s0", sb.initialize_bath(h_frozen, residue_mask)).detach()
                    n_seed = len(seeds)
                    roll_aa = aatype.expand(n_seed, -1)
                    roll_h = h_frozen.expand(n_seed, -1, -1)
                    roll_R = R_anchor.expand(n_seed, -1, -1, -1)
                    roll_t = t_anchor.expand(n_seed, -1, -1)
                    roll_mask = residue_mask.expand(n_seed, -1)
                    roll_temp = temp_k.expand(n_seed)
                    roll_dist = ca_dists.expand(n_seed, -1, -1)
                    for h_ps in eval_horizons:
                        q_t = q0_roll.expand(n_seed, -1, -1).clone()
                        s_t = s0_roll.expand(n_seed, -1, -1).clone()
                        n_steps = int(h_ps // base_step_ps)
                        remainder = h_ps - n_steps * base_step_ps
                        if remainder < 1e-3 * base_step_ps:
                            remainder = 0.0
                        for _ in range(n_steps):
                            with torch.no_grad():
                                q_t, s_t, _ = sb.transition(
                                    q_t, s_t, ca_distances=roll_dist,
                                    residue_mask=roll_mask,
                                    temperature_k=roll_temp,
                                    dt_ps=base_step_ps,
                                    max_substep_dt_ps=max_substep_dt_ps,
                                    stochastic=True,
                                )
                        if remainder > 0:
                            with torch.no_grad():
                                q_t, s_t, _ = sb.transition(
                                    q_t, s_t, ca_distances=roll_dist,
                                    residue_mask=roll_mask,
                                    temperature_k=roll_temp,
                                    dt_ps=remainder,
                                    max_substep_dt_ps=max_substep_dt_ps,
                                    stochastic=True,
                                )
                        pred_roll = dec(
                            roll_aa, roll_h, q_t, roll_R, roll_t, roll_mask,
                            basin_dt_local=basin_dt_local.expand(n_seed, -1, -1),
                        )["backbone_coords"]
                        for seed_i, seed in enumerate(seeds):
                            autonomous_metrics[(seed, h_ps)] = compute_all_metrics(
                                pred_roll[seed_i], target_bb[0], eval_residue_mask[0],
                                ri[0], ci[0],
                            )

                # These four endpoint conditions do not depend on the random
                # seed.  Compute them once per horizon, then reuse them for
                # every seed.  The previous implementation repeated all four
                # decoder/metric passes once per seed (3x redundant here).
                endpoint_metrics = {}
                for h_ps in eval_horizons:
                    pred_bb_o = decode_with_q(q_oracle)
                    metrics_o = compute_all_metrics(
                        pred_bb_o[0], target_bb[0], eval_residue_mask[0],
                        ri[0], ci[0],
                    )

                    h_tensor = torch.full((1,), h_ps, device=device)
                    q_direct_mean, _ = dh(h_frozen, None, h_tensor, temp_k, residue_mask)
                    pred_bb_d = decode_with_q(q_direct_mean)
                    metrics_d = compute_all_metrics(
                        pred_bb_d[0], target_bb[0], eval_residue_mask[0],
                        ri[0], ci[0],
                    )

                    metrics_p = compute_all_metrics(
                        anchor_bb[0], target_bb[0], eval_residue_mask[0],
                        ri[0], ci[0],
                    )

                    # With one protein per evaluation record, shuffle residue
                    # assignments so this is a real semantic ablation rather
                    # than the old batch-size-1 identity permutation.
                    q_shuffled = q_oracle.roll(shifts=1, dims=1)
                    pred_bb_qs = decode_with_q(q_shuffled)
                    metrics_qs = compute_all_metrics(
                        pred_bb_qs[0], target_bb[0], eval_residue_mask[0],
                        ri[0], ci[0],
                    )
                    endpoint_metrics[h_ps] = (metrics_o, metrics_d, metrics_p, metrics_qs)

                # ── For each seed & horizon ──
                for seed in seeds:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)

                    for h_ps in eval_horizons:
                        metrics_o, metrics_d, metrics_p, metrics_qs = endpoint_metrics[h_ps]
                        rec = {
                            "protein_id": protein_id,
                            "sample_idx": sample_idx,
                            "seed": seed,
                            "horizon_ps": h_ps,
                            "L": L,
                            "temperature_k": float(temp_k.mean().item()),
                        }
                        for k, v in metrics_o.items():
                            rec[f"oracle_{k}"] = v
                        for k, v in metrics_d.items():
                            rec[f"direct_{k}"] = v
                        for k, v in autonomous_metrics.get((seed, h_ps), {}).items():
                            rec[f"autonomous_{k}"] = v
                        for k, v in metrics_p.items():
                            rec[f"persistence_{k}"] = v
                        for k, v in metrics_qs.items():
                            rec[f"qshuffle_{k}"] = v
                        all_records.append(rec)

        except Exception as e:
            sample_ok = False
            sample_error = str(e)
            failed_samples.append({
                "sample_idx": sample_idx,
                "protein_id": str(protein_id),
                "error": str(e),
            })
            print(f"  [ERROR] sample {sample_idx} ({protein_id}) FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()

        elapsed = time.perf_counter() - eval_started
        rate = sample_idx / max(elapsed, 1e-6)
        eta_s = (n_val_samples - sample_idx) / max(rate, 1e-6)
        print(
            f"  [{sample_idx}/{n_val_samples}] processed | "
            f"{elapsed/60:.1f} min elapsed | {rate:.2f} samples/s | "
            f"ETA {eta_s/60:.1f} min",
            flush=True,
        )

    # ── Fail-fast: reject incomplete evaluations ──
    required_modes = ["oracle", "direct", "autonomous", "persistence", "qshuffle"]

    if sample_idx != n_val_samples:
        failed_samples.append({
            "sample_idx": sample_idx + 1,
            "protein_id": "<dataset>",
            "error": (
                f"requested {n_val_samples} samples but processed {sample_idx}; "
                "evaluation dataset was shorter or samples were skipped"
            ),
        })

    if failed_samples:
        print(f"\n{'='*90}")
        print(f"  FAIL-FAST: {len(failed_samples)} samples FAILED out of {sample_idx} processed")
        for f in failed_samples:
            print(f"    sample {f['sample_idx']} ({f['protein_id']}): {f['error']}")
        print(f"  Evaluation is INCOMPLETE — refusing to generate pseudo-report with status=passed")
        print(f"{'='*90}")
        sys.exit(1)

    # Mode count validation: every required mode must have at least one record
    mode_record_counts = {}
    for mode in required_modes:
        prefix = f"{mode}_"
        mode_record_counts[mode] = sum(1 for r in all_records if any(k.startswith(prefix) for k in r))
    empty_modes = [m for m, c in mode_record_counts.items() if c == 0]
    if empty_modes:
        print(f"\n{'='*90}")
        print(f"  FAIL-FAST: modes with n=0 records: {empty_modes}")
        print(f"  Mode record counts: {mode_record_counts}")
        print(f"  Refusing to generate report with missing mode columns")
        print(f"{'='*90}")
        sys.exit(1)

    expected_rows = sample_idx * len(seeds)
    if len(all_records) != expected_rows:
        print(
            f"FAIL-FAST: expected {expected_rows} complete records "
            f"({sample_idx} samples × {len(seeds)} seeds), got {len(all_records)}",
            flush=True,
        )
        sys.exit(1)

    # ── Save & summarize ──
    with open(out_dir / "per_protein_results.jsonl", "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved {len(all_records)} records to {out_dir / 'per_protein_results.jsonl'}")

    # Aggregate by metric × mode
    metric_names = ["rmsd_kabsch", "kabsch_ca_point_error", "dist_mae",
                    "contact_jaccard", "weak_contact_j", "transient_contact_j"]
    mode_names = ["oracle", "direct", "autonomous", "persistence", "qshuffle"]

    summary = {}
    invalid_metric_values = []
    for mode in mode_names:
        summary[mode] = {}
        for mn in metric_names:
            key = f"{mode}_{mn}"
            vals = [r[key] for r in all_records if key in r and r[key] is not None
                    and not math.isnan(r[key]) and not math.isinf(r[key])]
            invalid_count = sum(
                1 for r in all_records
                if key in r and (
                    r[key] is None or not math.isfinite(float(r[key]))
                )
            )
            if invalid_count:
                invalid_metric_values.append({"mode": mode, "metric": mn, "n": invalid_count})
            if vals:
                summary[mode][mn] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
            else:
                summary[mode][mn] = {"mean": 0, "std": 0, "n": 0}

    if invalid_metric_values:
        print(f"FAIL-FAST: non-finite metric values: {invalid_metric_values}", flush=True)
        sys.exit(1)

    with open(out_dir / "full_report.json", "w") as f:
        json.dump({
            "status": "passed",
            "n_samples": sample_idx,
            "n_records": len(all_records),
            "n_failed": len(failed_samples),
            "horizons_ps": list(horizons_ps) if horizons_ps is not None else sorted({
                r["horizon_ps"] for r in all_records
            }),
            "seeds": list(seeds),
            "mode_record_counts": mode_record_counts,
            "failed_samples": failed_samples,
            "metric_protocol": {
                "alignment": "OpenFold superimposition / Biopython SVDSuperimposer",
                "coordinate_unit": "angstrom",
                "rmsd": "CA Kabsch RMSD",
                "point_error": "mean Euclidean error over N/CA/C/O after CA fit",
                "distance_mae": "off-diagonal nonlocal CA pair distances",
                "contacts": "CA distance bins [0,8), [8,15), [15,25), same-chain |residue_index_delta|>3",
                "invalid_samples": "hard failure; never dropped into an aggregate",
                "rmwd": "not reported; requires the AlphaFlow ensemble protocol",
            },
            "summary": summary,
        }, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print("  WORLD MODEL EVALUATION — SUMMARY TABLE (status=passed)")
    print("=" * 90)
    header = f"{'Metric':<30}"
    for mode in mode_names:
        header += f" | {mode:<16}"
    print(header)
    print("-" * 90)
    for mn in metric_names:
        line = f"  {mn:<28}"
        for mode in mode_names:
            v = summary[mode].get(mn, {})
            mean_val = v.get("mean", 0)
            line += f" | {mean_val:<16.4f}"
        print(line)
    print("=" * 90)
    print(f"Saved to {out_dir}/full_report.json")

    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-c-ckpt", default="artifacts/mdcath_stage_c_compact/checkpoint_latest.pt")
    ap.add_argument("--stage-b-ckpt", default="artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt")
    ap.add_argument("--val-manifest", default="artifacts/mdcath_manifest_200dom/val.jsonl")
    ap.add_argument("--output-dir", default="artifacts/world_model_eval")
    ap.add_argument("--n-val-samples", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--horizons-ps", type=float, nargs="+", default=None,
                    help="optional exact dataset horizons; default evaluates each target horizon")
    ap.add_argument("--base-step-ps", type=float, default=100.0)
    ap.add_argument("--max-substep-dt-ps", type=float, default=100.0)
    ap.add_argument("--skip-autonomous", action="store_true")
    ap.add_argument("--num-frames", type=int, default=8,
                    help="context window frame count")
    ap.add_argument("--frame-stride", type=int, default=4,
                    help="raw-frame stride between context frames")
    ap.add_argument("--paired-horizons-frames", type=int, nargs="+", default=[32, 64, 96],
                    help="horizons in raw frames (mdCATH: 1 frame = 1 ns)")
    args = ap.parse_args()
    run_world_model_eval(
        args.stage_c_ckpt, args.stage_b_ckpt, args.val_manifest,
        args.output_dir, args.n_val_samples, tuple(args.seeds),
        tuple(args.horizons_ps) if args.horizons_ps is not None else None,
        args.base_step_ps, args.max_substep_dt_ps, not args.skip_autonomous,
        args.num_frames, args.frame_stride, tuple(args.paired_horizons_frames),
    )
