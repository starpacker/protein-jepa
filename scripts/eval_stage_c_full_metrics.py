"""Stage C single-structure validation using the canonical metric adapter.

RMSF/PCA/W2/RMWD are ensemble metrics and are intentionally not synthesized
from one endpoint; use ``stage_c_evaluator.AlphaFlowAdapter`` for those.
"""
import json, math, sys, statistics, os
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_jepa_frame.config import DataConfig, ModelConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import PairedTrajectoryDataset, load_manifest
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_c.structure_decoder import StructureDecoder, DecoderConfig
from protein_jepa_frame.models.stage_c.direct_head import DirectEndpointHead, DirectHeadConfig
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.eval.stage_c_checkpoint import (
    load_stage_c_modules,
    load_stage_b_for_stage_c,
)
from protein_jepa_frame.eval.structure_metrics import (
    compute_structure_metrics,
    align_backbone_by_ca,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eval on {device}")

cp = os.environ.get("STAGE_C_CKPT", "artifacts/mdcath_stage_c_compact/checkpoint_latest.pt")
ckpt = torch.load(cp, map_location="cpu")

sb, mc, stage_b_source = load_stage_b_for_stage_c(
    ckpt,
    os.environ.get("STAGE_B_CKPT", "artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt"),
    device,
)
print(f"Stage-B source: {stage_b_source}")

dec, dh, _ = load_stage_c_modules(cp, device)
print("Models loaded")

manifest = load_manifest(os.environ.get("VAL_MANIFEST", "artifacts/mdcath_manifest_200dom/val.jsonl"))
dcfg = DataConfig(batch_size=1, num_workers=0, num_frames=8, frame_stride=4, window_stride=4,
                  max_residues=256, coordinate_scale=1.0, training_mode="paired_window",
                  paired_horizons_frames=[32,64,96], paired_all_horizons=True,
                  paired_history_mode="uniform")
from torch.utils.data import DataLoader
dataset = PairedTrajectoryDataset(manifest, dcfg)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=collate_paired_windows, drop_last=False)

# Metrics accumulators
all_rmsd_ca, all_rmsd_bb, all_kabsch_err_ca, all_kabsch_err_bb, all_dist_mae = [], [], [], [], []
all_contact_j, all_weak_j, all_transient_j = [], [], []

count = 0
N_MAX = 200

for batch in loader:
    if count >= N_MAX: break
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
    B, L = batch["context_coords"].shape[0], batch["context_coords"].shape[2]
    aatype = batch.get("future_aatype", batch.get("aatype"))
    if aatype.dim()==1: aatype = aatype.unsqueeze(0).expand(B,-1)
    ri = batch.get("context_residue_index", batch.get("residue_index"))
    if ri is None: ri = torch.arange(L,device=device).unsqueeze(0).expand(B,-1)
    if ri.dim()==1: ri = ri.unsqueeze(0).expand(B,-1)
    ci = batch.get("context_chain_id", batch.get("chain_id"))
    if ci is None: ci = torch.zeros(B,L,dtype=torch.long,device=device)
    if ci.dim()==1: ci = ci.unsqueeze(0).expand(B,-1)

    prot_id = batch.get("protein_id", [f"p{count}"])
    if isinstance(prot_id, list): prot_id = prot_id[0] if prot_id else f"p{count}"

    with torch.no_grad():
        temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=device))
        ctx_state = sb.encode_state(
            batch["context_coords"], batch["context_residue_mask"],
            aatype, ri, ci,
            time_ps=batch.get("context_time_ps"),
            temperature_k=temp_k,
            use_temporal=True,
            return_bath=False,
            return_ca_distances=False,
            return_forces=False,
        )
        h = ctx_state["h_last"]
        fut_state = sb.encode_state(
            batch["future_coords"], batch["future_residue_mask"],
            aatype, ri, ci,
            time_ps=batch.get("future_time_ps"),
            temperature_k=temp_k,
            use_temporal=True,
            return_bath=False,
            return_ca_distances=False,
            return_forces=False,
        )
        qo = fut_state["q"]
        ab = batch["context_coords"][:, -1]
        R,t = RigidBodyFrame.from_backbone(ab[...,0,:], ab[...,1,:], ab[...,2,:])
        tb = batch["future_coords"][:, -1]
        tm = batch["future_residue_mask"][:, -1]
        out = dec(aatype, h, qo, R, t, tm)

        eval_mask = batch["context_residue_mask"][:, -1] & tm
        pred_full = out["backbone_coords"][0]
        targ_full = tb[0]
        metrics = compute_structure_metrics(
            pred_full, targ_full, eval_mask[0],
            residue_index=ri[0], chain_id=ci[0],
        )
        rmsd_ca = metrics["rmsd_kabsch"]
        kabsch_err = metrics["kabsch_ca_point_error"]
        dm = metrics["dist_mae"]
        cj = metrics["contact_jaccard"]
        wj = metrics["weak_contact_j"]
        tj = metrics["transient_contact_j"]
        all_rmsd_ca.append(rmsd_ca)
        all_kabsch_err_ca.append(kabsch_err)
        all_dist_mae.append(dm)
        all_contact_j.append(cj)
        all_weak_j.append(wj)
        all_transient_j.append(tj)

        count += 1
        if count % 20 == 0:
            print(f"  [{count}/{N_MAX}] RMSD={rmsd_ca:.2f} KabschErr={kabsch_err:.2f} Dist={dm:.2f} CJ={cj:.3f} WJ={wj:.3f}")

# ─── Summarize metrics ───
print(f"\n{'='*70}")
print(f"STAGE C — 6-METRIC VALIDATION ({count} samples)")
print(f"{'='*70}")

def stat_str(vals):
    if not vals: return "N/A"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}" if len(vals) > 1 else f"{statistics.mean(vals):.4f}"

print(f"\n  1. RMSD_CA (Kabsch)           = {statistics.mean(all_rmsd_ca):.2f} ± {statistics.stdev(all_rmsd_ca):.2f} Å")
print(f"  2. Kabsch-aligned CA err (not FAPE) = {statistics.mean(all_kabsch_err_ca):.2f} ± {statistics.stdev(all_kabsch_err_ca):.2f} Å")
print(f"  3. Distance MAE (CA pairwise)  = {statistics.mean(all_dist_mae):.2f} ± {statistics.stdev(all_dist_mae):.2f} Å")
print(f"  4. Contact Jaccard (8Å cutoff) = {statistics.mean(all_contact_j):.4f} ± {statistics.stdev(all_contact_j):.4f}")
print(f"  5. Weak Contact J (8-15Å)      = {statistics.mean(all_weak_j):.4f} ± {statistics.stdev(all_weak_j):.4f}")
print(f"  6. Transient Contact J (15-25Å) = {statistics.mean(all_transient_j):.4f} ± {statistics.stdev(all_transient_j):.4f}")
print(f"\n  Additional ensemble metrics (RMWD/RMSF/PCA) are intentionally omitted:")
print(f"  they require the official AlphaFlow multi-frame protocol.")

# Save results
results = {
    "n_samples": count,
    "rmsd_ca_kabsch": {"mean": statistics.mean(all_rmsd_ca), "std": statistics.stdev(all_rmsd_ca) if len(all_rmsd_ca)>1 else 0},
    "kabsch_ca_point_error": {"mean": statistics.mean(all_kabsch_err_ca), "std": statistics.stdev(all_kabsch_err_ca) if len(all_kabsch_err_ca)>1 else 0},
    "dist_mae_ca": {"mean": statistics.mean(all_dist_mae), "std": statistics.stdev(all_dist_mae) if len(all_dist_mae)>1 else 0},
    "contact_jaccard": {"mean": statistics.mean(all_contact_j), "std": statistics.stdev(all_contact_j) if len(all_contact_j)>1 else 0},
    "weak_contact_j": {"mean": statistics.mean(all_weak_j), "std": statistics.stdev(all_weak_j) if len(all_weak_j)>1 else 0},
    "transient_contact_j": {"mean": statistics.mean(all_transient_j), "std": statistics.stdev(all_transient_j) if len(all_transient_j)>1 else 0},
    "metric_protocol": {
        "alignment": "OpenFold superimposition / Biopython SVDSuperimposer",
        "distance_mae": "off-diagonal nonlocal CA pair distances",
        "ensemble_metrics": "omitted; use AlphaFlowAdapter for official RMWD/RMSF/PCA",
    },
}
json.dump(results, open(os.environ.get("EVAL_OUTPUT", "artifacts/mdcath_stage_c_compact/full_metrics.json"),"w"), indent=2)
print(f"\nSaved to full_metrics.json ({count} samples)")
