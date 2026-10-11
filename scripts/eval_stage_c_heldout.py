"""Stage C held-out evaluation with proper Kabsch-aligned metrics."""
import json, math, sys, os
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import PairedTrajectoryDataset, load_manifest
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.eval.stage_c_checkpoint import (
    load_stage_c_modules,
    load_stage_b_for_stage_c,
)
from protein_jepa_frame.eval.structure_metrics import compute_structure_metrics

cp = os.environ.get("STAGE_C_CKPT", "artifacts/mdcath_stage_c_compact/checkpoint_latest.pt")
val_mf = os.environ.get("VAL_MANIFEST", "artifacts/mdcath_manifest_200dom/val.jsonl")
eval_output = os.environ.get("EVAL_OUTPUT", "artifacts/mdcath_stage_c_compact/heldout_eval.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eval on {device}")

ckpt = torch.load(cp, map_location="cpu")

sb, mc, sb_source = load_stage_b_for_stage_c(
    ckpt,
    os.environ.get("STAGE_B_CKPT", "artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt"),
    device,
)
dec, dh, _ = load_stage_c_modules(cp, device)
print(f"Models loaded (Stage-B source: {sb_source})")

manifest = load_manifest(val_mf)
dcfg = DataConfig(batch_size=1, num_workers=0, num_frames=8, frame_stride=4, window_stride=4,
                  max_residues=256, coordinate_scale=1.0, training_mode="paired_window",
                  paired_horizons_frames=[32, 64, 96], paired_all_horizons=True,
                  paired_history_mode="uniform")
from torch.utils.data import DataLoader
dataset = PairedTrajectoryDataset(manifest, dcfg)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=collate_paired_windows, drop_last=False)

all_rmsd, all_kabsch_err, all_dist = [], [], []
count = 0

for batch in loader:
    if count >= 200: break
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    B, L = batch["context_coords"].shape[0], batch["context_coords"].shape[2]
    aatype = batch.get("future_aatype", batch.get("aatype"))
    if aatype.dim()==1: aatype = aatype.unsqueeze(0).expand(B,-1)
    ri = batch.get("context_residue_index", batch.get("residue_index"))
    if ri is None: ri = torch.arange(L,device=device).unsqueeze(0).expand(B,-1)
    if ri.dim()==1: ri = ri.unsqueeze(0).expand(B,-1)
    ci = batch.get("context_chain_id", batch.get("chain_id"))
    if ci is None: ci = torch.zeros(B,L,dtype=torch.long,device=device)
    if ci.dim()==1: ci = ci.unsqueeze(0).expand(B,-1)

    with torch.no_grad():
        # CANONICAL: encode context and future through the same temporal path
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
        metrics = compute_structure_metrics(
            out["backbone_coords"][0], tb[0], eval_mask[0],
            residue_index=ri[0], chain_id=ci[0],
        )
        rmsd = metrics["rmsd_kabsch"]
        kabsch_err = metrics["kabsch_ca_point_error"]
        dm = metrics["dist_mae"]

        all_rmsd.append(rmsd); all_kabsch_err.append(kabsch_err); all_dist.append(dm)
        count += 1
        if count % 20 == 0:
            print(f"  [{count}/200] RMSD={rmsd:.2f} KabschErr={kabsch_err:.2f} DistMAE={dm:.2f}")

mr = sum(all_rmsd)/len(all_rmsd) if all_rmsd else 0
mf = sum(all_kabsch_err)/len(all_kabsch_err) if all_kabsch_err else 0
md = sum(all_dist)/len(all_dist) if all_dist else 0

print(f"\n===== STAGE C HELD-OUT ({count} samples) =====")
print(f"  RMSD_CA (Kabsch):        {mr:.2f} Å")
print(f"  Kabsch-aligned BB err (not FAPE): {mf:.2f} Å")
print(f"  DistMAE_CA:               {md:.2f} Å")

json.dump({"rmsd_ca_kabsch_mean":mr,"kabsch_aligned_bb_point_error":mf,"dist_mae_ca_mean":md,"n_samples":count},
          open(eval_output,"w"), indent=2)
print("Saved.")
