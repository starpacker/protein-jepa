"""Stage C end-to-end eval: DirectHead Q → Decoder → Kabsch metrics."""
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
eval_output = os.environ.get("EVAL_OUTPUT", "artifacts/mdcath_stage_c_compact/endtoend_eval.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eval on {device}")

ckpt = torch.load(cp, map_location="cpu")

sb, mc, stage_b_source = load_stage_b_for_stage_c(
    ckpt,
    os.environ.get("STAGE_B_CKPT", "artifacts/mdcath_stage_b_nonoverlap/checkpoint_latest.pt"),
    device,
)
print(f"Stage-B source: {stage_b_source}")

dec, dh, _ = load_stage_c_modules(cp, device)
print("Models loaded")

manifest = load_manifest(val_mf)
dcfg = DataConfig(batch_size=1, num_workers=0, num_frames=8, frame_stride=4, window_stride=4,
                  max_residues=256, coordinate_scale=1.0, training_mode="paired_window",
                  paired_horizons_frames=[32,64,96], paired_all_horizons=True,
                  paired_history_mode="uniform")
from torch.utils.data import DataLoader
dataset = PairedTrajectoryDataset(manifest, dcfg)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=collate_paired_windows, drop_last=False)

# Compare: oracle Q vs DirectHead-predicted Q
all_oracle_rmsd, all_oracle_kerr, all_oracle_dist = [], [], []
all_dh_rmsd, all_dh_kerr, all_dh_dist = [], [], []
all_q_mse = []  # DirectHead prediction quality
count = 0

for batch in loader:
    if count >= 200: break
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

    horiz = batch.get("horizon_time_ps", torch.tensor([1000.0], device=device))
    if horiz.dim()==0: horiz = horiz.unsqueeze(0)
    temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=device))
    if temp_k.dim()>1: temp_k = temp_k[:, -1]

    with torch.no_grad():
        # CANONICAL: encode context and future through same temporal path
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

        # DirectHead prediction
        q_pred, _ = dh(h, None, horiz, temp_k, tm)
        eval_mask = batch["context_residue_mask"][:, -1] & tm
        q_mse = torch.nn.functional.mse_loss(
            q_pred[eval_mask], qo[eval_mask]
        ).item()

        # ── Oracle Q decoding ──
        out_o = dec(aatype, h, qo, R, t, tm)
        metrics_o = compute_structure_metrics(
            out_o["backbone_coords"][0], tb[0], eval_mask[0],
            residue_index=ri[0], chain_id=ci[0],
        )
        r_o = metrics_o["rmsd_kabsch"]
        kerr_o = metrics_o["kabsch_ca_point_error"]
        dm_o = metrics_o["dist_mae"]

        # ── DirectHead Q decoding ──
        out_dh = dec(aatype, h, q_pred, R, t, tm)
        metrics_dh = compute_structure_metrics(
            out_dh["backbone_coords"][0], tb[0], eval_mask[0],
            residue_index=ri[0], chain_id=ci[0],
        )
        r_dh = metrics_dh["rmsd_kabsch"]
        kerr_dh = metrics_dh["kabsch_ca_point_error"]
        dm_dh = metrics_dh["dist_mae"]

        all_oracle_rmsd.append(r_o); all_oracle_kerr.append(kerr_o); all_oracle_dist.append(dm_o)
        all_dh_rmsd.append(r_dh); all_dh_kerr.append(kerr_dh); all_dh_dist.append(dm_dh)
        all_q_mse.append(q_mse)
        count += 1
        if count % 20 == 0:
            print(f"  [{count}/200] Q_MSE={q_mse:.4f} | Oracle RMSD={r_o:.2f} KabschErr={kerr_o:.2f} | DH RMSD={r_dh:.2f} KabschErr={kerr_dh:.2f}")

mo_r = sum(all_oracle_rmsd)/len(all_oracle_rmsd)
mo_f = sum(all_oracle_kerr)/len(all_oracle_kerr)
mo_d = sum(all_oracle_dist)/len(all_oracle_dist)
mdh_r = sum(all_dh_rmsd)/len(all_dh_rmsd)
mdh_f = sum(all_dh_kerr)/len(all_dh_kerr)
mdh_d = sum(all_dh_dist)/len(all_dh_dist)
mq = sum(all_q_mse)/len(all_q_mse)

print(f"\n===== STAGE C END-TO-END ({count} samples) =====")
print(f"  DirectHead Q_MSE:                 {mq:.6f}")
print(f"")
print(f"  === Oracle Q (upper bound) ===")
print(f"  RMSD_CA (Kabsch):                  {mo_r:.2f} Å")
print(f"  Kabsch-aligned BB err (not FAPE):   {mo_f:.2f} Å")
print(f"  DistMAE_CA:                        {mo_d:.2f} Å")
print(f"")
print(f"  === DirectHead Q (end-to-end) ===")
print(f"  RMSD_CA (Kabsch):                  {mdh_r:.2f} Å")
print(f"  Kabsch-aligned BB err (not FAPE):   {mdh_f:.2f} Å")
print(f"  DistMAE_CA:                        {mdh_d:.2f} Å")

json.dump({
    "q_mse": mq,
    "oracle": {"rmsd_ca":mo_r,"kerr_kabsch":mo_f,"dist_mae_ca":mo_d},
    "directhead": {"rmsd_ca":mdh_r,"kerr_kabsch":mdh_f,"dist_mae_ca":mdh_d},
    "n_samples": count,
}, open(eval_output,"w"), indent=2)
print("Saved to endtoend_eval.json")
