"""End-to-end Stage A→B→C with real frozen encoders, generate backbone metrics."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np

from protein_jepa_frame.data.collate import collate_paired_windows
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.config import load_config
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_c.structure_decoder import (
    StructureDecoder, DecoderConfig,
)
from protein_jepa_frame.models.stage_c.direct_head import DirectEndpointHead, DirectHeadConfig
from protein_jepa_frame.models.stage_c.objectives import fape_loss
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from torch.utils.data import DataLoader


def _extract_frames(bb):
    return RigidBodyFrame.from_backbone(bb[...,0,:], bb[...,1,:], bb[...,2,:])


def evaluate_e2e(stage_c_config_path, stage_b_ckpt, stage_c_ckpt,
                 output_dir="artifacts/end_to_end_eval", n_batches=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_c = load_config(stage_c_config_path)
    latent_dim = cfg_c.model.latent_dim

    # Load frozen Stage B
    stage_b = StageBLASP(cfg_c.model).to(device)
    ckpt_b = torch.load(stage_b_ckpt, map_location=device, weights_only=False)
    state_b = ckpt_b.get("model", ckpt_b)
    remapped = {}
    for k, v in state_b.items():
        if k.startswith("context_encoder."):
            remapped["spatial_encoder." + k.removeprefix("context_encoder.")] = v
        elif k.startswith("temporal_encoder."):
            remapped[k] = v
    stage_b.load_state_dict(remapped, strict=False)
    stage_b.eval()
    for p in stage_b.parameters():
        p.requires_grad = False

    # Load Stage C decoder + direct head
    decoder = StructureDecoder(DecoderConfig(
        context_dim=latent_dim, q_dim=8, c_hidden=latent_dim)).to(device)
    direct_head = DirectEndpointHead(DirectHeadConfig(
        context_dim=latent_dim, condition_dim=0, hidden_dim=latent_dim)).to(device)
    ckpt_c = torch.load(stage_c_ckpt, map_location=device, weights_only=False)
    state_c = ckpt_c.get("model", ckpt_c)
    decoder.load_state_dict(state_c, strict=False)
    direct_head.load_state_dict(state_c, strict=False)
    decoder.eval()
    direct_head.eval()

    # Data
    dataset = build_dataset(cfg_c.data)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=collate_paired_windows, drop_last=True)

    all_metrics = []
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        B = batch["context_coords"].shape[0]
        L = batch["context_coords"].shape[2]

        # Frozen H from Stage B
        ctx_c = batch["context_coords"]
        ctx_rm = batch["context_residue_mask"]
        aatype = batch.get("future_aatype", batch.get("aatype"))
        if aatype.dim() == 1:
            aatype = aatype.unsqueeze(0).expand(B, -1)
        res_idx = batch.get("context_residue_index", batch.get("residue_index"))
        chain = batch.get("context_chain_id", batch.get("chain_id"))
        if res_idx is None:
            res_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        if chain is None:
            chain = torch.zeros(B, L, dtype=torch.long, device=device)
        if res_idx.dim() == 1:
            res_idx = res_idx.unsqueeze(0).expand(B, -1)
        if chain.dim() == 1:
            chain = chain.unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            temp_k = batch.get("context_temperature_k", torch.tensor([348.0], device=device))
            ctx_state = stage_b.encode_state(
                ctx_c, ctx_rm, aatype, res_idx, chain,
                time_ps=batch.get("context_time_ps"),
                temperature_k=temp_k,
                use_temporal=True,
                return_bath=False,
                return_ca_distances=False,
                return_forces=False,
            )
            h_last = ctx_state["h_last"]
            q_ctx = ctx_state["q"]

        anchor_bb = ctx_c[:, -1]
        R_anchor, t_anchor = _extract_frames(anchor_bb)
        target_bb = batch["future_coords"][:, -1]
        R_targ, t_targ = _extract_frames(target_bb)
        target_mask = batch["future_residue_mask"][:, -1]

        tau_ps = batch.get("horizon_time_ps", torch.ones(B, device=device)*1000.0)
        if tau_ps.dim() == 0:
            tau_ps = tau_ps.unsqueeze(0)
        tk = batch.get("context_temperature_k", torch.ones(B, device=device)*348.0)

        with torch.no_grad():
            qp, _ = direct_head(h_last.detach(), None, tau_ps, tk, target_mask)
            out = decoder(aatype, h_last, qp, R_anchor, t_anchor, target_mask)

            fape = fape_loss(out["backbone_coords"], target_bb,
                             pred_R=out.get("R"), pred_t=out.get("t"),
                             target_R=R_targ, target_t=t_targ, mask=target_mask)

            for b in range(B):
                mb = target_mask[b]
                ca_diff = out["CA"][b,mb] - target_bb[b,mb,1,:]
                ca_rmsd = torch.sqrt((ca_diff**2).sum(-1).mean()+1e-8)
                bb_diff = out["backbone_coords"][b,mb] - target_bb[b,mb]
                bb_rmsd = torch.sqrt((bb_diff**2).sum(-1).mean()+1e-8)
                all_metrics.append({
                    "batch": bi, "sample": b,
                    "ca_rmsd_A": float(ca_rmsd.cpu()),
                    "bb_rmsd_A": float(bb_rmsd.cpu()),
                    "fape_A": float(fape.cpu()),
                    "horizon_ps": float(tau_ps[b].cpu()),
                    "n_res": int(mb.sum().cpu()),
                })

    # Persistence baseline
    pers_metrics = []
    for bi, batch in zip(range(5), loader):
        if bi >= n_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        anchor_bb = batch["context_coords"][:, -1]
        target_bb = batch["future_coords"][:, -1]
        tm = batch["future_residue_mask"][:, -1]
        for b in range(anchor_bb.shape[0]):
            mb = tm[b]
            diff = anchor_bb[b,mb,1,:] - target_bb[b,mb,1,:]
            ca_r = torch.sqrt((diff**2).sum(-1).mean()+1e-8)
            bdiff = anchor_bb[b,mb] - target_bb[b,mb]
            bb_r = torch.sqrt((bdiff**2).sum(-1).mean()+1e-8)
            pers_metrics.append({
                "ca_rmsd_A": float(ca_r.cpu()),
                "bb_rmsd_A": float(bb_r.cpu()),
            })

    ca_model = np.array([m["ca_rmsd_A"] for m in all_metrics])
    bb_model = np.array([m["bb_rmsd_A"] for m in all_metrics])
    fape_m = np.array([m["fape_A"] for m in all_metrics])

    ca_pers = np.array([m["ca_rmsd_A"] for m in pers_metrics])
    bb_pers = np.array([m["bb_rmsd_A"] for m in pers_metrics])

    summary = {
        "n_samples": len(all_metrics),
        "model_ca_rmsd_mean": float(ca_model.mean()),
        "model_ca_rmsd_std": float(ca_model.std()),
        "model_bb_rmsd_mean": float(bb_model.mean()),
        "model_bb_rmsd_std": float(bb_model.std()),
        "fape_mean": float(fape_m.mean()),
        "fape_std": float(fape_m.std()),
        "persistence_ca_rmsd_mean": float(ca_pers.mean()),
        "persistence_bb_rmsd_mean": float(bb_pers.mean()),
        "per_sample": all_metrics,
    }

    results_path = output_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"End-to-End A→B→C Evaluation ({len(all_metrics)} samples)")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Model':>10} {'Persistence':>12}")
    print(f"{'-'*42}")
    print(f"{'CA RMSD (A)':<20} {ca_model.mean():>10.3f} {ca_pers.mean():>12.3f}")
    print(f"{'Backbone RMSD (A)':<20} {bb_model.mean():>10.3f} {bb_pers.mean():>12.3f}")
    print(f"{'FAPE (A)':<20} {fape_m.mean():>10.3f} {'N/A':>12}")
    print(f"{'='*60}")
    print(f"Results: {results_path}")
    return summary

if __name__ == "__main__":
    evaluate_e2e(
        "configs/stage_c_smoke.yaml",
        "artifacts/smoke_stage_b/checkpoint_latest.pt",
        "artifacts/smoke_stage_c/checkpoint_latest.pt",
    )
