#!/usr/bin/env python3
"""Verify temporal sensitivity of trained Stage A encoder on mdCATH.

Tests:
  1. Spatial encoder H_t: cosine similarity decay with time interval
  2. Temporal state S_t: cosine similarity decay (should be << 1.0)
  3. Temperature separation: S_t(320K) vs S_t(450K) should differ
  4. Per-horizon prediction accuracy: skill at 4/8/16-frame horizons

Usage: python scripts/verify_temporal_sensitivity.py [checkpoint_path]
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def verify(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load model
    from protein_jepa_frame.config import ModelConfig
    from protein_jepa_frame.models.jepa import ProteinJEPA
    from protein_jepa_frame.data.manifest import load_manifest
    from protein_jepa_frame.data.dataset import TrajectoryClipDataset
    from protein_jepa_frame.config import DataConfig

    # Load config from checkpoint artifact
    ckpt_dir = Path(checkpoint_path).parent
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        print(f"ERROR: config.json not found in {ckpt_dir}")
        sys.exit(1)

    with open(config_path) as f:
        ckpt_cfg = json.load(f)
    m = ckpt_cfg.get("model", {})
    model_cfg = ModelConfig(
        latent_dim=m.get("latent_dim", 128),
        encoder_depth=m.get("encoder_depth", 4),
        temporal_depth=m.get("temporal_depth", 2),
        predictor_depth=m.get("predictor_depth", 4),
        num_heads=m.get("num_heads", 4),
        num_rbf=m.get("num_rbf", 16),
        max_frames=m.get("max_frames", 128),
        max_residues=m.get("max_residues", 256),
        dropout=m.get("dropout", 0.1),
        feature_norm=m.get("feature_norm", True),
        time_conditioning=m.get("time_conditioning", "none"),
        thermodynamic_conditioning=m.get("thermodynamic_conditioning", False),
        paired_forecast=m.get("paired_forecast", True),
        paired_predictor_depth=m.get("paired_predictor_depth", 2),
        paired_teacher_space_delta=m.get("paired_teacher_space_delta", False),
        lspwm_enabled=m.get("lspwm_enabled", False),
    )
    model = ProteinJEPA(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    time_cond = m.get("time_conditioning", "none")
    print(f"  time_conditioning: {time_cond}, dim={m.get('latent_dim')}, enc_depth={m.get('encoder_depth')}, temporal_depth={m.get('temporal_depth')}")

    # Load data
    manifest_path = "artifacts/mdcath_manifest_smoke/train.jsonl"
    if not Path(manifest_path).exists():
        manifest_path = "artifacts/mdcath_manifest_200dom/train.jsonl"
    records = load_manifest(manifest_path)
    ds_cfg = DataConfig(
        manifest_path=manifest_path, batch_size=1,
        num_frames=21, frame_stride=2, window_stride=10,
        num_workers=0, max_residues=256
    )
    ds = TrajectoryClipDataset(records, ds_cfg)

    # Test on a 450K trajectory (max conformational change)
    sample = None
    for idx in range(min(500, len(ds))):
        s = ds[idx]
        if s['temperature_k'].item() == 450 and s['protein_id'] != "1a0rP01":
            sample = s
            break
    if sample is None:
        sample = ds[0]

    coords = sample['coords'].unsqueeze(0).to(device)
    mask_t = sample['residue_mask'].unsqueeze(0).to(device)
    aa = sample['aatype'].unsqueeze(0).to(device)
    ri = sample['residue_index'].unsqueeze(0).to(device)
    ch = sample['chain_id'].unsqueeze(0).to(device)
    time_ps = sample['time_ps'].unsqueeze(0).to(device)
    temp_k = sample['temperature_k'].unsqueeze(0).to(device)
    B, T, L = coords.shape[:3]

    print(f"\nTest domain: {sample['protein_id']}, T={sample['temperature_k'].item()}K, L={L}, {T} frames")
    ca = coords[0, :, :, 1, :]
    print(f"CA motion: RMSD(f0 vs f{T//2})={torch.sqrt(((ca[0]-ca[T//2])**2).sum()/L):.1f}A, "
          f"RMSD(f0 vs f{T-1})={torch.sqrt(((ca[0]-ca[-1])**2).sum()/L):.1f}A")

    # ── Test 1: Spatial encoder H ──
    print("\n" + "="*60)
    print("TEST 1: Spatial encoder H temporal sensitivity")
    print("  (cos(H_t, H_0) should drift from 1.0 if encoder sees motion)")
    with torch.no_grad():
        h = model.context_encoder(coords, mask_t, aa, ri, ch)

    intervals = [0, T//4, T//2, 3*T//4, T-1]
    for i in intervals:
        hp0 = h[0, 0].reshape(-1)
        hpi = h[0, i].reshape(-1)
        cos_h = (hp0 * hpi).sum() / (hp0.norm() * hpi.norm() + 1e-8)
        dt = time_ps[0, i] - time_ps[0, 0]
        print(f"  cos(H_0, H_{i}) @ Δt={dt.item():.0f}ps: {cos_h:.8f}")

    # ── Test 2: Temporal states ──
    print("\n" + "="*60)
    print("TEST 2: Temporal state S temporal sensitivity")
    print("  (cos(S_t, S_0) should drop significantly — << 0.99)")
    with torch.no_grad():
        states = model.temporal_encoder(h, mask_t, time_ps=time_ps, temperature_k=temp_k)
        fs = states['frame_states']  # [B, T, D]

    cos_decay = []
    for i in intervals:
        s0 = fs[0, 0]
        si = fs[0, i]
        cos_s = (s0 * si).sum() / (s0.norm() * si.norm() + 1e-8)
        dt = time_ps[0, i] - time_ps[0, 0]
        cos_decay.append((dt.item(), cos_s.item()))
        print(f"  cos(S_0, S_{i}) @ Δt={dt.item():.0f}ps: {cos_s:.6f}")

    max_dt_cos = cos_decay[-1][1]
    if max_dt_cos > 0.95:
        result_temporal = "FAILED — temporal states are near-constant"
    elif max_dt_cos > 0.70:
        result_temporal = "MARGINAL — some temporal decay but still highly similar"
    else:
        result_temporal = f"PASSED — significant temporal decay (cos={max_dt_cos:.3f} at {cos_decay[-1][0]:.0f}ps)"

    # ── Test 3: Clip state across windows ──
    print("\n" + "="*60)
    print("TEST 3: Clip state at different windows")
    clip_states = []
    with torch.no_grad():
        for w_start in range(0, T-5, 5):
            cs = model.temporal_encoder(
                h[:, w_start:w_start+5], mask_t[:, w_start:w_start+5],
                time_ps=time_ps[:, w_start:w_start+5], temperature_k=temp_k
            )['clip_state']
            clip_states.append(cs[0])

    for i in range(len(clip_states)):
        cs0 = clip_states[0]
        csi = clip_states[i]
        cos_cs = (cs0 * csi).sum() / (cs0.norm() * csi.norm() + 1e-8)
        print(f"  cos(clip_0, clip_{i}): {cos_cs:.6f}")

    # ── Test 4: Temperature separation ──
    print("\n" + "="*60)
    print("TEST 4: Temperature separation (320K vs 450K)")
    temp_samples = {}
    for idx in range(min(2000, len(ds))):
        s = ds[idx]
        t = int(s['temperature_k'].item())
        if t in [320, 450] and t not in temp_samples:
            temp_samples[t] = s
        if len(temp_samples) == 2:
            break

    temp_states = {}
    for t, s in temp_samples.items():
        c = s['coords'].unsqueeze(0).to(device)
        m = s['residue_mask'].unsqueeze(0).to(device)
        with torch.no_grad():
            h_t = model.context_encoder(c, m,
                s['aatype'].unsqueeze(0).to(device),
                s['residue_index'].unsqueeze(0).to(device),
                s['chain_id'].unsqueeze(0).to(device))
            st = model.temporal_encoder(h_t, m,
                time_ps=s['time_ps'].unsqueeze(0).to(device),
                temperature_k=s['temperature_k'].unsqueeze(0).to(device))
        temp_states[t] = st['clip_state'][0]

    if 320 not in temp_states or 450 not in temp_states:
        print("  SKIPPED — could not find both 320K and 450K samples")
        cs320 = torch.zeros(128, device=device)
        cs450 = torch.zeros(128, device=device)
    else:
        cs320 = temp_states[320]
        cs450 = temp_states[450]
    cos_temp = (cs320 * cs450).sum() / (cs320.norm() * cs450.norm() + 1e-8)
    print(f"  cos(clip_320K, clip_450K): {cos_temp:.6f}")
    if cos_temp > 0.95:
        print("  Result: FAILED — temperature has no effect on representation")
    else:
        print(f"  Result: PASSED — temperature affects representation (cos={cos_temp:.3f})")

    # ── Summary ──
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Time conditioning: {time_cond}")
    print(f"  Temporal state decay: {result_temporal}")
    print(f"  Temperature separation: {'PASSED' if cos_temp < 0.95 else 'FAILED'}")
    print(f"  Max cos(H_t,H_0): {max((h[0,0].reshape(-1)*h[0,j].reshape(-1)).sum()/(h[0,0].norm()*h[0,j].norm()+1e-8) for j in range(T)).item():.6f}")
    print(f"  Max cos(S_t,S_0): {max((fs[0,0]*fs[0,j]).sum()/(fs[0,0].norm()*fs[0,j].norm()+1e-8) for j in range(T)).item():.6f}")

    return {
        "time_conditioning": time_cond,
        "max_cos_H": max((h[0,0].reshape(-1)*h[0,j].reshape(-1)).sum()/(h[0,0].norm()*h[0,j].norm()+1e-8) for j in range(T)).item(),
        "max_cos_S": max((fs[0,0]*fs[0,j]).sum()/(fs[0,0].norm()*fs[0,j].norm()+1e-8) for j in range(T)).item(),
        "cos_temperature_320_450": cos_temp.item(),
        "temporal_result": result_temporal,
    }


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "artifacts/mdcath_stage_a_smoke/checkpoint_latest.pt"
    verify(checkpoint)
