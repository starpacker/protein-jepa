#!/usr/bin/env python3
"""Diagnostic: inspect parameter health of LSP-WM checkpoints (v5 / v9 / v3)."""
import sys
import torch

CKPTS = {
    "v9_s100": "artifacts/lspwm_v44_stage_b_v9_load_bath/checkpoint_step100.pt",
    "v9_s1000": "artifacts/lspwm_v44_stage_b_v9_load_bath/checkpoint_step1000.pt",
    "v9_s2000": "artifacts/lspwm_v44_stage_b_v9_load_bath/checkpoint_step2000.pt",
    "v9_s3000": "artifacts/lspwm_v44_stage_b_v9_load_bath/checkpoint_step3000.pt",
    "v9_latest": "artifacts/lspwm_v44_stage_b_v9_load_bath/checkpoint_latest.pt",
    "v5_s100": "artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_step100.pt",
    "v5_s4600": "artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_step4600.pt",
    "v5_latest": "artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_latest.pt",
    "v3_latest": "artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/checkpoint_latest.pt",
}

for name, path in CKPTS.items():
    print("=" * 90, flush=True)
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: LOAD FAIL {exc}", flush=True)
        continue
    print(name, "| top-level keys:", list(ck.keys())[:14], flush=True)
    sd = None
    for key in ("model", "state_dict", "model_state_dict", "ema_model", "ema"):
        if isinstance(ck.get(key), dict) and ck[key]:
            sd = ck[key]
            print("   using state dict under key:", key, flush=True)
            break
    scalar = {k: v for k, v in ck.items() if isinstance(v, (int, float, str))}
    if scalar:
        print("   scalars:", scalar, flush=True)
    if sd is None:
        continue
    stats = []
    for k, v in sd.items():
        if torch.is_tensor(v) and v.is_floating_point():
            vf = v.detach().float()
            stats.append((float(vf.abs().max()), float(vf.pow(2).mean().sqrt()), k))
    stats.sort(reverse=True)
    print(f"   n_params={len(sd)}  global_absmax={stats[0][0]:.6g}", flush=True)
    for mx, rms, k in stats[:8]:
        print(f"     {k[:78]:<78} absmax={mx:.6g} rms={rms:.6g}", flush=True)
