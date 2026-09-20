#!/usr/bin/env python3
"""Diagnostic: inspect LSP-WM parameter health of v5 / v9 / v3 checkpoints."""
import torch

CKPTS = {
    "v3_latest": "artifacts/lspwm_v44_stage_b_v3_improved_thermodynamics/checkpoint_latest.pt",
    "v5_s100": "artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_step100.pt",
    "v5_latest": "artifacts/lspwm_v44_stage_b_v5_fixed_pmf/checkpoint_latest.pt",
    "v9_s100": "artifacts/lspwm_v44_stage_b_v9_load_bath/checkpoint_step100.pt",
    "v9_s1800": "artifacts/lspwm_v44_stage_b_v9_load_bath/checkpoint_step1800.pt",
    "v9_latest": "artifacts/lspwm_v44_stage_b_v9_load_bath/checkpoint_latest.pt",
}

for name, path in CKPTS.items():
    print("=" * 95, flush=True)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    print(f"{name} | step={ck.get('step')} epoch={ck.get('epoch')}", flush=True)
    bad = [
        k for k, v in sd.items()
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()
    ]
    print(f"   non-finite tensors: {len(bad)} {bad[:6]}", flush=True)
    rows = []
    for k, v in sd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        if "lspwm" not in k and "paired" not in k:
            continue
        vf = v.detach().float()
        rows.append((float(vf.norm()), float(vf.abs().max()), k))
    rows.sort(reverse=True)
    print(f"   lspwm/paired tensors: {len(rows)}. Largest by norm:", flush=True)
    for nrm, mx, k in rows[:14]:
        print(f"     {k[:72]:<72} L2={nrm:.6g} absmax={mx:.6g}", flush=True)
    tot = sum(r[0] ** 2 for r in rows) ** 0.5
    print(f"   total L2 of lspwm+paired params: {tot:.6g}", flush=True)
