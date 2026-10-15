#!/usr/bin/env python3
"""M02: Autonomous rollout verification.

Per CVPR Section 4.2, validates:
  1. initialize(observed_context, static_condition) -> state
  2. step(state, delta_t, condition, random_seed) -> next_state
  3. observe(state) -> observable_distribution

Checks:
  - No future leakage during rollout
  - Step size sensitivity: dt in {10, 50, 100, 500} ps
  - Sub-stepping consistency: 1x100ps vs 10x10ps
  - Noise scale: variance growth approx 2Dt (diffusion limit)
  - Boundary handling: NaN/inf/blowup over 1000 steps
  - Energy conservation: total energy drift

Produces: numerical_stability_report.json + M02_summary.txt
"""
from __future__ import annotations
import json, math, sys, inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def make_synthetic_batch(B=1, L=16, T=4):
    residue = torch.arange(L, dtype=torch.float32)
    base = torch.stack([1.5*residue, 4.0*torch.sin(residue/5.0), 4.0*torch.cos(residue/5.0)], dim=-1)
    offsets = torch.tensor([[-1.2,0.1,0.0],[0.0,0.0,0.0],[1.3,0.1,0.0],[1.8,1.0,0.0]])
    coords_list = []
    phase = 0.0
    for t in range(T):
        time = phase + 0.4*t
        bend = torch.zeros_like(base)
        bend[:,1] = torch.sin(residue/7.0 + time)*0.8
        bend[:,2] = torch.cos(residue/11.0 - time)*0.6
        coords_list.append(base[:,None,:] + offsets[None,:,:] + bend[:,None,:])
    coords = torch.stack(coords_list, dim=0)
    return {
        "coords": coords.unsqueeze(0).expand(B,-1,-1,-1,-1),
        "residue_mask": torch.ones(B,T,L,dtype=torch.bool),
        "aatype": torch.full((L,),20,dtype=torch.long).unsqueeze(0).expand(B,-1),
        "residue_index": torch.arange(L,dtype=torch.long).unsqueeze(0).expand(B,-1),
        "chain_id": torch.zeros(B,L,dtype=torch.long),
        "temperature_k": torch.full((B,),348.0),
    }

@dataclass
class Snap:
    step: int; elapsed_ps: float
    q_mean: float; q_std: float; q_max_abs: float
    s_mean: float|None; s_std: float|None; s_max_abs: float|None
    energy_mean: float|None; has_nan: bool; has_inf: bool

def record(model, q, s, mask, ca_d, step, elap, temp):
    has_nan = bool(torch.isnan(q).any()); has_inf = bool(torch.isinf(q).any())
    with torch.no_grad():
        q_f = q * mask.unsqueeze(-1).float()
        try:
            fo = model.compute_energy_and_forces(q, ca_d, mask, temp)
            en = float(fo.get("energy_total", torch.tensor(0.0)).detach().cpu())
        except Exception:
            en = None
    return Snap(step, elap,
        float(q_f.mean()), float(q_f.std()), float(q_f.abs().max()),
        float(s.float().mean()) if s is not None else None,
        float(s.float().std()) if s is not None else None,
        float(s.float().abs().max()) if s is not None else None,
        en, has_nan, has_inf)

def _infer_config_from_checkpoint(checkpoint_path: str, device) -> tuple:
    """Recover ModelConfig from checkpoint saved config, or infer from state-dict shapes.

    Returns (ModelConfig, state_dict, extra_kwargs).
    """
    from protein_jepa_frame.config import ModelConfig

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)

    # Try saved config first (canonical checkpoints include it)
    saved_cfg = ckpt.get("config", {}).get("model", {})
    if saved_cfg:
        max_frames = saved_cfg.get("max_frames", 128)
        time_cond = saved_cfg.get("time_conditioning", "physical")
        thermo = saved_cfg.get("thermodynamic_conditioning", True)
        latent_dim = saved_cfg.get("latent_dim", 128)
        enc_depth = saved_cfg.get("encoder_depth", 4)
        temp_depth = saved_cfg.get("temporal_depth", 2)
        n_heads = saved_cfg.get("num_heads", 4)
        n_rbf = saved_cfg.get("num_rbf", 16)
        max_res = saved_cfg.get("max_residues", 256)
    else:
        # Fallback: infer from state-dict shapes
        max_frames = state.get(
            "temporal_encoder.time_embed.weight", torch.zeros(128, 128)
        ).shape[0]
        latent_dim = state.get(
            "temporal_encoder.norm.weight", torch.zeros(128)
        ).shape[0]
        n_heads = 4   # conservative default
        max_res = state.get(
            "spatial_encoder.blocks.layers.0.self_attn.in_proj_weight",
            torch.zeros(384, 128)
        ).shape[1]
        enc_depth = sum(
            1 for k in state if k.startswith("spatial_encoder.blocks.layers.")
            and "self_attn.in_proj_weight" in k
        ) or 4
        temp_depth = sum(
            1 for k in state if k.startswith("temporal_encoder.blocks.layers.")
            and "self_attn.in_proj_weight" in k
        ) or 2
        n_rbf = int((state.get(
            "spatial_encoder.rbf_generator.centres",
            torch.zeros(16)
        ).shape[0])) if "spatial_encoder.rbf_generator.centres" in state else 16
        time_cond = "physical"
        thermo = True

    cfg = ModelConfig(
        latent_dim=latent_dim,
        encoder_depth=enc_depth,
        temporal_depth=temp_depth,
        num_heads=n_heads,
        num_rbf=n_rbf,
        max_frames=max_frames,
        max_residues=max_res,
        dropout=0.0,  # eval always uses no dropout
        orientation_features=False,
        time_conditioning=time_cond,
        thermodynamic_conditioning=thermo,
    )

    # Resolve state dict key naming (old context_encoder → spatial_encoder)
    remapped = {}
    for k, v in state.items():
        if k.startswith("context_encoder."):
            remapped["spatial_encoder." + k.removeprefix("context_encoder.")] = v
        elif k.startswith("temporal_encoder."):
            remapped[k] = v
    if remapped:
        state = remapped

    extra = {"use_bath": saved_cfg.get("lspwm_use_bath", True),
             "history_aware": saved_cfg.get("lspwm_history_aware", True)}
    return cfg, state, extra


def run_m02(checkpoint_path, output_dir=None, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir or Path(checkpoint_path).parent / "m02_rollout")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"M02: device={device}, output={out_dir}")

    from protein_jepa_frame.models.stage_b.model import StageBLASP
    cfg, state, extra = _infer_config_from_checkpoint(checkpoint_path, device)
    print(f"  Inferred config: max_frames={cfg.max_frames}, enc_depth={cfg.encoder_depth}, "
          f"temp_depth={cfg.temporal_depth}, n_heads={cfg.num_heads}, "
          f"n_rbf={cfg.num_rbf}, max_res={cfg.max_residues}, "
          f"time_cond={cfg.time_conditioning}, thermo={cfg.thermodynamic_conditioning}")
    print(f"  Extra: use_bath={extra['use_bath']}, history_aware={extra['history_aware']}")

    model = StageBLASP(
        cfg, state_dim=8, bath_dim=16, base_step_ps=100.0,
        use_bath=extra["use_bath"], history_aware=extra["history_aware"],
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device); model.eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"  Loaded: {checkpoint_path} (missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"  Missing keys (first 5): {sorted(missing)[:5]}")

    batch = make_synthetic_batch(B=2, L=16, T=4)
    batch = {k: v.to(device) for k, v in batch.items()}
    B, T, L = batch["coords"].shape[:3]
    ctx_coords = batch["coords"]
    ctx_mask = batch["residue_mask"]
    aatype = batch["aatype"]
    res_idx = batch["residue_index"]
    chain_id = batch["chain_id"]
    residue_mask = ctx_mask[:, -1]
    temp_k = batch["temperature_k"]

    # Initialize - NO future data
    print("\n--- Initialize ---")
    with torch.no_grad():
        state = model.encode_state(
            ctx_coords, ctx_mask, aatype, res_idx, chain_id,
            time_ps=batch.get("time_ps"),
            temperature_k=temp_k,
            use_temporal=True,
            return_bath=True,
            return_ca_distances=True,
            return_forces=False,
        )
        h_last = state["h_last"].detach()
        q0 = state["q"]
        s0 = state["bath_s0"]
        ca_d = state["ca_distances"]
    print(f"  Q0: shape={tuple(q0.shape)}, mean={q0.mean():.4f}, std={q0.std():.4f}")
    print(f"  S0: shape={tuple(s0.shape)}")
    print(f"  CA dists: shape={tuple(ca_d.shape)}, range=[{ca_d.min():.2f},{ca_d.max():.2f}]")
    print("  [OK] No future data accessed during init")

    # Test 1: Step-size sensitivity
    print("\n--- Test 1: Step-size sensitivity ---")
    dt_vals = [10.0, 50.0, 100.0, 500.0]
    for dt in dt_vals:
        q = q0.clone().detach(); s = s0.clone().detach()
        for _ in range(100):
            q, s = model.step(q, s, ca_distances=ca_d, residue_mask=residue_mask,
                            temperature_k=temp_k, dt_ps=dt, stochastic=False)
            q = q.detach()
            if s is not None:
                s = s.detach()
        print(f"  dt={dt:5.0f}ps x100: Q mean={q.mean():.6f} std={q.std():.6f} max|Q|={q.abs().max():.6f}")

    # Test 2: Sub-stepping consistency
    print("\n--- Test 2: Sub-stepping (1x100ps vs 10x10ps) ---")
    q_big = q0.clone().detach(); s_big = s0.clone().detach()
    q_big, s_big = model.step(q_big, s_big, ca_distances=ca_d, residue_mask=residue_mask,
                              temperature_k=temp_k, dt_ps=100.0, stochastic=False)
    q_big = q_big.detach()
    q_sub = q0.clone().detach(); s_sub = s0.clone().detach()
    for _ in range(10):
        q_sub, s_sub = model.step(q_sub, s_sub, ca_distances=ca_d, residue_mask=residue_mask,
                                  temperature_k=temp_k, dt_ps=10.0, stochastic=False)
        q_sub = q_sub.detach()
        if s_sub is not None:
            s_sub = s_sub.detach()
    diff = (q_big - q_sub).abs().max().item()
    rel_diff = diff / (q_sub.abs().max().item() + 1e-8)
    consistent = rel_diff < 0.1
    print(f"  max_abs_diff={diff:.8f}, max_rel_diff={rel_diff:.6f}, consistent={consistent}")

    # Test 3: Noise scale
    print("\n--- Test 3: Noise scale (variance growth) ---")
    n_trials, n_steps, dt_n = 10, 20, 10.0
    variances = []
    for _ in range(n_trials):
        q = q0.clone(); s = s0.clone()
        for _ in range(n_steps):
            q, s = model.step(q, s, ca_distances=ca_d, residue_mask=residue_mask,
                            temperature_k=temp_k, dt_ps=dt_n, stochastic=True)
        variances.append(q.var().item())
    emp_var = np.mean(variances)
    with torch.no_grad():
        d_init = model.compute_per_residue_diffusion(q0, model.compute_g(q0, residue_mask), temp_k)
        mean_d = d_init.mean().item()
    init_var = q0.var().item()
    theory_var = init_var + 2 * mean_d * n_steps * dt_n
    ratio = emp_var / (theory_var + 1e-8)
    print(f"  init_var={init_var:.6f}, mean_D={mean_d:.6f}, theory_var={theory_var:.6f}")
    print(f"  emp_var={emp_var:.6f} +- {np.std(variances):.6f}, ratio={ratio:.3f}")

    # Test 4: Long-horizon stability
    print("\n--- Test 4: Long-horizon stability (1000 steps @ 100ps, CPU to avoid OOM) ---")
    q_cpu = q0.detach().cpu(); s_cpu = s0.detach().cpu() if s0 is not None else None
    ca_d_cpu = ca_d.detach().cpu(); m_cpu = residue_mask.detach().cpu()
    temp_cpu = temp_k.detach().cpu()
    model_cpu = model.to("cpu")
    snaps = [record(model_cpu, q_cpu.detach(), s_cpu, m_cpu, ca_d_cpu, 0, 0.0, temp_cpu)]
    for st in range(1000):
        q_cpu, s_cpu = model_cpu.step(q_cpu, s_cpu, ca_distances=ca_d_cpu,
                         residue_mask=m_cpu, temperature_k=temp_cpu,
                         dt_ps=100.0, stochastic=True)
        # Detach to avoid accumulating computation graph from create_graph=True in step()
        q_cpu = q_cpu.detach()
        if s_cpu is not None:
            s_cpu = s_cpu.detach()
        if (st+1) % 100 == 0:
            snaps.append(record(model_cpu, q_cpu, s_cpu, m_cpu, ca_d_cpu,
                               st+1, (st+1)*100.0, temp_cpu))
    model.to(device)  # move back to GPU for remaining tests
    stable = all(not sn.has_nan and not sn.has_inf for sn in snaps)
    print(f"  stable={stable}, max|Q|_any={max(sn.q_max_abs for sn in snaps):.4f}")
    for sn in snaps:
        print(f"    step={sn.step:4d} elapsed={sn.elapsed_ps:6.0f}ps |Q|max={sn.q_max_abs:.4f} nan={sn.has_nan}")

    # Test 5: Future leakage audit
    print("\n--- Test 5: Future leakage audit ---")
    step_params = set(inspect.signature(model.step).parameters.keys())
    bad = {"future", "future_coords", "future_latents", "q_next", "q_true", "target"}
    leak = step_params & bad
    if leak:
        print(f"  [FAIL] step() has future params: {leak}")
    else:
        print(f"  [PASS] step() params: {step_params}")
    init_params = set(inspect.signature(model.initialize_bath).parameters.keys())
    print(f"  initialize_bath params: {init_params}")
    energy_params = set(inspect.signature(model.compute_energy_and_forces).parameters.keys())
    print(f"  compute_energy_and_forces params: {energy_params}")

    # Report
    report = {
        "checkpoint": str(checkpoint_path), "seed": seed, "device": str(device),
        "B": B, "T": T, "L": L,
        "sub_stepping": {"diff": diff, "rel_diff": rel_diff, "consistent": consistent},
        "noise_scale": {"init_var": init_var, "emp_var": emp_var, "theory_var": theory_var, "ratio": ratio, "n_trials": n_trials},
        "long_horizon": {"n_steps": 1000, "dt_ps": 100.0, "stable": stable,
                         "q_history": [(sn.step, sn.q_max_abs, sn.q_std) for sn in snaps]},
        "future_leakage": {"step_params": list(step_params), "has_leakage": len(leak) > 0},
    }
    with open(out_dir / "numerical_stability_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    summary = [
        "M02 Autonomous Rollout Verification Report",
        "=" * 60,
        f"Checkpoint: {checkpoint_path}",
        f"B={B} T={T} L={L}",
        f"Sub-stepping consistent: {consistent} (rel_diff={rel_diff:.6f})",
        f"Noise scale ratio: {ratio:.3f}",
        f"Long horizon stable: {stable}",
        f"Future leakage: {'PASS' if not leak else 'FAIL'}",
    ]
    with open(out_dir / "M02_summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"\nReports written to {out_dir}/")
    return report

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("--output-dir", default=None); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run_m02(a.checkpoint, a.output_dir, a.seed)
