import json

# Stage A comparison
print("=== STAGE A (Representation Baseline) ===")
try:
    with open('/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_a_representation/train.jsonl') as f:
        for line in f:
            pass
        final_a = json.loads(line.strip())
    print(f"  Final step: {final_a['step']}")
    print(f"  Deterministic Skill: {final_a.get('lspwm_deterministic_skill', 0):.4f}")
    print(f"  Dynamic NLL: {final_a.get('lspwm_dynamic_nll', 0):.1f}")
    print(f"  Force Loss: {final_a.get('lspwm_force_loss', 0):.6f}")
except FileNotFoundError:
    print("  No Stage A training log found")

# Stage B analysis
print()
print("=== STAGE B (Modified Framework) ===")
metrics = []
with open('/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_single_step_thermodynamics_v2/train.jsonl') as f:
    for line in f:
        if line.strip():
            metrics.append(json.loads(line))

print(f"  Total steps: {len(metrics)}")
print(f"  Step range: {metrics[0]['step']} - {metrics[-1]['step']}")
print()

# Best metrics
best_det = max(metrics, key=lambda m: m.get('lspwm_deterministic_skill', 0))
best_nll = min(metrics, key=lambda m: m.get('lspwm_dynamic_nll', 9999))
best_force = min(metrics, key=lambda m: m.get('lspwm_force_loss', 9999))
best_energy = max(metrics, key=lambda m: m.get('lspwm_endpoint_energy_skill', -999))
best_koop = max(metrics, key=lambda m: m.get('lspwm_koopman_loss', -999))

print("BEST METRICS:")
print(f"  DetSkill: step {best_det['step']} = {best_det['lspwm_deterministic_skill']:.4f} (NLL={best_det['lspwm_dynamic_nll']:.1f}, Force={best_det['lspwm_force_loss']:.6f})")
print(f"  NLL: step {best_nll['step']} = {best_nll['lspwm_dynamic_nll']:.1f} (DetSkill={best_nll['lspwm_deterministic_skill']:.4f})")
print(f"  Force: step {best_force['step']} = {best_force['lspwm_force_loss']:.6f}")
print(f"  EnergySkill: step {best_energy['step']} = {best_energy['lspwm_endpoint_energy_skill']:.2f}")
print()

# Step 1500
m1500 = [m for m in metrics if m['step'] == 1500][0]
print("STEP 1500 CHECKPOINT (evaluated):")
print(f"  Loss: {m1500['loss']:.4f}")
print(f"  Dynamic NLL: {m1500['lspwm_dynamic_nll']:.1f}")
print(f"  Force Loss: {m1500['lspwm_force_loss']:.6f}")
print(f"  Deterministic Skill: {m1500['lspwm_deterministic_skill']:.4f}")
print(f"  Energy Skill: {m1500['lspwm_endpoint_energy_skill']:.2f}")
print(f"  Koopman Loss: {m1500['lspwm_koopman_loss']:.3f}")
print(f"  MSM Loss: {m1500['lspwm_msm_loss']:.3f}")
print(f"  VAMP Loss: {m1500['lspwm_sequence_vamp_loss']:.3f}")
print(f"  Paired Skill: {m1500['paired_skill']:.4f}")
print(f"  EMA: {m1500['ema']:.6f}")
print()

# Early vs late comparison
print("COMPARISON: Step 0 vs Step 1500 vs Final")
s0 = metrics[0]
sfinal = metrics[-1]
print(f"  Metric        | Step 0    | Step 1500 | Final    ")
print(f"  " + "-"*50)
print(f"  Loss          | {s0['loss']:8.4f} | {m1500['loss']:8.4f} | {sfinal['loss']:8.4f}")
print(f"  NLL           | {s0['lspwm_dynamic_nll']:8.1f} | {m1500['lspwm_dynamic_nll']:8.1f} | {sfinal['lspwm_dynamic_nll']:8.1f}")
print(f"  Force         | {s0['lspwm_force_loss']:8.6f} | {m1500['lspwm_force_loss']:8.6f} | {sfinal['lspwm_force_loss']:8.6f}")
print(f"  DetSkill      | {s0['lspwm_deterministic_skill']:8.4f} | {m1500['lspwm_deterministic_skill']:8.4f} | {sfinal.get('lspwm_deterministic_skill',0):8.4f}")
print(f"  EnergySkill   | {s0['lspwm_endpoint_energy_skill']:8.2f} | {m1500['lspwm_endpoint_energy_skill']:8.2f} | {sfinal.get('lspwm_endpoint_energy_skill',0):8.2f}")
print()

# Convergence analysis
print("CONVERGENCE ANALYSIS:")
nlls = [m['lspwm_dynamic_nll'] for m in metrics]
force = [m['lspwm_force_loss'] for m in metrics]
skills = [m.get('lspwm_deterministic_skill', 0) for m in metrics]
print(f"  NLL: {nlls[0]:.1f} -> min={min(nlls):.1f} -> final={nlls[-1]:.1f}")
print(f"  Force: {force[0]:.6f} -> min={min(force):.6f} -> final={force[-1]:.6f}")
print(f"  DetSkill: {skills[0]:.4f} -> max={max(skills):.4f} -> final={skills[-1]:.4f}")
print()

# Overfitting check
print("OVERFITTING CHECK:")
last_third = metrics[len(metrics)//3*2:]
best_skill_last = max(m.get('lspwm_deterministic_skill', 0) for m in last_third)
print(f"  Best DetSkill in last 1/3: {best_skill_last:.4f}")
early_best = max(m.get('lspwm_deterministic_skill', 0) for m in metrics[:len(metrics)//3])
print(f"  Best DetSkill in first 1/3: {early_best:.4f}")
if best_skill_last < early_best * 0.8:
    print("  WARNING: DetSkill degraded significantly (overfitting)")
else:
    print("  OK: No significant overfitting")