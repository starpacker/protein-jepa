import json

data = [json.loads(l) for l in open('/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_single_step_thermodynamics_v2/train.jsonl') if l.strip()]

print(f"Total entries: {len(data)}")
print(f"Step range: {data[0]['step']} - {data[-1]['step']}")
print()

# Show relevant keys
keys = [k for k in data[0] if 'lspwm' in k or 'force' in k or 'nll' in k or 'skill' in k or 'loss' in k or 'vamp' in k or 'paired' in k or 'delta' in k]
print("Keys:", sorted(keys))
print()

# Sample data at key steps
for step in [0, 100, 500, 1000, 1500, 2000, 2500, 2990]:
    matches = [d for d in data if d['step']==step]
    m = matches[0] if matches else None
    if m:
        print(f"Step {step:5d}:")
        print(f"  NLL={m.get('lspwm_dynamic_nll', 'N/A')}, Force={m.get('lspwm_force_loss', 'N/A')}")
        print(f"  DetSkill={m.get('lspwm_deterministic_skill', 'N/A')}, EnergySkill={m.get('lspwm_endpoint_energy_skill', 'N/A')}")
        print(f"  Koopman={m.get('lspwm_koopman_loss', 'N/A')}, MSM={m.get('lspwm_msm_loss', 'N/A')}, VAMP={m.get('lspwm_sequence_vamp_loss', 'N/A')}")
        print(f"  PairedDeltaLoss={m.get('paired_delta_loss', 'N/A')}, PairedStateLoss={m.get('paired_state_loss', 'N/A')}")
        print(f"  PairedSkill={m.get('paired_skill', 'N/A')}, EMA={m.get('ema', 'N/A')}")
        print(f"  Phase={m.get('lspwm_active_phase', 'N/A')}")
        print()

# Calculate rolling statistics to measure variance
nlls = [m['lspwm_dynamic_nll'] for m in data]
skills = [m.get('lspwm_deterministic_skill', 0) for m in data]
forces = [m.get('lspwm_force_loss', 0) for m in data]

# Measure variance: std over every 20-step window
def rolling_std(values, window=5):
    stds = []
    for i in range(0, len(values)-window+1):
        stds.append(__import__('numpy').std(values[i:i+window]))
    return stds

import numpy as np
nll_stds = rolling_std(nlls, 10)
skill_stds = rolling_std(skills, 10)

print(f"NLL: min={min(nlls):.1f}, max={max(nlls):.1f}, mean={np.mean(nlls):.1f}, std={np.std(nlls):.1f}")
print(f"  Rolling std (window=10): min={min(nll_stds):.2f}, max={max(nll_stds):.2f}, mean={np.mean(nll_stds):.2f}")
print(f"DetSkill: min={min(skills):.4f}, max={max(skills):.4f}, mean={np.mean(skills):.4f}, std={np.std(skills):.4f}")
print(f"  Rolling std (window=10): min={min(skill_stds):.4f}, max={max(skill_stds):.4f}, mean={np.mean(skill_stds):.4f}")
print()

# Check the curriculum phase transition
print("Phase transitions:")
for m in data:
    if m.get('lspwm_active_phase') and 'phase' in str(m.get('lspwm_active_phase')):
        phase = m['lspwm_active_phase']
        step = m['step']
        nll = m.get('lspwm_dynamic_nll', 'N/A')
        force = m.get('lspwm_force_loss', 'N/A')
        skill = m.get('lspwm_deterministic_skill', 'N/A')
        print(f"  Step {step}: phase={phase}, NLL={nll}, Force={force}, DetSkill={skill}")

# Final comparison
print("Early vs Late comparison:")
print(f"  DetSkill first 100 steps: max={max(skills[:20]):.4f}")
print(f"  DetSkill last 100 steps: max={max(skills[-20:]):.4f}")
print(f"  NLL first 100 steps: mean={np.mean(nlls[:20]):.1f}")
print(f"  NLL last 100 steps: mean={np.mean(nlls[-20:]):.1f}")