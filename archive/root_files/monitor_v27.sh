#!/bin/bash
LOG=/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v27_per_residue_fixed/stdout.log
python3 -c "
import json
with open('$LOG') as f:
    lines = f.readlines()
# Find steps after 2400
steps = {}
for line in lines:
    try:
        d = json.loads(line.strip())
        if 'step' in d and 'loss' in d and d.get('step', 0) >= 2400:
            s = d['step']
            if s not in steps or d.get('epoch', 0) >= steps[s].get('epoch', 0):
                steps[s] = d
    except: pass
for s in sorted(steps.keys())[-10:]:
    d = steps[s]
    ramp = min(1.0, max(0, d['step'] - 3000 + 1) / 2000.0)
    phase = d.get('lspwm_active_phase', '?')
    print(f'step={s:>5} epoch={d.get(\"epoch\",\"?\")} phase={phase:>12} loss={d[\"loss\"]:6.2f} '
          f'pair={d[\"paired_skill\"]:.4f} force={d[\"lspwm_force_loss\"]:.4f} '
          f'res_nll={d.get(\"lspwm_per_residue_nll\",0):8.1f} ramp={ramp:.4f} '
          f'res_cos={d.get(\"lspwm_residue_force_cosine\",0):.4f} '
          f'dyn={d.get(\"lspwm_dynamic_nll\",0):.2f} mori={d.get(\"lspwm_mori_nll\",0):.0f}')
"
