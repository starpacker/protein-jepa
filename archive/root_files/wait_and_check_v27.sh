#!/bin/bash
LOG=/liuyutian/protein_jepa_frame/artifacts/lspwm_v44_stage_b_v27_per_residue_fixed/stdout.log

# Wait until step reaches at least $1
target=${1:-3000}
echo "Waiting for step $target..."
while true; do
    last_step=$(tail -1 "$LOG" 2>/dev/null | python3 -c "import sys,json;print(json.loads(sys.stdin.read().strip()).get('step',0))" 2>/dev/null)
    if [ -n "$last_step" ] && [ "$last_step" -ge "$target" ]; then
        echo "Reached step $last_step >= $target"
        break
    fi
    sleep 60
done

python3 -c "
import json
with open('$LOG') as f:
    lines = f.readlines()
steps_above = []
for line in lines:
    try:
        d = json.loads(line.strip())
        if d.get('step', 0) >= $((target - 100)):
            steps_above.append(d)
    except: pass

print(f'\\n=== Progress around step $target ===')
for d in steps_above[-12:]:
    s = d['step']
    ramp = min(1.0, max(0, s - 3000 + 1) / 2000.0)
    contrib = 0.0005 * d.get('lspwm_per_residue_nll', 0) * ramp
    print(f'step={s:>5} loss={d[\"loss\"]:6.2f} pair={d[\"paired_skill\"]:.4f} '
          f'force={d[\"lspwm_force_loss\"]:.4f} '
          f'res_nll={d.get(\"lspwm_per_residue_nll\",0):8.1f} '
          f'ramp={ramp:.4f} contrib={contrib:.3f} '
          f'res_cos={d.get(\"lspwm_residue_force_cosine\",0):.4f} '
          f'mori={d.get(\"lspwm_mori_nll\",0):.0f}')
"
