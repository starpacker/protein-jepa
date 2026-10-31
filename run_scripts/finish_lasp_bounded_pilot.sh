#!/usr/bin/env bash
# Follow the verified session-created pilot, then audit before development eval.
set -euo pipefail
export TMPDIR=/tmp/lasp_stability_tmp
export CLAUDE_CODE_TMPDIR=/tmp/lasp_stability_runtime
export PYTHONPATH=/liuyutian/protein_jepa_frame:/tmp/lasp_stability_deps
cd /liuyutian/protein_jepa_frame
PILOT_PID=${1:?verified pilot pid required}
while kill -0 "$PILOT_PID" 2>/dev/null; do sleep 30; done
python - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/lasp_bounded_v1/pilot_seed42/metrics.jsonl')
rows=[json.loads(x) for x in p.read_text().splitlines()]
train=[r for r in rows if r.get('phase')=='train']
assert len(train)==2000 and train[-1]['step']==2000, 'pilot incomplete; inspect training failure'
assert not any('error' in r for r in rows), 'pilot contains failure'
assert all(r['gradient_elements_finite'] and r['norm64_finite'] and r['parameter_elements_finite'] for r in train)
assert any(r.get('step')==2000 and 'val' in r for r in rows), 'terminal validation missing'
print('PILOT_COMPLETE verified',flush=True)
PY
for tag in best latest; do
  python -u -m scripts.diagnose_lasp_bounded_energy \
    --checkpoint "artifacts/lasp_bounded_v1/pilot_seed42/checkpoint_$tag.pt" \
    --out "results/lasp_bounded_v1/pilot_${tag}_audit/numerical_gate.json"
done
python - <<'PY'
import json
for tag in ('best','latest'):
 d=json.load(open(f'results/lasp_bounded_v1/pilot_{tag}_audit/numerical_gate.json'))
 assert d.get('screening_passed') is True, f'{tag} numerical audit failed'
print('TRAINED_AUDITS_PASSED',flush=True)
PY
bash run_scripts/run_lasp_bounded_eval.sh pilot_seed42 artifacts/lasp_bounded_v1/pilot_seed42/checkpoint_best.pt
python -u -m scripts.compare_lasp_bounded_paired \
  --candidate results/lasp_bounded_eval/pilot_seed42/pilot_seed42_checkpoint_best.ensemble.json \
  --out results/lasp_bounded_eval/pilot_seed42/paired_standard200.json
printf 'BOUNDED_PILOT_EVALUATION_COMPLETE\n'
