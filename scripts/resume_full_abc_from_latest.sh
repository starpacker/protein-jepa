#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# Continue the already-launched fresh run from its latest atomic Stage-A
# checkpoint, then run the unchanged Stage-B and Stage-C configurations.
python - <<'PY'
from protein_jepa_frame.config import load_config
from protein_jepa_frame.training.stage_a import train_stage_a

cfg = load_config("configs/full_abc_stage_a_from_scratch.yaml")
cfg.resume_checkpoint = "artifacts/full_abc_stage_a/checkpoint_latest.pt"
cfg.init_checkpoint = None
train_stage_a(cfg)
PY

python -m protein_jepa_frame.cli train --stage B \
  --config configs/full_abc_stage_b_from_scratch.yaml
python -m protein_jepa_frame.cli train --stage C \
  --config configs/full_abc_stage_c_from_scratch.yaml

echo "FULL_ABC_RESUMED_COMPLETE $(date -Is)"
