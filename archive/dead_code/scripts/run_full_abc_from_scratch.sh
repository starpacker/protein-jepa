#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

python -m protein_jepa_frame.cli train --stage A \
  --config configs/full_abc_stage_a_from_scratch.yaml
python -m protein_jepa_frame.cli train --stage B \
  --config configs/full_abc_stage_b_from_scratch.yaml
python -m protein_jepa_frame.cli train --stage C \
  --config configs/full_abc_stage_c_from_scratch.yaml

echo "FULL_ABC_COMPLETE $(date -Is)"
