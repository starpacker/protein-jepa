#!/usr/bin/env bash
set -euo pipefail

FRAME_REPO=/liuyutian/protein_jepa_frame
CONFIG="$FRAME_REPO/configs/phase8k_tmjepa_v3_g750_coverage_calibration.yaml"
PYTHON=/liuyutian/protein_jepa/.venv/bin/python
GPU_ID=${GPU_ID:-1}

cd "$FRAME_REPO"
exec env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$FRAME_REPO" \
  "$PYTHON" -m protein_jepa_frame.cli train --config "$CONFIG"
