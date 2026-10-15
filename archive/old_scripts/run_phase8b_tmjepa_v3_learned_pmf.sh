#!/usr/bin/env bash
set -euo pipefail

FRAME_REPO=/liuyutian/protein_jepa_frame
CONFIG="$FRAME_REPO/configs/phase8b_tmjepa_v3_learned_pmf_balanced.yaml"
PYTHON=/liuyutian/protein_jepa/.venv/bin/python

cd "$FRAME_REPO"
exec env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$FRAME_REPO" \
  "$PYTHON" -m protein_jepa_frame.cli train --config "$CONFIG"
