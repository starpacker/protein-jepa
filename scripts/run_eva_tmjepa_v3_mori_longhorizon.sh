#!/usr/bin/env bash
set -euo pipefail

FRAME_REPO="${FRAME_REPO:-/liuyutian/protein_jepa_frame}"
PYTHON_BIN="${PYTHON_BIN:-/liuyutian/protein_jepa/.venv/bin/python}"
CONFIG="$FRAME_REPO/configs/eva_tmjepa_v3_mori_longhorizon.yaml"

cd "$FRAME_REPO"
exec env PYTHONPATH="$FRAME_REPO" "$PYTHON_BIN" -m protein_jepa_frame.cli train --config "$CONFIG"
