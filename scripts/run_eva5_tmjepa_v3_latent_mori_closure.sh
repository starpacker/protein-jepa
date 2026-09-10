#!/usr/bin/env bash
set -euo pipefail

ROOT=/liuyutian/protein_jepa_frame
PYTHON=/liuyutian/protein_jepa/.venv/bin/python

export PYTHONPATH="${ROOT}"
exec "${PYTHON}" -m protein_jepa_frame.cli train \
  --config "${ROOT}/configs/eva5_tmjepa_v3_latent_mori_closure.yaml"
