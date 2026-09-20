#!/bin/bash
set -uo pipefail
cd /liuyutian/protein_jepa_frame
CKPT=artifacts/lspwm_v44_stage_b_v19_bath_aware/checkpoint_latest.pt
CONFIG=artifacts/lspwm_v44_stage_b_v19_bath_aware/config.json
MANIFEST=manifests/tmjepa_v3.validation.jsonl
OUT=artifacts/reviewer_experiments/thermodynamic_closure/m1_strict_v19.json
PY=/liuyutian/protein_jepa/.venv/bin/python
echo "[M1v19] start $(date)"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/evaluate_m1_strict.py \
  --checkpoint "$CKPT" --config "$CONFIG" --manifest "$MANIFEST" \
  --output "$OUT" \
  --max-batches 50 --batch-size 32 --device cuda \
  --clip-frames 24 --window-stride 8 --history-window 8 \
  --kernel-length 4 --neural-epochs 200 --rollout-steps 256 --seed 6303 \
  && echo "[M1v19] DONE -> $OUT $(date)" || echo "[M1v19] FAILED $(date)"
