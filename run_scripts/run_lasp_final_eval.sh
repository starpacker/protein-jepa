#!/bin/bash
# Three-way apples-to-apples eval: Lasp (given ckpt) vs Model 1 vs bioemu2.
# Mirrors the bioemu2 harness window-for-window (persistence verified identical).
#   CKPT=artifacts/lasp_mh_1000dom/checkpoint_best.pt OUT=results/lasp_mh_vs_bio.json \
#     bash run_scripts/run_lasp_final_eval.sh
set -euo pipefail
cd /liuyutian/protein_jepa_frame
CKPT=${CKPT:?set CKPT to a Lasp kernel checkpoint}
OUT=${OUT:?set OUT to a results json path}
GPU=${GPU:-3}
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/eval_lasp_final.py \
  --kernel-ckpt "$CKPT" \
  --ensemble-ckpt artifacts/stage_ensemble_1000dom/checkpoint_best.pt \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --device cuda --horizons-ns 1 4 16 64 --n-samples 12 --max-windows 200 \
  --out "$OUT" 2>&1 | grep -v "UserWarning\|_transformer\|return data" | tail -60
