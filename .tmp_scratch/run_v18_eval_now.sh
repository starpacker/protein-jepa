#!/bin/bash
cd /liuyutian/protein_jepa_frame
CKPT=artifacts/lspwm_v44_stage_b_v18_dmc_fix/checkpoint_latest.pt
[ -f artifacts/lspwm_v44_stage_b_v18_dmc_fix/checkpoint_best.pt ] && CKPT=artifacts/lspwm_v44_stage_b_v18_dmc_fix/checkpoint_best.pt
PRED=artifacts/lspwm_v44_stage_b_v18_dmc_fix/eval/fastfolding_predictions_fixed_readout.jsonl
LABELS=artifacts/fastfolding/frame_labels.full.jsonl
OUT=artifacts/reviewer_experiments/protein_kinetics/k1_fastfolding_qnative_v18.json
PY=/liuyutian/protein_jepa/.venv/bin/python
mkdir -p artifacts/lspwm_v44_stage_b_v18_dmc_fix/eval artifacts/reviewer_experiments/protein_kinetics
echo "[v18eval] ckpt=$CKPT  start=$(date)"
echo "[v18eval] STEP 1/2 generate predictions"
CUDA_VISIBLE_DEVICES=0 $PY scripts/generate_fastfolding_predictions.py \
  --checkpoint "v18=$CKPT" --labels "$LABELS" --output "$PRED" \
  --readout-mode fixed --ridge 1.0 --feature-source context_endpoint \
  --device cuda --seed 6303 --horizons 1 4 8 16 \
  && echo "[v18eval] predictions OK" || { echo "[v18eval] PRED FAILED"; exit 1; }
echo "[v18eval] STEP 2/2 evaluate K1"
$PY scripts/evaluate_k1_fastfolding.py \
  --predictions "$PRED" --labels "$LABELS" --output "$OUT" \
  --label-scheme q_native --q-unfolded-max 0.4 --q-folded-min 0.6 \
  --seed 6303 --horizons 1 4 8 16 \
  && echo "[v18eval] EVAL DONE -> $OUT" || { echo "[v18eval] EVAL FAILED"; exit 1; }
echo "[v18eval] ALL DONE $(date)"
