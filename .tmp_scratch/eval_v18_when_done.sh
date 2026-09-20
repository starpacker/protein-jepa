#!/bin/bash
cd /liuyutian/protein_jepa_frame
V18_PID=1250821
CKPT=artifacts/lspwm_v44_stage_b_v18_dmc_fix/checkpoint_latest.pt
PRED=artifacts/lspwm_v44_stage_b_v18_dmc_fix/eval/fastfolding_predictions_fixed_readout.jsonl
LABELS=artifacts/fastfolding/frame_labels.full.jsonl
OUT=artifacts/reviewer_experiments/protein_kinetics/k1_fastfolding_qnative_v18.json
PY=/liuyutian/protein_jepa/.venv/bin/python

echo "[watcher] waiting for v18 PID $V18_PID to finish..."
while kill -0 $V18_PID 2>/dev/null; do sleep 30; done
echo "[watcher] v18 finished at $(date). Starting evaluation."

# use best/latest checkpoint (prefer best if exists)
if [ -f artifacts/lspwm_v44_stage_b_v18_dmc_fix/checkpoint_best.pt ]; then
  CKPT=artifacts/lspwm_v44_stage_b_v18_dmc_fix/checkpoint_best.pt
fi
echo "[watcher] using checkpoint: $CKPT"

echo "[watcher] STEP 1/2: generate predictions (GPU 0 now free)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/generate_fastfolding_predictions.py \
  --checkpoint "v18=$CKPT" \
  --labels "$LABELS" \
  --output "$PRED" \
  --readout-mode fixed --ridge 1.0 --feature-source context_endpoint \
  --device cuda --seed 6303 --horizons 1 4 8 16 \
  && echo "[watcher] predictions done" || { echo "[watcher] PRED FAILED"; exit 1; }

echo "[watcher] STEP 2/2: evaluate K1 (symmetric ITS)"
$PY scripts/evaluate_k1_fastfolding.py \
  --predictions "$PRED" \
  --labels "$LABELS" \
  --output "$OUT" \
  --label-scheme q_native --q-unfolded-max 0.4 --q-folded-min 0.6 \
  --seed 6303 --horizons 1 4 8 16 \
  && echo "[watcher] EVAL DONE -> $OUT" || { echo "[watcher] EVAL FAILED"; exit 1; }

echo "[watcher] ALL DONE at $(date)"
