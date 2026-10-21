#!/usr/bin/env bash
# Full ensemble eval for v4c once training finishes
set -euo pipefail
cd /liuyutian/protein_jepa_frame
CKPT=${1:-artifacts/fromnone/stage_cb1b_v4c/checkpoint_best.pt}
OUT=results/v4c_final_ensemble.json
echo "[eval] waiting for checkpoint: $CKPT"
while [ ! -f "$CKPT" ]; do sleep 30; done
echo "[eval] starting full eval: $CKPT"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} python -u -m evaluation.run_eval_fromnone_ensemble \
  --ckpt "$CKPT" \
  --manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --horizons-ns 1 4 16 --n-samples 32 \
  --max-proteins 20 --max-anchors-per-trajectory 2 \
  --history-frames 8 --max-residues 192 --device cuda \
  --out "$OUT" 2>&1 | tee "${OUT%.json}.log"
