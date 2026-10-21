#!/bin/bash
# Direction 3 (ANCHORED decoder): the anchor-free decoder overfits train and
# fails to generalize (val real-H ~27 A >> persist ~3.7 A) because it must
# build the fold from scratch from H alone, which H cannot carry (§2).
# The anchored decoder starts from the anchor's ACTUAL per-residue CA positions
# (persistence baseline) and predicts only the bounded conformational change
# d_local (anchor-local, <=12 A) -- the true forecast task.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=artifacts/stage_b1a_relpos_anchored
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-5} python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  --steps 6000 --warmup-steps 100 --lr 0.0005 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 6 --history-frames 4 --frame-stride 1 \
  --max-residues 256 --paired-horizons-frames 1 2 4 8 \
  --unfreeze-top-n-blocks 0 --encoder-lr-scale 0.1 \
  --decoder-hidden 128 --decoder-blocks 4 --decoder-heads 4 \
  --no-anchor-free --local-position-budget 12.0 \
  --relative-disp-weight 1.0 --ca-mse-weight 1.0 \
  --fape-weight 1.0 --torsion-weight 0.5 --distance-weight 0.2 \
  --phys-future-weight 1.0 --anchor-weight 0.1 \
  --diag-zero-h --val-every 250 --val-batches 8 --fit-normalizer-batches 20 \
  --log-every 10 --save-every 250 --seed 42 \
  2>&1 | tee .tmpwork/logs/stage_b1a_relpos_anchored.log
