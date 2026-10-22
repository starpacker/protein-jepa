#!/bin/bash
# Direction 3 (short-horizon variant): B1a decided convergence test.
# mdCATH frames are 1 ns apart (frame_interval_ps=1000), so 1 ns is the finest
# resolution available. Persistence RMSD from the anchor is ~3 A even at 1 ns,
# so <1.2 A from a single-frame anchor is physically unreachable; here we
# measure how close the decoder's H-conditioned forecast gets to the floor.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=artifacts/stage_b1a_relpos_short
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-5} python -u -m protein_jepa_frame.training.train_stage_b1a \
  --manifest artifacts/mdcath_manifest_200dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_200dom/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  --steps 6000 --warmup-steps 100 --lr 0.0005 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 6 --history-frames 4 --frame-stride 1 \
  --max-residues 256 --paired-horizons-frames 1 2 3 4 \
  --unfreeze-top-n-blocks 0 --encoder-lr-scale 0.1 \
  --decoder-hidden 128 --decoder-blocks 4 --decoder-heads 4 \
  --anchor-free --coord-scale 40.0 \
  --fape-weight 1.0 --torsion-weight 0.5 --distance-weight 0.2 \
  --ca-mse-weight 1.0 --phys-future-weight 1.0 --anchor-weight 0.1 \
  --diag-zero-h --val-every 250 --val-batches 8 --fit-normalizer-batches 20 \
  --log-every 10 --save-every 250 --seed 42 \
  2>&1 | tee .tmpwork/logs/stage_b1a_relpos_short.log
