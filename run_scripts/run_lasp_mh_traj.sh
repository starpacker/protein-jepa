#!/bin/bash
# Lasp-MH-Traj: multi-horizon TERMINAL ensemble loss + trajectory-level
# DISTRIBUTIONAL losses (contiguous path), encoder+decoder unfrozen.
#   * multi-horizon fixes the long-horizon over-dispersion (report §3.9)
#   * trajectory rmsf_corr (well-posed on the path, unlike the degenerate
#     T=1 terminal version) targets the RMSF-PROFILE metric bioemu2 wins.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=${OUT:-artifacts/lasp_mh_traj_1000dom}
RESUME=${RESUME:-}
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-4} python -u -m protein_jepa_frame.training.train_stage_b_hs \
  --manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  ${RESUME:+--resume "$RESUME"} \
  --max-train-samples 8000 --max-residues 192 --frame-stride 1 \
  --history-frames 4 --n-samples 6 \
  --multi-horizon --paired-horizons-frames 1 4 16 64 --multi-horizon-max-grad 16 \
  --traj-loss --use-contiguous-window \
  --traj-es-weight 0.1 --traj-rmsf-corr-weight 0.5 --traj-anchor-loss \
  --train-encoder --encoder-lr-scale 0.1 \
  --train-decoder --decoder-lr-scale 0.05 --temperature-k 320.0 \
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.0 \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --steps 8000 --warmup-steps 200 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 250 --seed 42 \
  "$@" 2>&1 | tee .tmpwork/logs/lasp_mh_traj_1000dom.log
