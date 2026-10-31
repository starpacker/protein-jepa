#!/bin/bash
# Lasp-v2: the three fixes that target the measured root causes.
#
#   1. --per-sample-temperature : mdCATH spans 320/348/379/413/450 K but every
#      batch used to be conditioned on one constant, so 4/5 of the data was
#      mislabelled and the kernel learned to ignore temperature.
#   2. --roundtrip-weight       : Enc(true X_t) -> Dec -> X_t.  Measured on the
#      1000dom checkpoint, enc->dec recovers only 31-35 % of the true RMSF
#      (RMSF-r 0.29-0.65 across horizons), i.e. the H->CA map discards the very
#      conformational differences the metrics score.  This is NOT the reverted
#      --dense-trajectory loss: the GLE rollout is untouched.
#   3. full mdCATH manifest      : 5398 domains instead of 800, with the SAME
#      val/test split copied verbatim so every number stays comparable.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=${OUT:-artifacts/lasp_v2_full}
MANIFEST=${MANIFEST:-artifacts/mdcath_manifest_full/train.jsonl}
VAL=${VAL:-artifacts/mdcath_manifest_full/val.jsonl}
RESUME=${RESUME:-}
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-3} python -u -m protein_jepa_frame.training.train_stage_b_hs \
  --manifest "$MANIFEST" --val-manifest "$VAL" \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  ${RESUME:+--resume "$RESUME"} \
  --max-train-samples 200000 --max-residues 192 --frame-stride 1 \
  --history-frames 4 --n-samples 6 \
  --multi-horizon --paired-horizons-frames 1 4 16 64 --multi-horizon-max-grad 16 \
  --ens-rmsf-corr-weight 0.5 \
  --per-sample-temperature \
  --train-encoder --encoder-lr-scale 0.1 \
  --train-decoder --decoder-lr-scale 0.1 \
  --roundtrip-weight 0.5 --roundtrip-frames 2 \
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.0 \
  --rollout-steps 8 \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --steps 20000 --warmup-steps 200 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 500 --seed 42 \
  "$@" 2>&1 | tee .tmpwork/logs/lasp_v2_full.log
