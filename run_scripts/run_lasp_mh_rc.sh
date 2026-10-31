#!/bin/bash
# Lasp-MH: multi-horizon terminal training (unfrozen encoder + unfrozen decoder).
# Fixes the single-horizon H=8 Lasp's over-dispersion at long horizons: the SAME
# terminal ensemble objective is trained at each eval horizon (1/4/16/64 ns) with
# truncated BPTT (grad_steps=16) so long rollouts stay in memory.  Validation is
# the MEAN terminal loss across horizons, so "best" rewards calibration, not fit
# to one window length.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=${OUT:-artifacts/lasp_mh_rc_1000dom}
RESUME=${RESUME:-}
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-3} python -u -m protein_jepa_frame.training.train_stage_b_hs \
  --manifest artifacts/mdcath_manifest_1000dom/train.jsonl \
  --val-manifest artifacts/mdcath_manifest_1000dom/val.jsonl \
  --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
  --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
  --output "$OUT" --device cuda --torch-threads 4 \
  ${RESUME:+--resume "$RESUME"} \
  --max-train-samples 8000 --max-residues 192 --frame-stride 1 \
  --history-frames 4 --n-samples 6 \
  --multi-horizon --paired-horizons-frames 1 4 16 64 --multi-horizon-max-grad 16 \
  --ens-rmsf-corr-weight 0.5 \
  --train-encoder --encoder-lr-scale 0.1 \
  --train-decoder --decoder-lr-scale 0.05 --temperature-k 320.0 \
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.0 \
  --rollout-steps 8 --temperature-k 320.0 \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --steps 8000 --warmup-steps 200 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 250 --seed 42 \
  "$@" 2>&1 | tee .tmpwork/logs/lasp_mh_rc_1000dom.log
