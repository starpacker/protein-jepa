#!/bin/bash
# Plan A: Stage-A encoder (warm start) + GLE kernel, JOINTLY fine-tuned, frozen B1a decoder.
#
# Why this differs from run_stage_hs_1000dom.sh (the version that diverged):
#   1) --train-encoder: Stage A's H already scores well on the ensemble metrics and
#      is linearly decodable to CA displacement (ridge R^2 ~ +0.29), so we keep it
#      and fine-tune at low LR instead of freezing it against a kernel that can only
#      hurt it.  The decoder stays frozen (Model 1's B1a decoder).
#   2) --rollout-steps == --paired-horizons-frames: the old run optimised a 1-step
#      rollout against a whole-window ensemble target, so the multi-step C@S bath
#      feedback was never in the loss and training drove it into a divergent regime
#      (measured: ||H_rollout-H0|| 33 -> 9220 for 1 -> 4 steps).  Matching the step
#      count to the window keeps the rollout the loss sees equal to the one used at
#      inference.
set -e
cd /liuyutian/protein_jepa_frame
export TMPDIR=/liuyutian/protein_jepa_frame/.tmpwork
OUT=${OUT:-artifacts/plan_a_hs_1000dom}
RESUME=${RESUME:-}
H=${H:-8}                    # future-window frames == rollout steps
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
  --paired-horizons-frames "$H" \
  --train-encoder --encoder-lr-scale 0.1 \
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.0 \
  --rollout-steps "$H" --rollout-grad-steps "$H" --temperature-k 320.0 \
  --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --steps 6000 --warmup-steps 200 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 250 --seed 42 \
  "$@" 2>&1 | tee .tmpwork/logs/plan_a_hs_1000dom.log
