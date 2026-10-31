#!/bin/bash
# Lasp-v5: identical to v4 but --pos-weight 0.3 instead of 1.0.
#
# v4 (weight 1.0) at 380 steps already fixed direction -- along/true 0.05->0.50
# and median drift/true 1.34 vs v3's 1.67@250 and 2.20@1800 -- but it collapsed
# the ensemble spread to 0.49-0.91x the truth, which would give back the rmsf /
# variance_w2 metrics that the velocity bath had fixed.  v5 tests whether a
# third of the weight keeps the directional gain without paying the spread.
#
# Lasp-v4: identical to v3 (FDT-correct "velocity" bath) PLUS the atom-position
# guide (--pos-weight 0.3) that v2/v3 disable.
#
# Why: diag_gle_drift.py shows the v3 rollout has the right SPEED but the wrong
# DIRECTION -- along/true = 0.14..0.42 while perp/true = 1.0..1.6, and it gets
# worse with more steps (median drift/true 1.67 @250 steps -> 1.93 @750).
# The ensemble objective (rmsf + contact + distance histogram) scores only the
# spread and the internal geometry of the cloud; nothing scores WHERE the cloud
# sits.  corr(|D|_residue, |true step|_residue) is ~0.02 even after 1500 steps,
# i.e. the mean trajectory is simply not identified by the objective.
# _position_loss does exactly the missing thing: Kabsch-aligned CA MSE between
# the DETERMINISTIC mean decode and the TRUE terminal frame, so it constrains
# direction without suppressing per-sample variance.
#
# Lasp-v3: identical to v2 except the GLE kernel is the FDT-correct
# "velocity" bath (--bath-coupling velocity) with the saturated force clamp
# removed.  v2 uses the legacy "force" bath, which at the bath steady state
# has mobility D + C*Gamma^-1*C^T > D: the bath amplifies instead of damping.
# v3 exists to A/B exactly that change against the still-running v2.
#
# Original v2 rationale: the three fixes that target the measured root causes.
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
OUT=${OUT:-artifacts/lasp_v5_pos03}
MANIFEST=${MANIFEST:-artifacts/mdcath_manifest_full/train.jsonl}
VAL=${VAL:-artifacts/mdcath_manifest_full/val.jsonl}
RESUME=${RESUME:-}
mkdir -p "$OUT" .tmpwork/logs
CUDA_VISIBLE_DEVICES=${GPU:-4} python -u -m protein_jepa_frame.training.train_stage_b_hs \
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
  --rmsf-weight 1.0 --contact-weight 10.0 --dist-weight 0.3 --pos-weight 0.3 \
  --rollout-steps 8 \
  --bath-coupling velocity --max-step-norm 20 --bath-dim 8 --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
  --steps 20000 --warmup-steps 200 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
  --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
  --log-every 10 --save-every 500 --seed 42 \
  "$@" 2>&1 | tee .tmpwork/logs/lasp_v2_full.log
