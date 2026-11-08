#!/usr/bin/env bash
# "Activated-S" arm: the v3 SOTA recipe with a CORRECTLY-SCALED bath S.
#
# This is run_lasp_v3_velocity.sh (== run_lasp_v3_h_only.sh with --bath-dim 8)
# with exactly THREE scale changes, per memory lasp-bath-inert-two-scale-bug
# (the validated recipe is coupling x50-100 + gamma x0.1 + gamma_floor 2e-6):
#   * --coupling-scale 0.05   (v3 default 1e-3  -> 50x)
#   * --gamma-scale    2e-5   (v3 default 2e-4  -> 0.1x)
#   * --gamma-floor    2e-6   (v3 default 2e-5  -> 0.1x; matches exact arm)
#
# Why: v3's S is INERT (coupling_scale=1e-3, bath_var/R ~ 1e-8). The bath
# contributes ~0.1% of per-step displacement, so "v3 SOTA vs v3-bath0" cannot
# distinguish S's dynamical value.  The two-scale fix (coupling x50-100 +
# gamma x0.1) brings |C.m0|/|f| ~ O(1), i.e. the bath actually drives H.
#
# Everything else is byte-identical to v3 SOTA (seed 42 baseline + 43/44 for
# seed variance), so the ONLY axis vs the other two arms is "is the bath S
# correctly scaled":
#   arm   bath_dim  coupling_scale  gamma_scale
#   H-only    0         (n/a)         (n/a)      -> run_lasp_v3_h_only.sh
#   v3 SOTA   8         1e-3          2e-4       (inert S, the shipped baseline)
#   activated 8         0.05          2e-5       (this script: S actually drives)
#
# Honest accounting: x50 coupling may over-drive and need the force-clamp; if a
# run diverges (non-finite traj_nll) it is reported as-is, not silently retuned.
#
# Usage: SEEDS="42 43 44" OUTROOT=artifacts/lasp_v3_activated_s_20261107 \
#          bash run_scripts/run_lasp_v3_activated_s.sh
set -euo pipefail
cd /liuyutian/protein_jepa_frame

OUTROOT=${OUTROOT:-artifacts/lasp_v3_activated_s_20261107}
MANIFEST=${MANIFEST:-artifacts/mdcath_manifest_full/train.jsonl}
VAL=${VAL:-artifacts/mdcath_manifest_full/val.jsonl}
SEEDS=${SEEDS:-"42 43 44"}
GPUS=${GPUS:-"0 1"}
STEPS=${STEPS:-20000}
COUPLING_SCALE=${COUPLING_SCALE:-0.05}
GAMMA_SCALE=${GAMMA_SCALE:-2e-5}
GAMMA_FLOOR=${GAMMA_FLOOR:-2e-6}
export TMPDIR=/tmp/lasp_review_runtime
export PYTHONPATH=/tmp/lasp_stability_deps:.
mkdir -p "$OUTROOT" "$TMPDIR"

read -r -a gpu_arr <<< "$GPUS"
read -r -a seed_arr <<< "$SEEDS"

run_job() {
  local seed=$1 gpu=$2 out=$3
  echo "=== activated-S seed $seed on GPU $gpu -> $out (coupling=$COUPLING_SCALE gamma=$GAMMA_SCALE floor=$GAMMA_FLOOR) ==="
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m protein_jepa_frame.training.train_stage_b_hs \
    --manifest "$MANIFEST" --val-manifest "$VAL" \
    --stage-a-ckpt artifacts/stage_a_disp_relpos/checkpoint_latest.pt \
    --b1a-ckpt artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt \
    --output "$out" --device cuda --torch-threads 4 \
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
    --bath-coupling velocity --max-step-norm 20 --bath-dim 8 \
    --coupling-scale "$COUPLING_SCALE" --gamma-scale "$GAMMA_SCALE" --gamma-floor "$GAMMA_FLOOR" \
    --energy-depth 3 --energy-heads 4 --energy-hidden 128 \
    --steps "$STEPS" --warmup-steps 200 --lr 5e-4 --weight-decay 0.01 --grad-clip 1.0 \
    --batch-size 4 --num-workers 4 --val-every 250 --val-batches 8 \
    --log-every 10 --save-every 500 --seed "$seed" \
    > "$out.log" 2>&1
}

i=0
while [ $i -lt ${#seed_arr[@]} ]; do
  pids=()
  for ((slot=0; slot<${#gpu_arr[@]} && i<${#seed_arr[@]}; slot++)); do
    seed=${seed_arr[$i]}
    gpu=${gpu_arr[$slot]}
    out="$OUTROOT/activated_s_seed${seed}"
    i=$((i+1))
    if [ -e "$out/checkpoint_best.pt" ]; then
      echo "skip $out (exists)"
      continue
    fi
    run_job "$seed" "$gpu" "$out" &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [ "$status" -eq 0 ] || echo "WARNING: a job in this wave failed" >&2
done

echo "=== all activated-S arms finished ==="
