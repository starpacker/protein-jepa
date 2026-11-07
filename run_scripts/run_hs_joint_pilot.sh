#!/usr/bin/env bash
# Diagnostic, paired feedback experiment. See HS_JOINT_PILOT_20261101.md.
set -euo pipefail
cd /liuyutian/protein_jepa_frame
export PYTHONPATH=/tmp/lasp_stability_deps:.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUTROOT=${OUTROOT:-artifacts/hs_joint_20261101/pilot_v1}
STEPS=${STEPS:-500}
mkdir -p "$OUTROOT"
args=(
  --component artifacts/lasp_full_v1/representation/checkpoint_best.pt
  --init-kernel artifacts/lasp_s0_arms_20261031/exact_seed42/checkpoint_best.pt
  --collective-mobility --noise-mix 0 --history-conditioning --s0-mode exact
  --steps "$STEPS" --stateless-data --train-windows 0 --cosine --warmup 50
  --val-windows 16 --val-every 100 --n-samples 12 --grad-steps 64
  --target path_mixture --max-residues 96 --lr .0001 --encoder-lr-scale .1
  --teacher-weight 1 --teacher-delta-weight .1 --latent-weight .1
  --contact-objective brier --contact-weight 20 --rmsf-weight 1
  --clash-weight 40 --mode-weight 2 --seed 42
)
# Refuse to overwrite either arm, including a partial prior run.
for arm in frozen joint; do
  if [ -e "$OUTROOT/$arm/metrics.jsonl" ]; then
    echo "Existing run: $OUTROOT/$arm; inspect its live process and checkpoint before resuming." >&2
    exit 2
  fi
done
CUDA_VISIBLE_DEVICES=0 python -u -m protein_jepa_frame.training.train_lasp_ca_dynamics \
  "${args[@]}" --output "$OUTROOT/frozen" > "$OUTROOT/frozen.log" 2>&1 &
frozen_pid=$!
CUDA_VISIBLE_DEVICES=1 python -u -m protein_jepa_frame.training.train_lasp_ca_dynamics \
  "${args[@]}" --train-encoder --output "$OUTROOT/joint" > "$OUTROOT/joint.log" 2>&1 &
joint_pid=$!
printf '%s\n' "frozen=$frozen_pid joint=$joint_pid" > "$OUTROOT/processes.txt"
status=0
wait "$frozen_pid" || status=1
wait "$joint_pid" || status=1
exit "$status"
