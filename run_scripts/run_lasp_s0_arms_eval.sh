#!/usr/bin/env bash
# Step 5 of LASP_CORRECT_FRAMEWORK_PLAN_20261031.md:
# harmonized standard-200 ensemble evaluation of every trained S_0 arm.
#
# Only the 200-window BioEmu2-protocol ensemble metrics are computed here.
# The per-temperature sweep / T2A suite is deliberately NOT run for the arms:
# the question these arms answer is the S_0 ablation, and the suite is ~6x the
# GPU time for metrics that cannot distinguish S_0 sources.
#
# The frozen SOTA ensemble files already exist (results/standard_eval/stopped_best)
# and are reused verbatim; SOTA is never re-evaluated.
#
# Usage: OUTROOT=artifacts/lasp_s0_arms_20261031 bash run_scripts/run_lasp_s0_arms_eval.sh
set -euo pipefail
cd /liuyutian/protein_jepa_frame

OUTROOT=${OUTROOT:-artifacts/lasp_s0_arms_20261031}
RESDIR=${RESDIR:-results/lasp_s0_arms_20261031}
GPUS=${GPUS:-"0 1"}
export TMPDIR=/tmp/lasp_review_runtime
export PYTHONPATH=/tmp/lasp_stability_deps:.
mkdir -p "$TMPDIR" "$RESDIR"

read -r -a gpu_arr <<< "$GPUS"
nper=${#gpu_arr[@]}

# Require the run to have REACHED the target step: checkpoint_best.pt is also
# written at step 0, so file existence alone would evaluate an unfinished run.
STEPS=${STEPS:-7000}
jobs=()
for d in "$OUTROOT"/{exact,prior,gru}_seed{42,43,44}; do
  if [ ! -e "$d/checkpoint_best.pt" ]; then
    echo "MISSING $d/checkpoint_best.pt"; continue
  fi
  step=$(python - "$d/checkpoint_latest.pt" <<'PY' 2>/dev/null || echo -1
import sys, torch
print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["step"]))
PY
)
  if [ "${step:--1}" -lt "$STEPS" ]; then
    echo "INCOMPLETE $d (latest step $step < $STEPS) -- refusing to evaluate"; continue
  fi
  jobs+=("$d")
done
[ "${#jobs[@]}" -gt 0 ] || { echo "no trained arms found" >&2; exit 2; }

run_eval() {
  local dir=$1 gpu=$2
  local name
  name=$(basename "$dir")
  echo "=== $name (GPU $gpu) ==="
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m scripts.eval_lasp_ca_dynamics \
    --checkpoint "$dir/checkpoint_best.pt" \
    --out "$RESDIR/$name.ensemble.json" \
    --manifest artifacts/mdcath_manifest_full/val.jsonl \
    --max-windows 200 --n-samples 12 \
    > "$RESDIR/$name.ensemble.log" 2>&1
}

i=0
while [ $i -lt ${#jobs[@]} ]; do
  pids=()
  for ((slot=0; slot<nper && i<${#jobs[@]}; slot++)); do
    dir=${jobs[$i]}
    gpu=${gpu_arr[$slot]}
    i=$((i+1))
    name=$(basename "$dir")
    if [ -e "$RESDIR/$name.ensemble.json" ]; then
      echo "skip $name (exists)"
      continue
    fi
    run_eval "$dir" "$gpu" &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  [ "$status" -eq 0 ] || { echo "ERROR: an eval job failed" >&2; exit 1; }
done

echo "=== all arm evaluations finished -> $RESDIR ==="
