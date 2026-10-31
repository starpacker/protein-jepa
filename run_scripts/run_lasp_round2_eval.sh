#!/usr/bin/env bash
# Round 2 of the extended evaluation: wait for the full dynamics schedule to
# finish, then run the standard suite (per-temperature + T3 matrix + T2A) on the
# SELECTED (best-validation) checkpoint of every seed.
#
# Round 1 used the frozen best snapshots taken mid-run
# (results/lasp_full_v1/mid_suite/seed{42,43,44}.pt). Round 2 uses
# artifacts/lasp_full_v1/dynamics_seed*/checkpoint_best.pt, i.e. the checkpoint
# the frozen recipe's selection rule actually picks, so the two rounds differ in
# the training budget and not in the selection rule.
#
# The pipeline (scripts/run_lasp_full_pipeline.py) is still running its own
# post-training 200-window comparison when this starts. That comparison is the
# BioEmu2 paired protocol; this script adds the per-temperature / T3 / T2A
# families. Both may run concurrently.
#
# Usage: bash run_scripts/run_lasp_round2_eval.sh
set -euo pipefail
cd /liuyutian/protein_jepa_frame

ROOT=${ROOT:-artifacts/lasp_full_v1}
SEEDS=${SEEDS:-"42 43 44"}
GPUS=${GPUS:-"2 5 7"}
POLL=${POLL:-120}
# Must match the frozen recipe's --steps (scripts/run_lasp_full_pipeline.py).
TARGET_STEPS=${TARGET_STEPS:-30000}
seeds_total=$(echo "$SEEDS" | wc -w)

echo "=== waiting for all dynamics seeds to reach the full schedule ==="
while :; do
  done_count=0
  for s in $SEEDS; do
    ck="$ROOT/dynamics_seed$s/checkpoint_latest.pt"
    [ -f "$ck" ] || continue
    step=$(python - "$ck" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
print(int(ck.get('step', 0)))
PY
)
    if [ "$step" -ge "$TARGET_STEPS" ]; then
      done_count=$((done_count + 1))
    fi
  done
  echo "  seeds at final step: $done_count/${seeds_total}"
  if [ "$done_count" -ge "$seeds_total" ]; then break; fi
  sleep "$POLL"
done

ckpts=()
for s in $SEEDS; do
  ck="$ROOT/dynamics_seed$s/checkpoint_best.pt"
  [ -f "$ck" ] || { echo "missing $ck" >&2; exit 1; }
  ckpts+=("$ck")
done

echo "=== round 2 standard evaluation on selected checkpoints ==="
GPUS="$GPUS" bash run_scripts/run_lasp_standard_eval.sh round2 "${ckpts[@]}"
echo "=== round 2 complete ==="