#!/usr/bin/env bash
# Print the current status of the from-none training runs.
cd /liuyutian/protein_jepa_frame
for run in stage_cb1b_main stage_cb1b_temp; do
  log="artifacts/fromnone/$run/train.log"
  echo "=== $run ==="
  if [[ -f "$log" ]]; then
    echo "last step: $(grep -oE 'step [0-9]+' "$log" | tail -1)"
    echo "last validation: $(grep 'validation' "$log" | tail -1)"
  else
    echo "(no log)"
  fi
done
echo "=== ckpts ==="
ls -la --time-style=+%H:%M artifacts/fromnone/*/checkpoint_best.pt 2>/dev/null
