#!/usr/bin/env bash
# Whole five-method sweep, one method per GPU, five GPUs in parallel.
set -uo pipefail
cd /liuyutian/protein_jepa_frame
CELLS=${CELLS:-"val_t320 val_t348 val_t379 val_t413 val_t450 train_t348 train_t450"}
mkdir -p logs/five_method results/five_method
EXTRA="--temp-split" bash run_scripts/run_five_method_ext.sh 2 model1 $CELLS > logs/five_method/l1.log 2>&1 &
EXTRA="--temp-split" bash run_scripts/run_five_method_ext.sh 5 model2 $CELLS > logs/five_method/l2.log 2>&1 &
EXTRA=""             bash run_scripts/run_five_method_ext.sh 6 lasp   $CELLS > logs/five_method/l3.log 2>&1 &
EXTRA=""             bash run_scripts/run_five_method_ext.sh 7 mdgen  $CELLS > logs/five_method/l4.log 2>&1 &
EXTRA=""             bash run_scripts/run_five_method_ext.sh 0 bioemu2 $CELLS > logs/five_method/l5.log 2>&1 &
wait
echo "sweep done"
