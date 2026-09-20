#!/bin/bash
# ============================================================================
# Sequential runner: R1 → K1 → M1 (one at a time to avoid GPU OOM)
# ============================================================================
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /liuyutian/protein_jepa_frame

echo "============================================"
echo " Starting ALL reviewer experiments (R1→K1→M1)"
echo "============================================"
echo "Date: $(date)"
echo ""

echo "--- [Step 1/3] R1: Committor correlation stability ---"
echo "Start: $(date)"
/liuyutian/protein_jepa_frame/scripts/run_r1_full.sh
echo "R1 exit code: $?"
grep -E "committor_corr|its_mae" /tmp/r1_full.log | tail -5 || true

echo ""
echo "--- [Step 2/3] K1: Kinetics gate ---"
echo "Start: $(date)"
/liuyutian/protein_jepa_frame/scripts/run_k1_full.sh
echo "K1 exit code: $?"
grep -E "committor_corr|its_mae|mfpt" /tmp/k1_full.log | tail -10 || true

echo ""
echo "--- [Step 3/3] M1: Thermodynamic closure ablation ---"
echo "Start: $(date)"
/liuyutian/protein_jepa_frame/scripts/run_m1_full.sh
echo "M1 exit code: $?"
grep -E "Markov|FreeEnergy|Mori|Full|GRU|CausalTransformer" /tmp/m1_full.log | tail -20 || true

echo ""
echo "============================================"
echo " All experiments complete!"
echo " Date: $(date)"
echo "============================================"
