#!/usr/bin/env bash
set -euo pipefail

ROOT=/liuyutian/protein_jepa_frame
PYTHON=/liuyutian/protein_jepa/.venv/bin/python
RUN="$ROOT/artifacts/phase5_v5_contextual_adaptive_2k"
CONFIG="$ROOT/configs/phase5_v5_contextual_adaptive_2k.yaml"
MANIFEST=/liuyutian/protein_jepa/artifacts/manifests/mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl

for step in 250 500 750 1000 1250 1500 1750 2000; do
    checkpoint="$RUN/checkpoint_step${step}.pt"
    while [[ ! -f "$checkpoint" ]]; do
        sleep 20
    done
    sleep 5
    PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/evaluate_v5_contextual_dynamics.py" \
        --config "$CONFIG" \
        --checkpoint "$checkpoint" \
        --manifest "$MANIFEST" \
        --output "$ROOT/artifacts/paper_eval/phase5_v5_contextual_step${step}_controls.json" \
        --max-batches 50 \
        --batch-size 1 \
        --seed 1701 \
        --device cuda:1
done
