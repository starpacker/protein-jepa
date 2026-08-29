#!/usr/bin/env bash
set -euo pipefail

ROOT=/liuyutian/protein_jepa_frame
PYTHON=/liuyutian/protein_jepa/.venv/bin/python
RUN="$ROOT/artifacts/phase4_v4_d2_reversible_kinetics_1k"
CONFIG="$ROOT/configs/phase4_v4_d2_reversible_kinetics_1k.yaml"
MANIFEST=/liuyutian/protein_jepa/artifacts/manifests/mdcath.deeptime_lcc_pcca_v2.full_lag1.validation.filtered.jsonl

for step in 250 500 750 1000; do
    checkpoint="$RUN/checkpoint_step${step}.pt"
    while [[ ! -f "$checkpoint" ]]; do
        sleep 15
    done
    sleep 5
    PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/evaluate_paired_dynamics.py" \
        --config "$CONFIG" \
        --checkpoint "$checkpoint" \
        --manifest "$MANIFEST" \
        --output "$ROOT/artifacts/paper_eval/phase4_v4_d2_reversible_step${step}_controls.json" \
        --max-batches 50 \
        --batch-size 2 \
        --seed 1701 \
        --device cuda:0
done
