#!/usr/bin/env bash
set -euo pipefail
cd /liuyutian/protein_jepa_frame
TAG=${1:?usage: run_lasp_bounded_eval.sh <tag> <checkpoint...>}
shift
[ "$#" -gt 0 ] || { echo 'provide bounded checkpoints' >&2; exit 2; }
OUTDIR=${OUTDIR:-results/lasp_bounded_eval/$TAG}
mkdir -p "$OUTDIR"
for ck in "$@"; do
  name=$(basename "$(dirname "$ck")")_$(basename "$ck" .pt)
  echo "=== bounded endpoint standard evaluation: $name ==="
  python -u -m scripts.eval_lasp_ca_dynamics --checkpoint "$ck" --out "$OUTDIR/$name.ensemble.json" --manifest artifacts/mdcath_manifest_full/val.jsonl --max-windows 200 --n-samples 12
  python -u -m scripts.eval_lasp_ca_suite --checkpoint "$ck" --out "$OUTDIR/$name.suite.json" --t2a --modes correct wrong none --wrong-temp 320 --proteins-per-temp 12 --anchors-per-trajectory 2 --n-samples 12 --horizons 1 4 16 64 --max-residues 96
  python -u -m scripts.eval_lasp_ca_suite --checkpoint "$ck" --out "$OUTDIR/$name.matched8.json" --t2a --manifest artifacts/mdcath_manifest_full/val.jsonl --modes correct --proteins-per-temp 8 --anchors-per-trajectory 1 --n-samples 12 --context-frames 8 --horizons 1 4 16 64 --max-residues 96
done
python -u -m scripts.report_lasp_ca_suite "$OUTDIR"/*.suite.json --out "$OUTDIR/suite_report.json"
