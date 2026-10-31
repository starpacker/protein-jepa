#!/usr/bin/env bash
# Finalize round 1: merge the T=320 wrong-temperature supplements into the main
# suite results, then render the consolidated round report.
#
# Round 1 was launched before the anti-degeneracy guard landed, so the T*=320
# "wrong" cell in each main run fed the observed temperature and is meaningless.
# Each seed has a supplementary one-cell run (--modes wrong --temps 320
# --wrong-temp 450) over exactly the same windows, started under the same
# (pre-guard) seed scheme, so it replaces that cell cleanly.
#
# Usage: bash run_scripts/finalize_lasp_round1.sh [dir]
set -euo pipefail
cd /liuyutian/protein_jepa_frame
DIR=${1:-results/lasp_full_v1/mid_suite}
SEEDS=${SEEDS:-"42 43 44"}

merged=()
for s in $SEEDS; do
  main="$DIR/seed$s.json"
  supp="$DIR/seed${s}_supp320.json"
  [ -f "$main" ] || { echo "missing $main" >&2; exit 1; }
  [ -f "$supp" ] || { echo "missing $supp" >&2; exit 1; }
  out="$DIR/seed$s.merged.json"
  echo "=== merging seed$s ==="
  python -u -m scripts.merge_lasp_ca_suite --main "$main" --supplement "$supp" --out "$out"
  merged+=("$out")
done

echo "=== consolidated round-1 report ==="
python -u -m scripts.report_lasp_ca_suite "${merged[@]}" \
  --out "$DIR/round1_report.json" | tee "$DIR/round1_report.txt"

echo "=== T2A anchors for reference (model-free) ==="
if [ -f results/lasp_full_v1/t2a_baselines/anchors.json ]; then
  python - <<'PY'
import json
d = json.load(open('results/lasp_full_v1/t2a_baselines/anchors.json'))
for kind in ('oracle', 'persistence'):
    print(f'\n--- {kind} ---')
    print(f"{'T':>6}  {'coverage':>9}  {'F_mae':>8}  {'popTV':>8}  {'popJSD':>8}")
    for T, row in sorted(d[kind].items(), key=lambda kv: float(kv[0])):
        print(f"{T:>6}  {row['coverage']['mean']:>9.4f}  "
              f"{row['free_energy_mae_kcal']['mean']:>8.4f}  "
              f"{row['population_tv']['mean']:>8.4f}  "
              f"{row['population_jsd']['mean']:>8.4f}")
PY
else
  echo "anchors.json not present; run scripts/t2a_baselines.py first"
fi

if [ -f results/lasp_full_v1/t2a_bioemu2/bioemu2_t2a.json ]; then
  echo "=== bioemu2 T2A (same references) ==="
  python - <<'PY'
import json
d = json.load(open('results/lasp_full_v1/t2a_bioemu2/bioemu2_t2a.json'))
print(f"{'T':>6}  {'coverage':>9}  {'F_mae':>8}  {'popTV':>8}  {'popJSD':>8}  {'n_prot':>6}")
for T, row in sorted(d['by_temperature'].items(), key=lambda kv: float(kv[0])):
    print(f"{T:>6}  {row['coverage']['mean']:>9.4f}  "
          f"{row['free_energy_mae_kcal']['mean']:>8.4f}  "
          f"{row['population_tv']['mean']:>8.4f}  "
          f"{row['population_jsd']['mean']:>8.4f}  {row['n_proteins']:>6}")
PY
else
  echo "bioemu2 T2A not ready yet (run_scripts/run_bioemu2_t2a_samples.sh)"
fi

echo "=== round 1 finalized ==="