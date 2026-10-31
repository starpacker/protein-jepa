"""Summarize one or more ``eval_lasp_ca_suite`` runs into the round report.

Reads the JSON produced by ``scripts/eval_lasp_ca_suite.py`` for each seed and
prints the three tables the suite exists for:

  1. per-temperature accuracy (mode == correct), one row per temperature,
     averaged across seed-level protein-balanced means; original run documents
     retain protein-clustered bootstrap CIs;
  2. the temperature-conditioning matrix (correct / wrong / none), which is the
     mechanism test -- if feeding a wrong temperature does not change the
     generated ensemble, the model is not actually using its temperature input;
  3. BioEmu-style static-ensemble metrics (T2A) per temperature.

All numbers are development diagnostics on the val split.  No test claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HEADLINE = ('rmsf_mae', 'rmsf_corr', 'root_mean_w2', 'md_pca_w2',
            'weak_contact_j', 'transient_contacts_j')
POSITION = ('kabsch_ca_rmsd', 'aligned_ca_rmsd', 'skill_rmsf', 'skill_position')
T2A = ('coverage', 'free_energy_mae_kcal', 'population_mae', 'population_tv',
       'population_jsd')


def fmt(entry):
    if not entry:
        return 'n/a'
    lo, hi = entry.get('ci95', [float('nan')] * 2)
    return f"{entry['mean']:.4f} [{lo:.4f},{hi:.4f}]"


def short(entry):
    return 'n/a' if not entry else f"{entry['mean']:.4f}"


def load(paths):
    runs = [json.load(open(p)) for p in paths]
    return runs


def table(rows, header):
    """Right-aligned table; ``header`` must label every column of every row."""
    if not rows:
        print('(no rows)')
        return
    if any(len(r) != len(header) for r in rows):
        raise ValueError(f'row/header width mismatch: header={len(header)} '
                         f'rows={sorted({len(r) for r in rows})}')
    width = max(len(str(header[0])), *(len(str(r[0])) for r in rows))
    line = f'{str(header[0]):>{width}}  ' + '  '.join(f'{str(h):>12}' for h in header[1:])
    print(line)
    print('-' * len(line))
    for row in rows:
        print(f'{str(row[0]):>{width}}  ' + '  '.join(f'{str(c):>12}' for c in row[1:]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', help='seed JSON files from eval_lasp_ca_suite')
    p.add_argument('--out', default=None, help='also write a consolidated JSON')
    a = p.parse_args()

    runs = load(a.runs)
    temps = sorted(runs[0]['by_temperature'], key=float)
    modes = list(runs[0]['args']['modes'])
    print(f'runs: {len(runs)}  windows/run: {[r["n_windows"] for r in runs]}  '
          f'temperatures: {temps}  modes: {modes}\n')

    def agg(fetch):
        """Cross-seed mean of the bootstrap entries a ``fetch`` returns per run."""
        vals = {}
        for r in runs:
            for metric, entry in (fetch(r) or {}).items():
                if isinstance(entry, dict) and 'mean' in entry:
                    vals.setdefault(metric, []).append(entry['mean'])
        return {k: dict(mean=float(np.mean(v)), seed_values=v,
                        seed_min=float(np.min(v)), seed_max=float(np.max(v)))
                for k, v in vals.items()}

    def at(temp, mode):
        return agg(lambda r: r['by_temperature'].get(temp, {}).get(mode))

    print('=== 1. per-temperature accuracy (mode=correct, cross-seed mean) ===')
    rows = []
    for T in temps:
        block = at(T, 'correct')
        rows.append([f'T={T}'] + [short(block.get(m)) for m in HEADLINE]
                    + [short(block.get('skill_rmsf')), short(block.get('skill_position'))])
    table(rows, ['temperature'] + list(HEADLINE) + ['skill_rmsf', 'skill_pos'])

    print('\n=== 2. temperature-conditioning matrix (cross-seed mean) ===')
    for metric in ('rmsf_mae', 'md_pca_w2', 'aligned_ca_rmsd'):
        rows = []
        for T in temps:
            rows.append([f'T={T}'] + [short(at(T, m).get(metric)) for m in modes])
        table(rows, [f'{metric} (lower=better)'] + modes)

    print('\n=== 2b. aggregate over temperatures (cross-seed mean) ===')
    rows = []
    for m in modes:
        vals = {k: v['mean'] for k, v in agg(lambda r: r['aggregate'].get(m)).items()}
        rows.append([m] + [f'{vals.get(k, float("nan")):.4f}' for k in HEADLINE]
                    + [f'{vals.get("skill_rmsf", float("nan")):+.4f}',
                       f'{vals.get("skill_position", float("nan")):+.4f}'])
    table(rows, ['mode'] + list(HEADLINE) + ['skill_rmsf', 'skill_pos'])

    print('\n=== 3. BioEmu-style static ensemble (T2A), mode=correct ===')
    if runs[0].get('t2a'):
        rows = []
        for T in temps:
            vals = {k: [] for k in T2A}
            for r in runs:
                block = (r.get('t2a') or {}).get(T)
                if not block:
                    continue
                for k in T2A:
                    if k in block:
                        vals[k].append(block[k]['mean'])
            rows.append([f'T={T}'] + [f'{np.mean(v):.4f}' if v else 'n/a'
                                      for v in (vals[k] for k in T2A)]
                        + [str(len({r['protein'] for r in
                                    runs[0]['t2a_per_protein'][T]}))])
        table(rows, ['temperature'] + list(T2A) + ['n_prot'])
    else:
        print('no T2A block in these runs')

    print('\n=== metric directions ===')
    print('lower better : rmsf_mae, root_mean_w2, variance_w2, md_pca_w2, '
          'kabsch_ca_rmsd, aligned_ca_rmsd, free_energy_mae_kcal, population_*')
    print('higher better: rmsf_corr, weak_contact_j, transient_contacts_j, coverage, '
          'skill_rmsf (persistence - model), skill_position')

    if a.out:
        consolidated = dict(sources=a.runs, n_runs=len(runs), temperatures=temps,
                            modes=modes,
                            by_temperature={T: {m: at(T, m) for m in modes} for T in temps},
                            aggregate={m: agg(lambda r: r['aggregate'].get(m)) for m in modes},
                            t2a={r['args']['checkpoint']: r.get('t2a') for r in runs})
        Path(a.out).write_text(json.dumps(consolidated, indent=2))
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()