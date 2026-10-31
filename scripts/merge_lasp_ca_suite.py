"""Merge supplementary ``eval_lasp_ca_suite`` runs into a main suite result.

Why this exists: the suite's "wrong temperature" arm used a fixed 320 K, which is
also the lowest evaluated mdCATH temperature. At T* = 320 K the wrong arm
therefore reproduced the correct arm exactly, making the mechanism test at that
temperature vacuous. The guard in ``eval_lasp_ca_suite.temperature_tensor`` now
substitutes the furthest mdCATH temperature, but a suite that was already
mid-flight when the guard landed still holds the degenerate cell.

Rather than restarting the whole sweep, a supplementary one-cell run is executed

    python -m scripts.eval_lasp_ca_suite --checkpoint CK --out SUPP.json \
        --modes wrong --temps 320 --wrong-temp 450 \
        --proteins-per-temp 12 --anchors-per-trajectory 2 --n-samples 12 \
        --horizons 1 4 16 64 --max-residues 96

and this script replaces only that cell. Per-window seeds are derived from
``(mode, protein, trajectory, anchor, horizon)`` and do NOT include the
temperature, so the supplementary run draws the same sampling noise as the
degenerate cell it replaces and differs from it only in the temperature fed to
the model. (Note the seed key does contain ``mode``, so the supplementary "wrong"
arm is not bit-identical to the "correct" arm either in noise or in input; it is
the control for the discarded cell, which is what the merge needs.)

The aggregate / by-horizon blocks are RECOMPUTED from the merged per-window rows
(they contain the same per-window metric dictionaries the suite archives), so no
stale average of the degenerate cell survives anywhere in the output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.eval_lasp_ca_suite import summarize


def cell_of(row):
    return (float(row['T_star']), row['mode'])


def window_key(row):
    return (row['protein'], row['traj'], int(row['anchor_frame']), float(row['horizon']))


def merge(main, supplements):
    """Replace whole (T_star, mode) cells of ``main`` with the supplements'."""
    replace = {}
    for supp in supplements:
        for field in ('checkpoint', 'manifest', 'horizons', 'proteins_per_temp',
                      'anchors_per_trajectory', 'n_samples', 'context_frames',
                      'max_residues', 'seed'):
            if supp['args'].get(field) != main['args'].get(field):
                raise ValueError(f'supplement protocol mismatch: {field}')
        for key in {(float(r['T_star']), r['mode']) for r in supp['per_window']}:
            if key in replace:
                raise ValueError(f'two supplements both cover cell {key}')
            replace[key] = supp
    if not replace:
        raise ValueError('no supplementary cells given')

    rows, used = [], set()
    dropped = {}
    for row in main['per_window']:
        key = cell_of(row)
        if key in replace:
            used.add(key)
            dropped.setdefault(key, Counter())[window_key(row)] += 1
            continue
        rows.append(row)
    missing = set(replace) - used
    if missing:
        raise ValueError(f'supplement cells absent from the main result: {sorted(missing)}')

    for key, supp in sorted(replace.items()):
        cell_rows = [r for r in supp['per_window'] if cell_of(r) == key]
        # The cell must be replaced over the SAME windows, otherwise the merged
        # cell silently mixes two protocols (e.g. a different protein budget).
        got = Counter(window_key(r) for r in cell_rows)
        if any(n != 1 for n in got.values()) or any(n != 1 for n in dropped[key].values()):
            raise ValueError(f'duplicate windows in cell {key}')
        if got != dropped[key]:
            only_supp = sorted((got - dropped[key]).elements())[:3]
            only_main = sorted((dropped[key] - got).elements())[:3]
            raise ValueError(
                f'supplement {key} covers different windows than the main cell: '
                f'{len(got)} vs {len(dropped[key])}; '
                f'only-supplement={only_supp} only-main={only_main}')
        # A "wrong" arm that equals the observed temperature is the degenerate
        # cell this whole merge exists to remove.
        for r in cell_rows:
            used_k = r.get('temperature_used_k')
            if r['mode'] == 'wrong' and used_k is not None and \
                    abs(float(used_k) - float(r['temperature_k'])) < 1e-6:
                raise ValueError(
                    f'supplement {key} is still degenerate: window feeds '
                    f'{used_k} K at an observed {r["temperature_k"]} K')
        rows.extend(cell_rows)

    temps = sorted({float(r['T_star']) for r in rows})
    modes = sorted({r['mode'] for r in rows})
    horizons = sorted({float(r['horizon']) for r in rows})

    out = dict(main)
    out['per_window'] = rows
    out['n_windows'] = len(rows)
    out['by_temperature'] = {f'{T:g}': {m: summarize(
        [r for r in rows if float(r['T_star']) == T and r['mode'] == m]) for m in modes}
        for T in temps}
    out['by_horizon'] = {f'{h:g}': {m: summarize(
        [r for r in rows if float(r['horizon']) == h and r['mode'] == m]) for m in modes}
        for h in horizons}
    out['aggregate'] = {m: summarize([r for r in rows if r['mode'] == m]) for m in modes}
    out['merged_from'] = dict(
        main=main['args']['checkpoint'],
        supplements={f'{T:g}|{m}': s['args']['checkpoint'] for (T, m), s in replace.items()},
        note='supplementary cells recomputed with a distinct wrong temperature; '
             'aggregate and by_horizon recomputed from merged per-window rows')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main', required=True)
    p.add_argument('--supplement', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    merged = merge(json.load(open(a.main)),
                   [json.load(open(s)) for s in a.supplement])
    Path(a.out).write_text(json.dumps(merged, indent=2))
    print(json.dumps(dict(out=a.out, n_windows=merged['n_windows'],
                          cells=merged['merged_from']['supplements']), indent=2))


if __name__ == '__main__':
    main()