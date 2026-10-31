"""Exploratory T2A interpolation; NOT a normalized scientific performance score.

WARNING: the legacy oracle fits states on half the reference, whereas source
metrics use full-reference states. Its interpolation is not a fraction of an
established ceiling. This script does not verify matched window identities.

Place model T2A numbers between their reference-split and persistence diagnostics.

Reference-state coverage depends on structural dispersion and the fixed state
radius threshold; a value cannot be interpreted without its protocol. This script joins

  * a model/source T2A file (``scripts/t2a_from_samples.py`` output), and
  * the anchors file for the SAME protocol (``scripts/t2a_baselines.py`` output),

and reports an arithmetic interpolation from persistence (0) to reference split
(1), per temperature. The caller must independently establish identical protein,
window, mask and reference protocols; the printed counts cannot establish that.
Free-energy MAE is also reported directly in the units of this implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TERMS = ('coverage', 'free_energy_mae_kcal', 'population_mae', 'population_tv',
         'population_jsd')
# Which direction "better" is, so the persistence->oracle interpolation is
# computed for both directions correctly.
HIGHER_BETTER = {'coverage': True}


def block_of(doc, kind):
    """T2A block of a source document, keyed by temperature string."""
    if kind == 'anchors':
        return doc['oracle'], doc['persistence']
    if 'by_temperature' in doc:
        return doc['by_temperature'], None
    raise ValueError('unrecognized T2A document: expected by_temperature or oracle')


def frac_to_oracle(value, persist, oracle, higher_better):
    span = (oracle - persist) if higher_better else (persist - oracle)
    if abs(span) < 1e-12:
        return float('nan')
    delta = (value - persist) if higher_better else (persist - value)
    return delta / span


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchors', required=True)
    p.add_argument('--source', nargs='+', required=True,
                   help='NAME=PATH pairs of model/source T2A documents')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    anchors = json.load(open(a.anchors))
    oracle, persistence = block_of(anchors, 'anchors')
    temps = sorted(oracle, key=float)

    sources = {}
    for spec in a.source:
        if '=' not in spec:
            raise ValueError(f'--source expects NAME=PATH, got {spec!r}')
        name, path = spec.split('=', 1)
        sources[name] = json.load(open(path))['by_temperature']

    header = f"{'T (K)':>6}  {'metric':>20}  " + '  '.join(
        f'{n:>12}' for n in ['persist'] + list(sources) + ['oracle'])
    print(header)
    print('-' * len(header))
    summary = {}
    for T in temps:
        for term in TERMS:
            higher = HIGHER_BETTER.get(term, False)
            po = persistence[T][term]['mean']
            oo = oracle[T][term]['mean']
            row = [f'{T:>6}', f'{term:>20}', f'{po:>12.4f}']
            for name, doc in sources.items():
                if T not in doc:
                    row.append(f"{'n/a':>12}")
                    continue
                mv = doc[T][term]['mean']
                frac = frac_to_oracle(mv, po, oo, higher)
                row.append(f'{mv:>7.4f}({frac:+.2f})')
                summary.setdefault(name, {}).setdefault(term, {})[T] = dict(
                    value=mv, persist=po, oracle=oo, frac_to_oracle=frac)
            row.append(f'{oo:>12.4f}')
            print('  '.join(row))
        print()

    print('Fraction in parentheses = position between persistence (0.00) and the')
    print('reference-half split (1.00). This is exploratory arithmetic only:')
    print('the split uses different reference states and is NOT a performance ceiling.')
    for name, doc in sources.items():
        counts = {T: doc[T].get('n_proteins', '?') for T in doc}
        print(f'  {name}: proteins per temperature {counts}')

    if a.out:
        Path(a.out).write_text(json.dumps(dict(anchors=a.anchors,
            source_documents={n: p for n, p in
                              (s.split('=', 1) for s in a.source)},
            anchor_proteins={T: oracle[T].get('n_proteins') for T in temps},
            by_source=summary), indent=2))
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()