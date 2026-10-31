"""Reference anchors for the T2A (BioEmu-style static-ensemble) metrics.

T2A numbers are hard to read in isolation: is a coverage of 0.85 good? This
script computes the two anchor points on exactly the same windows, trajectories
and reference ensembles that ``eval_lasp_ca_suite --t2a`` uses, WITHOUT sampling
from any model:

  * **oracle** -- half of the reference trajectory's own frames, scored against
    the reference ensemble built from the other half. This is the ceiling: an
    idealized sampler of the true equilibrium distribution. Reported as a
    two-fold split, so it is a lower bound on the true oracle ceiling (finite
    sample effects alone cost some coverage).
  * **persistence** -- the observed anchor frame of each window, repeated. This
    is the trivial "the structure does not move" baseline, and it is the same
    baseline the ensemble metrics use. Coverage should collapse toward zero as
    the trajectory's true ensemble spreads.

Both use the identical protocol (proteins / replicas / anchors / horizons /
max_residues) and the identical per-trajectory reference ensembles, so a model's
T2A numbers can be placed between these two anchors directly.

Cost: this reads MD frames only -- no denoising, no GPU sampling.

Usage:
  python -m scripts.t2a_baselines --out results/.../t2a_baselines.json \
      --proteins-per-temp 12 --anchors-per-trajectory 2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.t2a_static import static_ensemble_terms
from scripts.eval_lasp_ca_probe import bootstrap
from scripts.eval_lasp_ca_suite import (MDCATH_TEMPS, population_counts,
                                        reference_ensemble)

TERM_KEYS = ('coverage', 'free_energy_mae_kcal', 'population_mae',
             'population_tv', 'population_jsd')


def split_reference(ca, seed=0):
    """Two-fold split of a reference ensemble into (model_half, ref_half)."""
    n = ca.shape[0]
    if n < 4:
        raise ValueError(f'reference ensemble too small to split: {n} frames')
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    half = n // 2
    return ca[order[:half]], ca[order[half:]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--temps', type=float, nargs='+', default=list(MDCATH_TEMPS))
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    p.add_argument('--proteins-per-temp', type=int, default=12)
    p.add_argument('--anchors-per-trajectory', type=int, default=2)
    p.add_argument('--context-frames', type=int, default=8)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--max-frames', type=int, default=512)
    p.add_argument('--k-states', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    # Collect the anchor frames actually used, per (protein, trajectory, T), so
    # persistence is evaluated on the identical window set as the model.
    anchors = defaultdict(list)
    for T in a.temps:
        protocol = EvalProtocol(manifest=a.manifest,
            horizons_frames=tuple(a.horizons), context_frames=a.context_frames,
            frame_stride=1, max_residues=a.max_residues,
            max_proteins=a.proteins_per_temp,
            max_anchors_per_trajectory=a.anchors_per_trajectory,
            temperatures=(T,), seed=a.seed, round_robin_proteins=True,
            round_robin_conditions=True)
        n_before = sum(len(v) for v in anchors.values())
        for w in iter_windows(protocol):
            mask = w.context['residue_mask'].all(0).bool()
            if int(mask.sum()) < 4:
                continue
            key = (w.protein_id, w.trajectory_id, float(w.temperature_k))
            anchors[key].append(w.context['coords'][-1, :, 1][mask].clone())
        if sum(len(v) for v in anchors.values()) == n_before:
            raise ValueError(f'no eligible T2A baseline windows for T={T:g}')
        print(f'[t2a-baselines] T={T:g} collected', flush=True)

    if not anchors:
        raise ValueError('no eligible T2A baseline windows; refusing empty report')
    refs = reference_ensemble(a.manifest, [k[1] for k in anchors],
                              a.max_residues, max_frames=a.max_frames)

    per_kind = {kind: defaultdict(list) for kind in ('oracle', 'persistence')}
    for (protein, traj, T), anchor_list in sorted(anchors.items()):
        # reference_ensemble returns FULL-length frames plus the mask; the
        # anchors were already restricted to the protocol mask, so restrict the
        # reference the same way before comparing widths.
        ca_ref_full, ref_mask = refs[traj]
        ca_ref = ca_ref_full[:, ref_mask]
        persist_ca = torch.stack(anchor_list, 0)
        if persist_ca.shape[1] != ca_ref.shape[1]:
            raise ValueError(f'anchor/reference width mismatch for {traj}: '
                             f'{persist_ca.shape[1]} vs {ca_ref.shape[1]}')
        model_half, ref_half = split_reference(ca_ref, seed=a.seed)
        oracle = static_ensemble_terms(model_half, ref_half, None, T,
                                       k_states=a.k_states, seed=a.seed)
        persistence = static_ensemble_terms(persist_ca, ca_ref, None, T,
                                           k_states=a.k_states, seed=a.seed)
        for kind, terms in (('oracle', oracle), ('persistence', persistence)):
            per_kind[kind][f'{T:g}'].append(dict(protein=protein, trajectory=traj,
                n_anchors=len(anchor_list), **terms))

    def block(kind):
        return {T: dict(**population_counts(v), **{k: bootstrap(
            [dict(protein=r['protein'], value=r[k]) for r in v], 'value')
            for k in TERM_KEYS}) for T, v in per_kind[kind].items()}

    result = dict(manifest=a.manifest, args=vars(a),
        note='Oracle and persistence anchors for T2A on the identical windows '
             'and reference ensembles as scripts/eval_lasp_ca_suite.py --t2a. '
             'The oracle is a two-fold split, so it is a LOWER bound on the '
             'ideal ceiling (finite-sample cost included).',
        oracle=block('oracle'), persistence=block('persistence'),
        per_protein={kind: {T: v for T, v in d.items()} for kind, d in per_kind.items()})
    Path(a.out).write_text(json.dumps(result, indent=2))

    for kind in ('oracle', 'persistence'):
        print(f'\n=== T2A anchor: {kind} ===')
        print(f"{'temperature':>12}  {'coverage':>9}  {'F_mae':>8}  {'popTV':>8}  {'popJSD':>8}  {'n_prot':>6}  {'n_traj':>6}")
        for T, row in sorted(result[kind].items(), key=lambda kv: float(kv[0])):
            print(f"{T:>12}  {row['coverage']['mean']:>9.4f}  "
                  f"{row['free_energy_mae_kcal']['mean']:>8.4f}  "
                  f"{row['population_tv']['mean']:>8.4f}  "
                  f"{row['population_jsd']['mean']:>8.4f}  "
                  f"{row['n_proteins']:>6}  {row['n_trajectories']:>6}")


if __name__ == '__main__':
    main()