"""BioEmu-style static-ensemble metrics (T2A) from saved per-window samples.

Consumes the ``.npz`` files written by any evaluation runner that supports
per-window sample dumping (``--save-samples``; see
``evaluation/baselines/run_bioemu2_zeroshot.py``) and scores them against the
*same* reference ensembles this repository builds for its own models.

Contract shared with ``scripts/eval_lasp_ca_suite.py``:
  * one reference ensemble per trajectory, built from that trajectory's true MD
    frames resampled to at most ``--max-frames`` frames;
  * the residue mask is taken from the protocol window and must equal the
    reference mask, otherwise the run aborts instead of silently scoring a
    different subsystem;
  * samples from every saved window of a (protein, trajectory, temperature)
    triple are pooled, exactly like the in-suite T2A pool.

Saved samples are already restricted to the masked residues, so the metrics see
the reference restricted to the identical residue subset.

Future coordinates are reference data only: nothing here feeds a model.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.t2a_static import static_ensemble_terms
from scripts.eval_lasp_ca_probe import bootstrap
from scripts.eval_lasp_ca_suite import (MDCATH_TEMPS, population_counts,
                                        reference_ensemble)

TERM_KEYS = ('coverage', 'free_energy_mae_kcal', 'population_mae',
             'population_tv', 'population_jsd')


def expected_bioemu2_windows(manifest, temps=MDCATH_TEMPS,
                            horizons=(1, 4, 16, 64), proteins_per_temp=8,
                            anchors_per_trajectory=1, context_frames=8,
                            max_residues=96, seed=0):
    """Replay each launcher's baseline protocol, including pre-sampling skips.

    Do not use the suite's pool or a nominal 800-file count: equal counts can
    hide missing windows replaced by unrelated ones. No model/GPU is involved.
    """
    expected = set()
    for T in temps:
        protocol = EvalProtocol(manifest=manifest,
            horizons_frames=tuple(max(1, int(round(h))) for h in horizons),
            context_frames=context_frames, context_stride=1, frame_stride=1,
            max_residues=max_residues, max_proteins=proteins_per_temp,
            temperatures=(float(T),),
            max_anchors_per_trajectory=anchors_per_trajectory,
            anchor_stride=1, temporal_lags=(), emit_multi_lag=False,
            seed=seed, round_robin_proteins=True, round_robin_conditions=True,
            max_windows_per_protein=None)
        n_before = len(expected)
        for w in iter_windows(protocol):
            mask = w.context['residue_mask'].all(dim=0).bool()
            if int(mask.sum()) < 4:
                continue
            fp = getattr(w, 'future_path', None)
            if fp is None:
                continue
            true_ca = fp['coords'][:, :, 1, :][:, mask.cpu()]
            if true_ca.shape[0] < 2:
                continue
            expected.add((str(w.protein_id), str(w.trajectory_id),
                          float(w.temperature_k), int(w.anchor_frame), float(w.horizon)))
        if len(expected) == n_before:
            raise ValueError(f'no eligible BioEmu2 protocol windows for T={T:g}')
    if not expected:
        raise ValueError('empty expected BioEmu2 window set')
    return expected


def validate_completeness(pool, expected):
    actual = {identity for entry in pool.values() for identity in entry['identities']}
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        raise ValueError('incomplete/mismatched BioEmu2 samples; refusing final report: '
                         f'expected={len(expected)} actual={len(actual)} '
                         f'missing={len(missing)} unexpected={len(unexpected)}; '
                         f'missing examples={sorted(missing)[:3]}; '
                         f'unexpected examples={sorted(unexpected)[:3]}')
    return dict(complete=True, expected_windows=len(expected), actual_windows=len(actual))


def load_pool(sample_dir, horizons=None):
    """Group saved windows by (protein, trajectory, temperature)."""
    pool = {}
    seen = set()
    for path in sorted(Path(sample_dir).glob('*.npz')):
        with np.load(path) as data:
            horizon = float(data['horizon'])
            if horizons and horizon not in horizons:
                continue
            # Identity is stored inside the archive; the filename is only a
            # fallback for dumps written before the ids were embedded.
            if 'protein' in data and 'trajectory' in data:
                protein, trajectory = str(data['protein']), str(data['trajectory'])
            else:
                parts = path.name.split('__')
                if len(parts) < 2:
                    raise ValueError(f'cannot recover protein/trajectory from {path.name}')
                protein, trajectory = parts[0], parts[1]
            key = (protein, trajectory, float(data['temperature_k']))
            # Older dumps without embedded ids still carry the anchor in the
            # baseline's protein__trajectory__anchor__horizon filename.
            parts = path.stem.split('__')
            if 'anchor_frame' in data:
                anchor = int(data['anchor_frame'])
            elif len(parts) == 4:
                anchor = int(parts[2])
            else:
                raise ValueError(f'cannot recover anchor_frame from {path.name}')
            identity = (*key, anchor, horizon)
            if identity in seen:
                raise ValueError(f'duplicate saved window identity: {identity}')
            seen.add(identity)
            entry = pool.setdefault(key, dict(samples=[], mask=None, horizons=[],
                                              files=[], identities=[]))
            mask = torch.as_tensor(np.asarray(data['residue_mask'])).bool()
            samples = torch.as_tensor(np.asarray(data['model_ca'])).float()
            if (mask.ndim != 1 or int(mask.sum()) < 4 or samples.ndim != 3
                    or samples.shape[0] < 1 or samples.shape[1] != int(mask.sum())
                    or samples.shape[2] != 3 or not torch.isfinite(samples).all()):
                raise ValueError(f'invalid saved sample shape/mask or non-finite values: {path}')
            if entry['mask'] is None:
                entry['mask'] = mask
            elif not torch.equal(entry['mask'], mask):
                raise ValueError(f'inconsistent residue mask within {key}')
            entry['samples'].append(samples)
            entry['horizons'].append(horizon)
            entry['files'].append(path.name)
            entry['identities'].append(identity)
    return pool


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', required=True, help='directory of per-window .npz')
    p.add_argument('--manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--source', default='bioemu2')
    p.add_argument('--horizons', type=float, nargs='+', default=None,
                   help='restrict the pool to these lag times (ns)')
    p.add_argument('--temps', type=float, nargs='+', default=list(MDCATH_TEMPS),
                   help='expected BioEmu2 launch temperatures (not inferred from available files)')
    p.add_argument('--proteins-per-temp', type=int, default=8)
    p.add_argument('--anchors-per-trajectory', type=int, default=1)
    p.add_argument('--context-frames', type=int, default=8)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--max-frames', type=int, default=512)
    p.add_argument('--k-states', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    pool = load_pool(a.samples, set(a.horizons) if a.horizons else None)
    if not pool:
        raise ValueError(f'no saved samples under {a.samples}')
    completeness = None
    if a.source.lower() == 'bioemu2':
        expected = expected_bioemu2_windows(a.manifest, temps=a.temps,
            horizons=a.horizons if a.horizons is not None else (1, 4, 16, 64),
            proteins_per_temp=a.proteins_per_temp,
            anchors_per_trajectory=a.anchors_per_trajectory,
            context_frames=a.context_frames, max_residues=a.max_residues, seed=a.seed)
        completeness = validate_completeness(pool, expected)
    refs = reference_ensemble(a.manifest, [traj for _, traj, _ in pool],
                              a.max_residues, max_frames=a.max_frames)
    per_protein = defaultdict(list)
    for (protein, traj, T), entry in sorted(pool.items()):
        ca_ref, ref_mask = refs[traj]
        use = entry['mask']
        if int(use.sum()) < 4:
            continue
        if not torch.equal(use, ref_mask):
            raise ValueError('saved samples and reference residue masks differ for ' + traj)
        # Samples and reference are both pre-restricted to `use`; mask=None.
        pooled = torch.cat(entry['samples'], 0)
        terms = static_ensemble_terms(pooled, ca_ref[:, use], None, T,
                                      k_states=a.k_states, seed=a.seed)
        per_protein[f'{T:g}'].append(dict(protein=protein, trajectory=traj,
            n_windows=len(entry['samples']), horizons=sorted(set(entry['horizons'])),
            **terms))
    scored_rows = [r for rows in per_protein.values() for r in rows]
    result = dict(source=a.source, sample_dir=a.samples, manifest=a.manifest,
        args=vars(a), completeness=completeness,
        n_windows=sum(r['n_windows'] for r in scored_rows), **population_counts(scored_rows),
        note='Static-ensemble (T2A) metrics; reference ensembles identical to the in-suite T2A.',
        total={k: bootstrap([dict(protein=r['protein'], value=r[k])
                             for v in per_protein.values() for r in v], 'value')
               for k in TERM_KEYS},
        by_temperature={T: dict(**population_counts(v), **{k: bootstrap(
            [dict(protein=r['protein'], value=r[k]) for r in v], 'value') for k in TERM_KEYS})
            for T, v in per_protein.items()},
        per_protein={T: v for T, v in per_protein.items()})
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(source=a.source, **population_counts(scored_rows),
        n_windows=result['n_windows'], completeness=completeness,
        temperatures=sorted(per_protein)), indent=2), flush=True)
    for T in sorted(per_protein, key=float):
        row = result['by_temperature'][T]
        print(f"T={T:>5}  cov={row['coverage']['mean']:.3f}  "
              f"F_mae={row['free_energy_mae_kcal']['mean']:.3f}  "
              f"popTV={row['population_tv']['mean']:.3f}  "
              f"popJSD={row['population_jsd']['mean']:.3f}  "
              f"n_proteins={row['n_proteins']} n_trajectories={row['n_trajectories']}",
              flush=True)


if __name__ == '__main__':
    main()