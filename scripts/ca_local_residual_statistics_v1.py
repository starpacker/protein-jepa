"""Protein-paired statistics for a fixed-cache CA decoder capacity probe.

No model construction, sampling, trajectory reads or checkpoint selection.
Missing scalars never cause an implicit change of the registered panel.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import math

import numpy as np

VERSION = 'ca_local_residual_statistics_v1'
BOOTSTRAPS, SEED = 10000, 20261101
HORIZONS = (1, 4, 16, 64)
TEMPERATURES = (320., 348., 379., 413., 450.)
ARMS = ('baseline', 'base_tuned', 'local_residual')
CONTRASTS = (('local_residual', 'base_tuned'),
             ('base_tuned', 'baseline'), ('local_residual', 'baseline'))
STAGES = ('raw', 'projected')
ROLES = ('oracle', 'forecast', 'anchor_identity', 'wrong_horizon_H')
GEOMETRY = tuple(f'{kind}_{metric}' for kind in ('canonical', 'psf') for metric in
    ('ca_bond_mean_angstrom', 'ca_bond_invalid_fraction',
     'ca_nonlocal_clash_fraction', 'ca_radius_gyration_angstrom'))
SCALARS = ('coordinate_mse', 'ca_rmsd', 'rmsd_to_persistence_ratio', *GEOMETRY)
FORECAST_SCALARS = (*SCALARS, 'coordinate_energy_score',
                    'physical_distance_energy_score', 'contact_brier')
INTERVAL_CELLS = 1 + len(CONTRASTS)*10*len(STAGES)*(len(SCALARS)+len(FORECAST_SCALARS))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number))
            and math.isfinite(float(value)), 'metric must be a finite number or explicit null')
    return float(value)


def key(row):
    require(isinstance(row['protein'], str) and isinstance(row['trajectory'], str), 'invalid identity')
    for name in ('anchor_frame', 'horizon'):
        value = row[name]
        require(type(value) is int and value >= (1 if name == 'horizon' else 0), 'invalid frame identity')
    return (row['protein'], row['trajectory'], number(row['temperature_k']),
            row['anchor_frame'], row['horizon'])


def indexed(rows, split):
    require(split in ('train', 'val'), 'unknown split')
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, 'duplicate window identity')
        require(k[4] in HORIZONS and k[2] in TEMPERATURES, 'unregistered condition')
        result[k] = row
    expected, proteins = (500, 500) if split == 'train' else (200, 50)
    require(len(result) == expected and len({k[0] for k in result}) == proteins, 'incomplete panel')
    require(Counter(k[4] for k in result) == {h: expected//4 for h in HORIZONS}, 'unbalanced horizons')
    if split == 'val':
        require(Counter(k[2] for k in result) == {t: 40 for t in TEMPERATURES}, 'unbalanced temperatures')
        for protein in sorted({k[0] for k in result}):
            selected = [k for k in result if k[0] == protein]
            require(len(selected) == 4 and {k[4] for k in selected} == set(HORIZONS)
                    and len({k[1:4] for k in selected}) == 1, 'validation anchor/condition mismatch')
    return result


def groups_for(keys):
    keys = sorted(keys)
    return dict(overall=keys,
        **{f'h{h}': [k for k in keys if k[4] == h] for h in HORIZONS},
        **{f'T{int(t)}': [k for k in keys if k[2] == t] for t in TEMPERATURES})


def support(rows, keys, getter):
    require(bool(keys) and len(keys) == len(set(keys)), 'empty/duplicate support')
    require(all(k in rows for k in keys), 'missing window; no intersection fallback')
    by_protein, missing = defaultdict(list), []
    for k in sorted(keys):
        value = getter(rows[k])
        if value is None:
            missing.append(list(k))
        else:
            by_protein[k[0]].append(number(value))
    means = {p: float(np.mean(values)) for p, values in sorted(by_protein.items())}
    available = float(np.mean(list(means.values()))) if means else None
    return dict(complete_panel_mean=available if not missing else None,
        available_support_mean=available, n_windows=len(keys), n_valid_windows=len(keys)-len(missing),
        n_proteins=len({k[0] for k in keys}), n_valid_proteins=len(means),
        missing_window_keys=missing, per_protein_available_means=means)


@lru_cache(maxsize=None)
def bootstrap_indices(count):
    require(type(count) is int and count > 0, 'invalid protein count')
    value = np.random.default_rng(SEED).integers(count, size=(BOOTSTRAPS, count))
    value.setflags(write=False)
    return value


def paired(rows, keys, candidate, reference, direction):
    """Null complete-panel CI if either scalar is missing; never paired deletion."""
    require(direction in ('lower', 'higher', None), 'invalid direction')
    first, second = support(rows, keys, candidate), support(rows, keys, reference)
    result = dict(candidate=first['complete_panel_mean'], reference=second['complete_panel_mean'],
        candidate_minus_reference=None, ci95=None, better=direction,
        improvement=None, improvement_ci95=None, n_windows=len(keys),
        n_proteins=first['n_proteins'], candidate_support=first, reference_support=second)
    if first['n_valid_windows'] != len(keys) or second['n_valid_windows'] != len(keys):
        result['status'] = 'undefined_on_complete_registered_panel'
        return result
    proteins = sorted(first['per_protein_available_means'])
    require(proteins == sorted(second['per_protein_available_means']), 'protein support differs')
    differences = np.array([first['per_protein_available_means'][p]-second['per_protein_available_means'][p]
                            for p in proteins], dtype=np.float64)
    ci = np.quantile(differences[bootstrap_indices(len(proteins))].mean(1), [.025, .975])
    result.update(status='complete', candidate_minus_reference=float(differences.mean()), ci95=ci.tolist())
    if direction is not None:
        sign = -1 if direction == 'lower' else 1
        result.update(improvement=sign*float(differences.mean()), improvement_ci95=sorted((sign*ci).tolist()))
    return result


def long_horizon_primary(rows, getter):
    selected = sorted(k for k in rows if k[4] in (16, 64))
    counts = Counter(k[0] for k in selected)
    require(len(selected) == 100 and len(counts) == 50 and set(counts.values()) == {2},
            'primary requires 50 proteins with both long horizons')
    result = paired(rows, selected, lambda r: getter(r, 'local_residual'),
                    lambda r: getter(r, 'base_tuned'), 'lower')
    require(result['status'] == 'complete', 'primary cannot be undefined')
    result.update(horizons=[16, 64], unit='angstrom',
        metric='fixed_gauge_oracle_CA_RMSD_after_P32',
        contrast='local_residual_minus_base_tuned')
    return result


def scalar(row, arm, role, stage, metric):
    require(arm in ARMS and role in ROLES and stage in STAGES, 'unknown scalar route')
    return row['routes'][arm][role][stage]['scalars'][metric]


def direction(metric):
    require(metric in FORECAST_SCALARS, 'unknown metric')
    return None if metric.endswith(('bond_mean_angstrom', 'radius_gyration_angstrom')) else 'lower'


def analysis(data):
    """Numerical analysis only; outer wrapper must audit provenance/executions."""
    require(data['schema_version'] == 'ca_local_residual_probe_v1', 'raw schema differs')
    require(set(data['per_window']) == {'train', 'val'}, 'splits differ')
    rows = {split: indexed(data['per_window'][split], split) for split in ('train', 'val')}
    require(not ({k[0] for k in rows['train']} & {k[0] for k in rows['val']}), 'protein split overlap')
    primary = long_horizon_primary(rows['val'],
        lambda r, arm: scalar(r, arm, 'oracle', 'projected', 'ca_rmsd'))
    descriptive = {}
    for split, selected_rows in rows.items():
        descriptive[split] = {}
        roles = ('oracle',) if split == 'train' else ROLES
        for arm in ARMS:
            descriptive[split][arm] = {}
            for role in roles:
                descriptive[split][arm][role] = {}
                metrics = FORECAST_SCALARS if role == 'forecast' else SCALARS
                for stage in STAGES:
                    descriptive[split][arm][role][stage] = {
                        group: {metric: support(selected_rows, keys,
                            lambda r, metric=metric: scalar(r, arm, role, stage, metric)) for metric in metrics}
                        for group, keys in groups_for(selected_rows).items()}
    comparisons = {}
    for candidate, reference in CONTRASTS:
        contrast = f'{candidate}_minus_{reference}'
        comparisons[contrast] = {}
        for role in ('oracle', 'forecast'):
            metrics = FORECAST_SCALARS if role == 'forecast' else SCALARS
            comparisons[contrast][role] = {}
            for stage in STAGES:
                comparisons[contrast][role][stage] = {
                    group: {metric: paired(rows['val'], keys,
                        lambda r, metric=metric: scalar(r, candidate, role, stage, metric),
                        lambda r, metric=metric: scalar(r, reference, role, stage, metric), direction(metric))
                        for metric in metrics} for group, keys in groups_for(rows['val']).items()}
    actual = [value for contrast in comparisons.values() for role in contrast.values()
              for stage in role.values() for group in stage.values() for value in group.values()]
    require(len(actual)+1 == INTERVAL_CELLS, 'interval cell count differs')
    return dict(primary=primary, comparisons=comparisons, descriptive=descriptive,
        requested_interval_cells=INTERVAL_CELLS,
        complete_interval_cells=1+sum(value['status'] == 'complete' for value in actual),
        undefined_interval_cells=sum(value['status'] != 'complete' for value in actual))


def contract():
    return dict(schema_version=VERSION, arms=list(ARMS), stages=list(STAGES),
        primary=dict(contrast='local_residual_minus_base_tuned', split='val', role='oracle',
            stage='projected', metric='ca_rmsd', horizons=[16, 64],
            n_windows=100, n_proteins=50, unit='angstrom', better='lower',
            estimand='Mean of the paired per-protein mean of the two horizon errors. '
                     'RMSD=sqrt(sum over valid residues and xyz squared error/Nres), without prediction realignment.'),
        contrasts=[list(c) for c in CONTRASTS], roles=list(ROLES),
        oracle_scalars=list(SCALARS), forecast_scalars=list(FORECAST_SCALARS),
        scalar_units={metric: ('angstrom^2' if metric == 'coordinate_mse' else
            'angstrom' if metric == 'ca_rmsd' or metric == 'coordinate_energy_score'
                or metric.endswith(('bond_mean_angstrom', 'radius_gyration_angstrom')) else
            'dimensionless distance/1 angstrom' if metric == 'physical_distance_energy_score' else
            'dimensionless') for metric in FORECAST_SCALARS},
        coordinate_mse='Mean squared xyz coordinate error=sum squared error/(3*Nvalid); '
                       'CA-RMSD=sqrt(3*MSE) per member, then mean members. '
                       'Never take sqrt of the ensemble-mean MSE and call it mean member RMSD.',
        requested_interval_cells=INTERVAL_CELLS, primary_intervals=1,
        secondary_interval_cells=INTERVAL_CELLS-1,
        scalar_directions={metric: direction(metric) for metric in FORECAST_SCALARS},
        groups=['overall', *[f'h{h}' for h in HORIZONS], *[f'T{int(t)}' for t in TEMPERATURES]],
        bootstrap=dict(reps=BOOTSTRAPS, seed=SEED, unit='protein', interval='percentile95',
            rng='numpy.default_rng, sorted proteins, identical restarted draws per scalar',
            weighting='Use each window scalar as defined, then equal windows within protein and equal proteins.',
            limits='Marginal exploratory intervals; no multiplicity, adaptive-selection, or training-seed adjustment.'),
        null_policy='Missing/nonfinite row keys fail. Explicit null scalars yield null complete-panel means and CIs, '
                    'with available support means/counts/identities separately reported. No pairwise deletion. '
                    'The primary requires complete finite support; an undefined primary fails closed.',
        train='Fixed final checkpoints evaluated on all500 training windows; descriptive only, not generalization.',
        controls='Anchor identity and fixed cross-horizon H mismatch are descriptive. Same-protein same-horizon '
                 'nonidentity permutation is unavailable because there is one validation window per such cell.',
        wrong_horizon_map={'1': 4, '4': 16, '16': 64, '64': 1},
        forecast='The original spatial_exact sampler provided all cached12 H draws. These are secondary decoder '
                 'compatibility results; no new dynamics feedback or sampling. No future-path-mixture seven metrics.',
        gate='Ratios are mean window RMSD/persistence RMSD; zero denominator is null. They diagnose this '
             'decoder/projection combination, not an information bound or a complete rerun of the old H gate.',
        topology='Legacy P32 is identical between arms. Report all7 known train topology differences and '
                 'separate actual-PSF geometry; do not silently repair the old projection.',
        baseline='Original D0 is redecoded on the same CPU float32 backend as both tuned decoders. '
                 'Do not require bitwise identity to historical GPU-decoded cache coordinates.',
        geometry='Per-member geometry first, then mean members; canonical and explicit-PSF definitions separate. '
                 'Bond mean and radius of gyration have no monotonic improvement direction.',
        test_opened=False, bioemu2_rerun=False,
        interpretation='A capacity-and-optimization intervention with384 extra parameters. Oracle improvement '
                       'does not establish forecasting improvement, JEPA pretraining benefit, or all-metric BioEmu2 superiority.')
