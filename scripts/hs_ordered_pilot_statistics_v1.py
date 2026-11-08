"""Statistics for matched joint-path versus time-marginal training.

Pure aggregation of audited window dictionaries. No models, sampling, label
reads, model selection, or changes to the frozen endpoint estimators.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from scripts import summarize_ca_ordered_paths as paths
from scripts import summarize_hs_target_pilot as endpoint

ARMS = ('joint_path', 'marginal_path')
CONTRAST = 'joint_path_vs_marginal_path'
SCORES = paths.SCORES
require = endpoint.require


def rename_standard(value):
    """Frozen endpoint helper uses positional slots, never target semantics."""
    names = dict(zip(endpoint.ARMS, ARMS))
    names['endpoint_vs_path_mixture'] = CONTRAST
    names['primary_long_horizon_coordinate_UES'] = 'secondary_long_horizon_endpoint_UES'
    if isinstance(value, dict):
        return {names.get(k, k): rename_standard(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rename_standard(v) for v in value]
    return value


def standard_analysis(ensemble, endpoints, baseline):
    require(set(ensemble) == set(endpoints) == set(ARMS), 'training arms differ')
    slots = dict(zip(ARMS, endpoint.ARMS))
    result = endpoint.analysis({slots[a]: ensemble[a] for a in ARMS},
                              {slots[a]: endpoints[a] for a in ARMS}, baseline)
    return rename_standard(result)


def observed_reference_pairing(rows):
    require(set(rows) == set(ARMS), 'ordered training arms differ')
    first, second = (rows[a] for a in ARMS)
    require(set(first) == set(second) and bool(first), 'ordered keys differ')
    for key in first:
        a, b = first[key], second[key]
        for field in ('target_frame', 'context_frames', 'horizon_ps', 'valid_residues',
                      'time_grid', 'n_forecast_paths', 'n_reference_paths',
                      'sampling_seed', 'temporal_controls'):
            require(a[field] == b[field], f'ordered reference contract differs: {field}')
        for field in ('observed', 'reference'):
            require(a['input_fingerprints'][field] == b['input_fingerprints'][field],
                    f'ordered input tensor differs: {field}')
        for role in ('persistence', 'reference_MD'):
            require(a['arms'][role] == b['arms'][role], 'ordered reference values differ')
        for role in paths.ARMS:
            ac, bc = (x['arms'][role]['metrics'] for x in (a, b))
            require(ac['conventions'] == bc['conventions'], 'ordered metric conventions differ')
            require(ac['lag_statistics']['reference'] == bc['lag_statistics']['reference'],
                    'ordered reference lag values differ')
    return sorted(first)


def descriptive(rows, keys, getter):
    """Do not present a changing available subset as the complete panel."""
    require(bool(keys) and len(keys) == len(set(keys)), 'empty or duplicate descriptive support')
    require(all(k in rows for k in keys), 'missing descriptive window')
    by_protein, missing = defaultdict(list), []
    for key in sorted(keys):
        value = getter(rows[key])
        require(paths.finite_or_none(value), 'nonfinite descriptive metric')
        if value is None:
            missing.append(list(key))
        else:
            by_protein[key[0]].append(float(value))
    means = {p: float(np.mean(v)) for p, v in sorted(by_protein.items())}
    available = float(np.mean(list(means.values()))) if means else None
    return dict(complete_panel_mean=available if not missing else None,
                available_support_mean=available, n_windows=len(keys),
                n_valid_windows=len(keys)-len(missing),
                n_proteins=len({k[0] for k in keys}), n_valid_proteins=len(means),
                missing_window_keys=missing, per_protein_available_means=means)


def scalar(row, role, metric):
    # A deterministic path has no ensemble RMSF correlation; preserve the raw
    # legacy scalar separately rather than treating its numerical convention as
    # an interpretable correlation.
    if (role, metric) == ('persistence', 'walk_rmsf_corr'):
        return None
    return row['arms'][role]['metrics']['scalars'][metric]


def ordered_analysis(rows):
    keys = observed_reference_pairing(rows)
    groups = endpoint.groups_for(keys)
    long_keys = [k for k in keys if k[4] in (16, 64)]
    counts = Counter(k[0] for k in long_keys)
    require(len(long_keys) == 100 and len(counts) == 50 and set(counts.values()) == {2},
            'primary requires both long horizons for 50 proteins')
    get_score = lambda row, metric: scalar(row, 'forecast', metric)
    primary = endpoint.paired(rows[ARMS[0]], rows[ARMS[1]], long_keys,
        lambda r: get_score(r, SCORES[0]), lambda r: get_score(r, SCORES[0]), 'lower')
    primary.update(metric=SCORES[0], horizons=[16, 64], unit='angstrom', per_protein=[])
    for protein in sorted(counts):
        selected = [k for k in long_keys if k[0] == protein]
        values = [float(np.mean([get_score(rows[a][k], SCORES[0]) for k in selected])) for a in ARMS]
        primary['per_protein'].append(dict(protein=protein, candidate=values[0], reference=values[1],
            candidate_minus_reference=values[0]-values[1], n_windows=2))
    paired = {CONTRAST: endpoint.comparisons(rows[ARMS[0]], rows[ARMS[1]], groups, SCORES,
                                          (get_score, get_score), {m: 'lower' for m in SCORES})}
    for arm in ARMS:
        paired[arm+'_vs_persistence'] = endpoint.comparisons(rows[arm], rows[arm], groups, SCORES,
            (get_score, lambda r, m: scalar(r, 'persistence', m)), {m: 'lower' for m in SCORES})
    reports = {}
    for arm in ARMS:
        reports[arm] = {}
        for group, selected in groups.items():
            roles = {}
            for role in paths.ARMS:
                values = {m: descriptive(rows[arm], selected,
                    lambda r, m=m, role=role: scalar(r, role, m))
                    for m in (*SCORES, *paths.DESCRIPTIVE, 'distance_acf_corr_n_lags')}
                lags = {}
                for index, lag in enumerate(paths.LAGS):
                    get = lambda r, index=index, role=role: r['arms'][role]['metrics']['lag_statistics']['prediction']['aggregate'][index]
                    lags[str(lag)] = dict(
                        acf=descriptive(rows[arm], selected, lambda r: get(r)['acf']),
                        msd=descriptive(rows[arm], selected, lambda r: get(r)['msd']),
                        n_time_pairs=sum(get(rows[arm][k])['n_pairs'] for k in selected),
                        n_acf_time_pairs=sum(get(rows[arm][k])['n_acf_pairs'] for k in selected))
                roles[role] = dict(scalars=values, lag_diagnostics=lags)
            reports[arm][group] = roles
    return dict(primary_long_horizon_joint_path_UES=primary,
        ordered_path_scores=dict(task_version='ca_ordered_path_metrics_v2', units=dict.fromkeys(SCORES, 'angstrom'),
            comparisons=paired, descriptive=reports,
            scalar_averaging='Use each window\'s defined ensemble scalar, including U-pair correction or pooled walk RMSF; then equal windows within protein and equal proteins. Per-member diagnostics retain metric-specific reductions.',
            missing_support='A missing scalar yields a null complete-panel mean; available-support means and identities are separate. No pairwise deletion for U-score CIs.',
            temporal_controls='Reverse/shuffle are descriptive permutations, not fitted competing models; they change time marginals and possibly endpoints.',
            persistence_correlation='walk_rmsf_corr is undefined for deterministic persistence; original raw scalars remain in the hashed raw JSON.',
            interpretation='ACF/MSD and path-length diagnostics are not assigned a monotonic quality direction or additional CIs. No equilibrium relaxation-time or S-memory claim.'))
