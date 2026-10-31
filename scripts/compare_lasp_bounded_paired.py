"""Compare shared dynamics metrics on paired development windows, resampling proteins."""
import argparse
import json
from pathlib import Path

import numpy as np


HIGHER = {'rmsf_corr', 'weak_contact_j', 'transient_contacts_j', 'coverage'}
SHARED_METRICS = {'md_pca_w2', 'rmsf_corr', 'rmsf_mae', 'root_mean_w2',
                  'variance_w2', 'weak_contact_j', 'transient_contacts_j'}


def identity(row):
    return (row['protein'], row.get('traj', row.get('trajectory')),
            int(row['anchor_frame']), float(row['horizon']), float(row['temperature_k']))


def paired_summary(candidate, baseline, draws=10000, seed=20261024,
                   expected_windows=None, temperatures=None, horizons=None,
                   required_metrics=None):
    left = {identity(r): r for r in candidate['per_window']}
    right = {identity(r): r for r in baseline['per_window']}
    if len(left) != len(candidate['per_window']) or len(right) != len(baseline['per_window']):
        raise ValueError('duplicate window identities')
    if not left or set(left) != set(right):
        raise ValueError(f'window population differs or empty: candidate={len(left)}, baseline={len(right)}, overlap={len(set(left)&set(right))}')
    for artifact in (candidate, baseline):
        if artifact.get('n_windows', len(left)) != len(left):
            raise ValueError('declared window count differs from completed rows')
    if expected_windows is not None and (len(left) != expected_windows or
            any(a.get('n_windows') != expected_windows for a in (candidate, baseline))):
        raise ValueError('incomplete expected evaluation population')
    n_samples = candidate.get('args', {}).get('n_samples')
    if not isinstance(n_samples, int) or n_samples < 2 or n_samples != baseline.get('n_samples'):
        raise ValueError('sample budgets differ or absent')
    metric_version = candidate.get('metric_version')
    if not metric_version or metric_version != baseline.get('metric_version'):
        raise ValueError('metric versions differ or absent')
    observed_temperatures = sorted({key[-1] for key in left})
    if temperatures is not None and set(observed_temperatures) != set(temperatures):
        raise ValueError('unexpected temperature population')
    if horizons is not None and {key[-2] for key in left} != set(horizons):
        raise ValueError('unexpected horizon population')
    proteins = sorted({key[0] for key in left})
    metric_names = sorted(required_metrics or next(iter(right.values()))['bioemu2'])
    for key, row in left.items():
        model = row.get('lasp', row.get('model', {}))
        for metrics in (model, right[key]['bioemu2']):
            if not set(metric_names) <= set(metrics):
                raise ValueError('required shared dynamics metrics missing')
        a, b = row.get('persist', {}), right[key].get('persist', {})
        if not a or set(a) != set(b):
            raise ValueError('persistence reference fields missing or differ')
        for name in a:
            if not np.isfinite([a[name], b[name]]).all() or not np.isclose(a[name], b[name], rtol=1e-5, atol=1e-5):
                raise ValueError('persistence reference disagrees')
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(proteins), (draws, len(proteins)))
    results = {}
    for metric in metric_names:
        cells = {}
        for key, row in left.items():
            model = row.get('lasp', row.get('model'))
            a, b = model[metric], right[key]['bioemu2'][metric]
            if not np.isfinite([a, b]).all():
                raise ValueError('nonfinite paired metric')
            cells.setdefault((key[0], key[-1]), []).append((b-a) if metric in HIGHER else (a-b))
        matrix = np.full((len(proteins), len(observed_temperatures)), np.nan)
        per_protein = {}
        for i, protein in enumerate(proteins):
            values = {}
            for j, temperature in enumerate(observed_temperatures):
                if (protein, temperature) in cells:
                    matrix[i, j] = np.mean(cells[protein, temperature])
                    values[str(temperature)] = float(matrix[i, j])
            per_protein[protein] = values
        # Missing cells are allowed in the frozen window population. Each T uses
        # its represented proteins. Reject bootstrap draws that lose an entire T.
        sampled = matrix[resamples]
        counts = np.isfinite(sampled).sum(axis=1)
        valid = (counts > 0).all(axis=1)
        if not valid.any():
            raise ValueError('no bootstrap draws cover every temperature')
        boot = (np.nansum(sampled[valid], axis=1) / counts[valid]).mean(axis=1)
        means = np.nanmean(matrix, axis=0)
        results[metric] = {'signed_error_gap': float(means.mean()),
                           'ci95': np.quantile(boot, [.025, .975]).tolist(),
                           'one_sided_upper_983333': float(np.quantile(boot, .9833333333)),
                           'per_temperature': {str(t): float(v) for t, v in zip(observed_temperatures, means)},
                           'per_protein_per_temperature': per_protein,
                           'bootstrap_valid_draws': int(valid.sum())}
    return {'scope': 'shared dynamics metrics only; paired development comparison; negative gap favors LASP',
            'protocol_success_assessed': False,
            'aggregation': 'equal represented proteins within temperature, then equal temperatures',
            'missing_cell_policy': 'use represented proteins per temperature; discard and report bootstrap draws missing an entire temperature',
            'sample_count_evidence': 'declared artifact budgets; historical baseline lacks per-window actual counts',
            'n_proteins': len(proteins), 'n_windows': len(left),
            'baseline_provenance': {k: baseline[k] for k in ('baseline', 'sources', 'evaluation_args') if k in baseline},
            'bootstrap_draws': draws, 'bootstrap_seed': seed, 'metrics': results}


STATIC_METRICS = {'population_tv', 'population_jsd', 'population_mae',
                  'coverage', 'free_energy_mae_kcal'}


def paired_occupancy(candidate, baseline, baseline_windows, reference_evidence=None,
                     draws=10000, seed=20261024):
    """Compare completed matched800 artifacts, without inference or test access.

    Historical suites do not serialize masks or the ensemble metric version.
    Reference-mask evidence is therefore reported separately, never inferred
    from matching frame counts or Persistence alone.
    """
    settings = dict(k_states=8, seed=0, max_residues=96, context_frames=8,
                    anchors_per_trajectory=1, proteins_per_temp=8)
    temperatures = {320., 348., 379., 413., 450.}
    horizons = {1., 4., 16., 64.}
    for artifact in (candidate, baseline):
        args = artifact.get('args', {})
        if any(args.get(k) != v for k, v in settings.items()):
            raise ValueError('static reference settings differ or absent')
        if artifact.get('n_windows') != 800:
            raise ValueError('expected matched800 population')
        if 't2a_per_protein' in artifact:
            if (args.get('modes') != ['correct'] or args.get('n_samples') != 12
                    or set(args.get('horizons', [])) != horizons):
                raise ValueError('candidate sample budget or conditioning differs')
        elif (args.get('max_frames') != 512
              or artifact.get('completeness', {}).get('complete') is not True):
            raise ValueError('saved static reference incomplete or frame setting differs')
    left_rows = candidate['per_window']
    left = {identity(r): r for r in left_rows}
    right = {identity(r): r for r in baseline_windows}
    if len(left) != 800 or len(right) != 800 or len(left_rows) != 800 or len(baseline_windows) != 800:
        raise ValueError('incomplete or duplicate matched800 identities')
    if set(left) != set(right):
        raise ValueError('matched800 window population differs')
    if ({k[-1] for k in left} != temperatures or {k[-2] for k in left} != horizons
            or len({k[0] for k in left}) != 8):
        raise ValueError('unexpected matched800 conditions or proteins')
    trajectory_windows = {}
    for key, row in left.items():
        if row.get('mode') != 'correct':
            raise ValueError('matched800 requires correct arm')
        trajectory_windows.setdefault((key[0], key[1], key[-1]), []).append(key)
        a, b = row.get('persist', {}), right[key].get('persist', {})
        if not SHARED_METRICS <= set(a) or set(a) != set(b):
            raise ValueError('Persistence metrics missing or differ')
        if any(not np.isfinite([a[m], b[m]]).all() or
               not np.isclose(a[m], b[m], rtol=1e-5, atol=1e-5) for m in a):
            raise ValueError('Persistence reference disagrees')
    if len(trajectory_windows) != 200 or any(
            len(keys) != 4 or {k[-2] for k in keys} != horizons
            or len({k[2] for k in keys}) != 1 for keys in trajectory_windows.values()):
        raise ValueError('trajectory pooling differs from four horizons at one anchor')
    for temperature in temperatures:
        keys = [k for k in left if k[-1] == temperature]
        if len(keys) != 160 or len({k[0] for k in keys}) != 8:
            raise ValueError('unbalanced matched800 temperature panel')

    def static_rows(artifact):
        grouped = artifact.get('t2a_per_protein', artifact.get('per_protein'))
        if not isinstance(grouped, dict):
            raise ValueError('static trajectory rows missing')
        rows = [(r['protein'], r['trajectory'], float(t), r)
                for t, values in grouped.items() for r in values]
        result = {(p, tr, t): r for p, tr, t, r in rows}
        if len(rows) != 200 or len(result) != 200 or set(result) != set(trajectory_windows):
            raise ValueError('static trajectory population differs or duplicated')
        return result

    static_left, static_right = static_rows(candidate), static_rows(baseline)
    normalized_left, normalized_right = [], []
    for index, key in enumerate(sorted(static_left)):
        a, b = static_left[key], static_right[key]
        if (a.get('n_model_samples') != 48 or b.get('n_model_samples') != 48
                or a.get('n_states') != 8 or b.get('n_states') != 8
                or a.get('n_ref_frames') != b.get('n_ref_frames')
                or not isinstance(a.get('n_ref_frames'), int) or a['n_ref_frames'] < 8):
            raise ValueError('static sample budget or reference metadata differs')
        if not all(m in r and np.isfinite(r[m]) for r in (a, b) for m in STATIC_METRICS):
            raise ValueError('missing or nonfinite static metric')
        # The internal adapter uses one row per actual pooled trajectory, not
        # invented sampling observations. Static reference equality is above.
        common = dict(protein=key[0], trajectory=key[1], temperature_k=key[2],
                      anchor_frame=trajectory_windows[key][0][2], horizon=0.,
                      persist={'reference_frames': a['n_ref_frames']})
        normalized_left.append(dict(common, model={m: a[m] for m in STATIC_METRICS}))
        normalized_right.append(dict(common, bioemu2={m: b[m] for m in STATIC_METRICS}))
    # This label identifies the validated adapter, not a recovered historical
    # metric-version assertion. Both sides use the same static reference contract.
    version = 'validated-static-reference-adapter-v1'
    result = paired_summary(
        dict(per_window=normalized_left, args={'n_samples': 48}, metric_version=version),
        dict(per_window=normalized_right, n_samples=48, metric_version=version),
        draws=draws, seed=seed, required_metrics=STATIC_METRICS)
    result.update(scope='paired pooled static ensembles; negative gap favors candidate',
                  n_windows=800, n_trajectories=200,
                  identity_and_persistence_checks_passed=True,
                  sample_count_evidence='48 pooled samples recorded for every trajectory',
                  ensemble_metric_version_evidence='historical suite artifacts omit version; no ensemble-version compatibility claim',
                  reference_mask_evidence=reference_evidence,
                  reference_masks_verified=False,
                  limitations=['Raw masks are absent from comparison JSONs; historical evidence is recorded separately.',
                               'Candidate suite scorer checks protocol masks equal its MD reference.',
                               'Reference-frame setting512 is fixed in the suite scorer, not serialized in historical suite args.'])
    if reference_evidence is not None:
        if (reference_evidence.get('masks_equal_reference') is not True
                or reference_evidence.get('actual_windows') != 800
                or reference_evidence.get('trajectories') != 200
                or reference_evidence.get('samples_per_trajectory') != 48):
            raise ValueError('saved reference-mask evidence inconsistent')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--baseline', default='results/lasp_v19/bioemu2_complete.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--occupancy', action='store_true',
                        help='compare matched800 pooled static ensembles, not standard200')
    parser.add_argument('--baseline-windows', nargs='+',
                        help='completed ensemble JSONs paired with a static baseline')
    parser.add_argument('--reference-evidence', help='saved baseline mask-validation JSON')
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text())
    baseline = json.loads(Path(args.baseline).read_text())
    if args.occupancy:
        windows = baseline.get('per_window')
        if args.baseline_windows:
            windows = []
            for path in args.baseline_windows:
                artifact = json.loads(Path(path).read_text())
                if (artifact.get('n_samples') != 12
                        or artifact.get('n_windows') != len(artifact['per_window'])
                        or artifact.get('n_skipped', 0) != 0):
                    raise ValueError('saved baseline window budget/completeness differs')
                windows.extend(artifact['per_window'])
        if windows is None:
            parser.error('--occupancy requires baseline suite windows or --baseline-windows')
        evidence = (json.loads(Path(args.reference_evidence).read_text())
                    if args.reference_evidence else None)
        result = paired_occupancy(candidate, baseline, windows, evidence)
    else:
        if args.baseline_windows or args.reference_evidence:
            parser.error('static baseline options require --occupancy')
        result = paired_summary(candidate, baseline, expected_windows=200,
                                temperatures=[320, 348, 379, 413, 450],
                                horizons=[1, 4, 16, 64], required_metrics=SHARED_METRICS)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v['signed_error_gap'] for k, v in result['metrics'].items()}, indent=2))


if __name__ == '__main__':
    main()
