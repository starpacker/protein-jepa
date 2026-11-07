"""Freeze statistical definitions before the ordered-path training pilot.

The numerical helpers are frozen here. The later provenance/summary wrapper
may be implemented separately without changing these estimands or support.
This registration opens no model score JSON and never launches a job.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts import hs_ordered_pilot_statistics_v1 as numeric
from scripts import summarize_hs_target_pilot as old

BASE = Path('results/hs_joint_20261101/ordered_path_v1')
ARTIFACTS = Path('artifacts/hs_joint_20261101/ordered_path_v1')
VERSION = 'hs_ordered_path_pilot_statistics_v1'
SOURCES = (
    'scripts/register_hs_ordered_pilot_statistics_v1.py',
    'scripts/hs_ordered_pilot_statistics_v1.py',
    'tests/test_hs_ordered_pilot_statistics_v1.py',
    'scripts/summarize_hs_target_pilot.py',
    'scripts/summarize_hs_readout_pilot.py',
    'scripts/summarize_ca_ordered_paths.py',
)


def contract():
    base = old.statistical_contract()
    return dict(schema_version=VERSION, role='single_seed_adaptive_development_pilot',
        arms=list(numeric.ARMS), contrast='joint_path minus marginal_path',
        primary=dict(metric='coordinate_path_energy_score', horizons=[16, 64],
            n_windows=100, n_proteins=50, unit='angstrom', better='lower',
            definition='Within each protein, average its two complete joint-path U-scores at 16/64 ns, then average paired differences equally across all 50 proteins.'),
        bootstrap=base['bootstrap'], groups=base['groups'],
        ordered_path_scores=list(numeric.SCORES),
        ordered_comparisons=[numeric.CONTRAST, 'joint_path_vs_persistence', 'marginal_path_vs_persistence'],
        ordered_intervals=60, primary_intervals=1, standard_intervals=341,
        total_paired_intervals=402,
        standard_endpoint_secondary='The previous long-horizon endpoint U-score primary is retained as a labelled secondary, alongside every standard group. It is not a second primary.',
        standard_endpoint_scores=base['secondary_endpoint_scores'],
        endpoint_units=base['endpoint_units'], endpoint_estimator=base['endpoint_estimator'],
        historical_seven_metrics=base['historical_seven_metrics'],
        historical_metric_directions=base['historical_metric_directions'],
        historical_reference=base['historical_reference'],
        canonical_ca_geometry=base['canonical_ca_geometry'], geometry_directions=base['geometry_directions'],
        standard_descriptive_means=670, suite=base['suite'],
        ordered_descriptive=dict(roles=list(numeric.paths.ARMS),
            scalars=[*numeric.SCORES, *numeric.paths.DESCRIPTIVE, 'distance_acf_corr_n_lags'],
            lags=list(numeric.paths.LAGS),
            means='Use each window\'s defined ensemble scalar, including U-pair correction or pooled walk RMSF; then equal windows within protein and equal proteins. Per-member diagnostics retain metric-specific reductions.',
            missing='Report null complete-panel means if any scalar is missing. Separately report available support means, counts and missing identities. Pairing never drops a window.',
            deterministic_persistence_correlation='walk_rmsf_corr undefined; raw historical numerical convention remains in the hashed source output.',
            raw='Retain all per-member diagnostics, curves, permutations, input fingerprints and lag pair counts in immutable raw results.',
            inference='Only the two joint-path U-scores receive ordered-score CIs. ACF/MSD/path length/reverse/shuffle are descriptive, with no assigned monotonic quality direction.'),
        training=dict(optimizer_steps=500, seed=42, n_samples=12, grad_steps=64,
            coordinate='0.5 endpoint U-ES + 0.5 joint-path U-ES, or + 0.5 mean-time marginal U-ES',
            identical_future_labels_and_sampling_contract=True,
            remaining_terms='Endpoint latent .1 + endpoint soft-contact Brier20 + endpoint clash40; all prior zero-weight terms stay zero.',
            frozen_encoder_decoder=True, selection='latest exactly500; no training-time validation, best selection, resume or restart',
            old_endpoint='Historical reference only; its training included validation calls absent in both new arms. No assertion of complete global-RNG equality to the historical run.'),
        implementation='Frozen numerical helper calls the old endpoint analysis through explicit positional-slot renaming; new primary and ordered aggregation are defined here. A separate provenance/summary wrapper will be hashed when implemented.',
        scientific_limits=[*base['scientific_limits'],
            'Joint versus marginal training compares time dependence supervision given identical future labels; old path_mixture is not this matched control.',
            'Energy scores can have weak finite-budget sensitivity to high-dimensional dependence; properness does not guarantee temporal diagnostic improvement.',
            'The total objective contains a clash penalty and is not a pure proper-scoring objective.',
            'The current H/CA readout gate remains unchanged; improvement cannot certify latent information sufficiency.',
            'No continuous192ns or MSM/T2B panel is added in this pilot; the full-goal gaps remain.'],
        test_opened=False, bioemu2_rerun=False)


def register():
    output = old.local(BASE/'summary.protocol.json')
    old.require(not output.exists(), 'refusing registration overwrite')
    for arm in numeric.ARMS:
        old.require(not old.local(BASE/f'{arm}.execution.json').exists()
                    and not old.local(ARTIFACTS/arm/'checkpoint_latest.pt').exists(),
                    'training already started before statistics registration')
        for kind in ('ensemble', 'endpoint', 'ordered', 'suite', 'matched8'):
            old.require(not old.local(BASE/f'{arm}.{kind}.json').exists(),
                        'formal model score exists before statistics registration')
    inputs = old.Inputs()
    for source in SOURCES: inputs.track(source)
    inputs.verify()
    record = dict(contract(), registered_utc=datetime.now(timezone.utc).isoformat(),
        registration_stage='Before both new-arm training and any new formal evaluation score. Old target and Markov development results exist; this is not unseen confirmation.',
        source_hashes={old.label(p): d for p, d in inputs.hashes.items()},
        output=str(BASE/'summary.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(output), sha256=old.sha(output))))


if __name__ == '__main__':
    register()
