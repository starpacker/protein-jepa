"""Paired comparison rejects unmatched populations and preserves gap direction."""
from copy import deepcopy
from scripts.compare_lasp_bounded_paired import (paired_summary, paired_occupancy,
                                                 SHARED_METRICS, STATIC_METRICS)
import pytest


def test_temperature_weighting_with_missing_protein_cells():
    rows = [dict(protein=p, traj=p+'_R0', anchor_frame=7, horizon=1., temperature_k=t,
                 persist={'rmsf_mae': 3.}, bioemu2={'rmsf_mae': 0.})
            for p, t in [('a', 320.), ('b', 320.), ('b', 450.)]]
    baseline = dict(per_window=rows, n_samples=12, metric_version='v', n_windows=3)
    candidate = dict(per_window=deepcopy(rows), args={'n_samples': 12}, metric_version='v', n_windows=3)
    for row, value in zip(candidate['per_window'], [0., 0., 10.]):
        row['lasp'] = {'rmsf_mae': value}
    result = paired_summary(candidate, baseline, draws=100)
    metric = result['metrics']['rmsf_mae']
    assert metric['signed_error_gap'] == 5.
    assert metric['per_temperature'] == {'320.0': 0., '450.0': 10.}
    assert metric['bootstrap_valid_draws'] < 100
    with pytest.raises(ValueError, match='incomplete'):
        paired_summary(candidate, baseline, expected_windows=200)
    candidate['per_window'][0]['persist'] = {}
    with pytest.raises(ValueError, match='persistence'):
        paired_summary(candidate, baseline)



def test_paired_error_direction_and_identity_checks():
    rows = [dict(protein=p, traj=p+'_R0', anchor_frame=7, horizon=1., temperature_k=348.,
                 persist={'rmsf_mae': 3.}, bioemu2={'rmsf_mae': 2., 'rmsf_corr': .5})
            for p in ('a', 'b')]
    baseline = dict(per_window=rows, n_samples=12, metric_version='v')
    candidate = dict(per_window=deepcopy(rows), args={'n_samples':12}, metric_version='v')
    for row in candidate['per_window']:
        row['lasp'] = {'rmsf_mae': 1., 'rmsf_corr': .8}
    result = paired_summary(candidate, baseline, draws=20)
    assert result['metrics']['rmsf_mae']['signed_error_gap'] == -1
    assert abs(result['metrics']['rmsf_corr']['signed_error_gap'] + .3) < 1e-12
    candidate['per_window'][0]['anchor_frame'] = 8
    try:
        paired_summary(candidate, baseline)
    except ValueError as exc:
        assert 'population' in str(exc)
    else:
        raise AssertionError('unmatched population accepted')


def occupancy_artifacts():
    args = dict(k_states=8, seed=0, max_residues=96, context_frames=8,
                anchors_per_trajectory=1, proteins_per_temp=8,
                n_samples=12, modes=['correct'], horizons=[1,4,16,64])
    windows, static = [], {}
    for t in (320,348,379,413,450):
        static[str(t)] = []
        for p in range(8):
            for replica in range(5):
                trajectory = f'p{p}_T{t}_R{replica}'
                for h in (1,4,16,64):
                    windows.append(dict(protein=f'p{p}', traj=trajectory,
                        anchor_frame=7, horizon=h, temperature_k=t, mode='correct',
                        persist={m:1. for m in SHARED_METRICS}))
                static[str(t)].append(dict(protein=f'p{p}', trajectory=trajectory,
                    n_model_samples=48, n_ref_frames=440, n_states=8,
                    **{m:.5 for m in STATIC_METRICS}))
    candidate = dict(args=args, n_windows=800, per_window=windows, t2a_per_protein=static)
    baseline = deepcopy(candidate)
    baseline['per_protein'] = baseline.pop('t2a_per_protein')
    baseline['args']['max_frames'] = 512
    baseline['completeness'] = {'complete':True}
    return candidate, baseline


def test_occupancy_pairing_sign_and_reproducibility():
    a,b = occupancy_artifacts()
    for rows in a['t2a_per_protein'].values():
        for row in rows:
            row['population_tv'] = .25
            row['coverage'] = .75
    x = paired_occupancy(a,b,b['per_window'],draws=100)
    y = paired_occupancy(a,b,list(reversed(b['per_window'])),draws=100)
    assert x == y
    for metric in ('population_tv','coverage'):
        assert x['metrics'][metric]['signed_error_gap'] == -.25
        assert x['metrics'][metric]['ci95'] == [-.25,-.25]
    assert not x['protocol_success_assessed']
    assert not x['reference_masks_verified']
    assert x['n_trajectories'] == 200
    # Saved LASP baseline schema is accepted without relabeling it BioEmu2.
    other = deepcopy(a)
    zero = paired_occupancy(a,other,other['per_window'],draws=20)
    assert all(m['signed_error_gap']==0 for m in zero['metrics'].values())


@pytest.mark.parametrize('change', [
    'duplicate','missing','budget','settings','arm','persistence','trajectory',
    'frames','states','samples','nonfinite','evidence',
])
def test_occupancy_fail_closed(change):
    a,b = occupancy_artifacts()
    evidence = None
    if change == 'duplicate': a['per_window'][0] = deepcopy(a['per_window'][1])
    elif change == 'missing': a['per_window'].pop()
    elif change == 'budget': a['args']['n_samples'] = 24
    elif change == 'settings': b['args']['max_frames'] = 256
    elif change == 'arm': a['per_window'][0]['mode'] = 'none'
    elif change == 'persistence': a['per_window'][0]['persist']['rmsf_mae'] = 2.
    elif change == 'trajectory': a['t2a_per_protein']['320'][0]['trajectory'] = 'wrong'
    elif change == 'frames': a['t2a_per_protein']['320'][0]['n_ref_frames'] = 441
    elif change == 'states': a['t2a_per_protein']['320'][0]['n_states'] = 7
    elif change == 'samples': a['t2a_per_protein']['320'][0]['n_model_samples'] = 47
    elif change == 'nonfinite': a['t2a_per_protein']['320'][0]['population_tv'] = float('nan')
    elif change == 'evidence': evidence = {'masks_equal_reference':False}
    with pytest.raises(ValueError):
        paired_occupancy(a,b,b['per_window'],evidence,draws=20)
