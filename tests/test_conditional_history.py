import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.probe_conditional_history as probe


def synthetic_rows(prefix, count=18, *, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for protein in range(count):
        n = 8 + protein % 4
        anchor = rng.normal(size=(n, 4))
        history = rng.normal(size=(n, 3))
        y = anchor[:, :3] * .2 + 2 * history + rng.normal(scale=.01, size=(n, 3))
        rows.append(dict(protein=f'{prefix}{protein}', trajectory='r0', temperature_k=320.,
            anchor_frame=7, horizon=4, target=y,
            features={'anchor': anchor, 'history': np.concatenate((anchor, history), axis=-1)}))
    return rows


def backbone():
    ca = torch.tensor([[0., 0., 0.], [3.8, 0., 0.], [6., 3., 0.],
                       [5., 6., 2.], [2., 8., 1.], [-1., 6., 2.]])
    coords = ca[:, None].expand(-1, 4, -1).clone()
    coords[:, 0, 1] += 1.
    coords[:, 2, 0] += 1.
    coords[:, 3, 2] += 1.
    return coords


def test_endpoint_label_removes_global_motion_and_uses_observed_local_basis():
    coords = backbone()
    mask = torch.ones(6, dtype=torch.bool)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    context = coords[None].expand(8, -1, -1, -1)
    target = coords.clone()
    target[3, :, 2] += .8
    reference = probe.endpoint_label(context, target[None], mask)
    transformed = probe.endpoint_label(context @ rotation + 5, target[None] @ rotation + 5, mask)
    assert np.allclose(reference, transformed, atol=2e-6)
    assert np.linalg.norm(reference) > .1
    assert np.linalg.norm(probe.endpoint_label(context, (coords @ rotation + 5)[None], mask)) < 2e-6


def test_features_use_all_anchor_channels_and_observed_deltas_with_masked_means():
    h = torch.arange(3 * 5 * 2, dtype=torch.float64).reshape(3, 5, 2)
    mask = torch.tensor([True, True, True, True, False])
    h[:, -1] = 99999
    features = probe.feature_arrays(h, mask, 348.)
    assert features['anchor'].shape == (4, 5)
    assert features['history'].shape == (4, 13)
    assert np.array_equal(features['history'][:, :5], features['anchor'])
    assert np.array_equal(features['anchor'][:, :2], h[-1, mask].numpy())
    assert np.allclose(features['anchor'][:, 2:4], h[-1, mask].mean(0).numpy())
    assert np.all(features['anchor'][:, -1] == 348.)
    assert np.allclose(features['history'][:, 5:7], (h[-2, mask] - h[-1, mask]).numpy())


def test_collection_encodes_only_context_and_ignores_future_path(monkeypatch):
    coords = backbone()
    context = coords[None].expand(8, -1, -1, -1).clone()
    target = coords[None].clone()
    meta = dict(residue_mask=torch.ones(8, 6, dtype=torch.bool), aatype=torch.zeros(6, dtype=torch.long),
                residue_index=torch.arange(6), chain_id=torch.zeros(6, dtype=torch.long))
    window = SimpleNamespace(context=dict(coords=context, **meta),
        target=dict(coords=target, residue_mask=torch.ones(1, 6, dtype=torch.bool)),
        future_path={'coords': 'must never be read'}, protein_id='p', trajectory_id='r',
        temperature_k=320., anchor_frame=7, target_frame=11, horizon=4, horizon_ps=4000.,
        context_frames=np.arange(8))
    class Bundle:
        def encode(self, observed, *args):
            assert torch.equal(observed, context)
            return observed[:, :, 1]
    monkeypatch.setattr(probe, 'iter_windows', lambda protocol: iter([window]))
    protocol = SimpleNamespace(manifest='synthetic', horizons_frames=(4,))
    first = probe.collect_windows(Bundle(), protocol, 1, 'cpu')[0]
    window.target['coords'][0, 3, :, 2] += 2
    second = probe.collect_windows(Bundle(), protocol, 1, 'cpu')[0]
    assert np.array_equal(first['features']['history'], second['features']['history'])
    assert not np.array_equal(first['target'], second['target'])


def test_weighted_ridge_uses_training_only_statistics_and_equal_protein_mass():
    rows = synthetic_rows('p', count=4)
    extra = copy.deepcopy(rows[0])
    extra['anchor_frame'] = 10
    x, y, weights = probe.pack_rows(rows, 'history')
    model = probe.ridge_path(x, y, weights, [.01])[.01]
    assert np.allclose(model.mean, weights @ x)
    expanded = probe.ridge_path(*probe.pack_rows(rows + [extra], 'history'), [.01])[.01]
    assert np.allclose(model.coefficients, expanded.coefficients, atol=1e-9)
    assert np.allclose(model.predict(x), expanded.predict(x), atol=1e-9)
    before = model.mean.copy(), model.scale.copy(), model.coefficients.copy()
    model.predict(x + 1000)
    assert all(np.array_equal(a, b) for a, b in zip(before, (model.mean, model.scale, model.coefficients)))
    with pytest.raises(ValueError, match='duplicate'):
        probe.pack_rows(rows + [rows[0]], 'anchor')


def test_train_protein_cv_and_one_locked_evaluation_detect_injected_history_signal():
    train, val = synthetic_rows('train'), synthetic_rows('val', seed=9)
    fitted, selection = probe.fit_probes(train, lambdas=[1e-5, .01, 10.], folds=3, seed=2)
    report = probe.evaluate_probes(fitted, val, bootstrap_reps=500, seed=8)['4']
    assert report['fractional_mse_reduction']['mean'] > .99
    assert report['paired_history_minus_anchor']['mse']['ci95'][1] < 0
    for arm in probe.ARMS:
        folds = selection['by_horizon']['4'][arm]['folds']
        assert all(not set(f['fit_proteins']) & set(f['heldout_proteins']) for f in folds)
        assert set().union(*(set(f['heldout_proteins']) for f in folds)) == {r['protein'] for r in train}
    fixed = copy.deepcopy(selection)
    changed = copy.deepcopy(val)
    for row in changed:
        row['target'][:] = 1000
        row['features']['history'][:] = -50
    probe.evaluate_probes(fitted, changed, bootstrap_reps=10)
    assert selection == fixed
    reordered = probe.evaluate_probes(fitted, val[::-1], bootstrap_reps=500, seed=8)['4']
    assert report == reordered


def test_identical_features_cannot_create_apparent_history_gain():
    train, val = synthetic_rows('train'), synthetic_rows('val', seed=9)
    for row in train + val:
        row['features']['history'] = row['features']['anchor'].copy()
    fitted, _ = probe.fit_probes(train, lambdas=[.01, 10.], folds=3)
    report = probe.evaluate_probes(fitted, val, bootstrap_reps=100)['4']
    assert report['fractional_mse_reduction'] == {'mean': 0., 'ci95': [0., 0.]}
    assert report['paired_history_minus_anchor']['mse'] == {'mean': 0., 'ci95': [0., 0.]}


def test_probe_rejects_invalid_regularization_and_incomplete_horizon_groups():
    train = synthetic_rows('train', count=4)
    with pytest.raises(ValueError, match='regularization'):
        probe.ridge_path(*probe.pack_rows(train, 'anchor'), [0.])
    with pytest.raises(ValueError, match='fold'):
        probe.fit_probes(train, lambdas=[1.], folds=5)
    train[-1]['horizon'] = 16
    with pytest.raises(ValueError, match='every training protein'):
        probe.fit_probes(train, lambdas=[1.], folds=2)
