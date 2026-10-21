import copy

import torch

from evaluation.run_h_gate import CONDITIONS, _observables, assess_rows


def _rows():
    rows = []
    for p in range(4):
        for a in range(2):
            row = dict(protein_id=str(p), horizon_frames=1)
            for condition in (*CONDITIONS, 'persistence'):
                error = 0.1 if condition == 'true_future_H' else 1.0
                row[condition] = dict(rmsd=error, shape_rmsd=error,
                                      distance_change_rmse=error, contact_change_error=error)
            rows.append(row)
    return rows


def test_h_gate_requires_time_information_and_error_budget():
    rows = _rows()
    assert assess_rows(rows)['passed']
    for row in rows:
        row['wrong_time_H'] = copy.deepcopy(row['true_future_H'])
    assert not assess_rows(rows)['passed']
    rows = _rows()
    for row in rows:
        row['true_future_H']['rmsd'] = 0.5
    result = assess_rows(rows)
    assert not result['passed']
    assert not result['by_horizon']['1']['checks']['oracle_error_budget']


def test_h_gate_equal_protein_and_deterministic_bootstrap():
    rows = _rows()
    rows[0]['true_future_H']['rmsd'] = 0.3
    # Oversampling all windows of one protein must not reweight the protein.
    duplicate = copy.deepcopy([r for r in rows if r['protein_id'] == '0'])*20
    original = assess_rows(rows, seed=5)
    expanded = assess_rows(rows+duplicate, seed=5)
    assert original['by_horizon'] == expanded['by_horizon'] | {'1': {
        **expanded['by_horizon']['1'], 'windows': original['by_horizon']['1']['windows']}}
    assert original == assess_rows(rows, seed=5)
    assert not assess_rows([r for r in rows if r['protein_id'] != '3'])['passed']


def test_internal_observable_errors_are_rigid_invariant_and_detect_shape():
    x = torch.randn(9, 3, generator=torch.Generator().manual_seed(2))
    y = x.clone()
    y[4] += torch.tensor([2., 1., -1.])
    mask = torch.ones(9, dtype=torch.bool)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    before = _observables(x, y, mask)
    after = _observables(x@rotation+4, y@rotation+4, mask)
    assert before['distance_change_rmse'] > 0.1
    for key in before:
        assert abs(before[key]-after[key]) < 1e-6
