import copy
import math

import numpy as np
import pytest
import torch

from protein_jepa_frame.training import frozen_h_predictor_readout_v1 as api


def row(index=400, L=5):
    g = torch.Generator().manual_seed(17)
    mask = torch.ones(L, dtype=torch.bool); mask[-1] = False
    history = torch.randn(1, 8, L, 128, generator=g); history[:, :, ~mask] = 0
    target = torch.randn(1, L, 128, generator=g); target[:, ~mask] = 0
    pairs = torch.tensor([[0, 2], [1, 3]])
    td = torch.tensor([3., 4.]); od = torch.tensor([2., 6.])
    return dict(cache_row_index=index, metadata=dict(protein=f'p{index}', horizon=api.HORIZONS[index % 4]),
        split='fit' if index < 400 else 'readout_heldout', history_H=history, target_H=target,
        mask=mask, pairs=pairs, target_distance=td, target_log=td.log1p(), observed_distance=od,
        persistence_log=od.log1p(), lag_ps=float(api.HORIZONS[index % 4]*1000), temp=300.)


def tiny_rows():
    base = row(0)
    rows = []
    for i in range(400):
        r = dict(base, cache_row_index=i, metadata=dict(protein=f'p{i}', horizon=api.HORIZONS[i % 4]),
                 lag_ps=float(api.HORIZONS[i % 4]*1000))
        rows.append(r)
    return rows


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.tensor(.1))
    def forward(self, H, mask, pairs):
        return self.w.expand(len(pairs))


def records():
    rows = []
    keys = [f'{s}/{r}' for s in api.HEAD_SEEDS for r in api.ROLES] + ['CA_persistence']
    for i in range(100):
        scores = {}
        for j, key in enumerate(keys):
            mse = 1 + i/100 + j/10
            scores[key] = dict(n_pairs=i+1, negative_count=i//2, log_mse=mse/4,
                mae_angstrom=mse/3, mse_angstrom2=mse, rmse_angstrom=math.sqrt(mse))
        rows.append(dict(cache_row_index=400+i, metadata=dict(protein=f'p{i}', horizon=api.HORIZONS[i % 4]),
                         n_pairs=i+1, scores=scores))
    return rows


def test_seed_initialization_preserves_rng_and_frozen_cpu_parity():
    rng = torch.get_rng_state().clone()
    head = api.new_head(42); old = api.old.new_probe(42); other = api.new_head(43)
    assert torch.equal(rng, torch.get_rng_state())
    assert api.state_sha256(head) == api.state_sha256(old)
    assert api.state_sha256(head) != api.state_sha256(other)
    assert head.calibration_seed == 42 and other.calibration_seed == 43
    r = row(); h1 = r['target_H'].clone().requires_grad_(); h2 = h1.detach().clone().requires_grad_()
    p1 = head(h1, r['mask'], r['pairs']); p2 = old(h2, r['mask'], r['pairs'])
    assert torch.equal(p1, p2)
    g1 = torch.autograd.grad(p1.square().sum(), (h1, *head.parameters()))
    g2 = torch.autograd.grad(p2.square().sum(), (h2, *old.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(g1, g2))


@pytest.mark.parametrize('seed', [0, 44, True])
def test_bad_seed(seed):
    with pytest.raises(ValueError): api.new_head(seed)


def test_frozen_head_backprop_to_H_only_and_padding():
    head = api.new_head().eval().requires_grad_(False); r = row()
    H = r['target_H'].clone().requires_grad_()
    head(H, r['mask'], r['pairs']).sum().backward()
    assert H.grad[:, r['mask']].abs().sum() > 0
    assert (H.grad[:, ~r['mask']] == 0).all()
    assert all(p.grad is None for p in head.parameters())


@pytest.mark.parametrize('field', ['lag_ps', 'temp', 'target_log', 'persistence_log', 'history_H', 'split', 'pairs'])
def test_row_faults(field):
    r = row()
    if field == 'lag_ps': r[field] = 1001.
    elif field == 'temp': r[field] = float('nan')
    elif field in ('target_log', 'persistence_log'): r[field] = r[field]+.01
    elif field == 'history_H': r[field][0, 0, -1, 0] = 1.
    elif field == 'split': r[field] = 'fit'
    else: r[field] = r[field].flip(0)
    with pytest.raises(ValueError): api.validate_row(r, split='score')


def test_fit_fixed500_order_states_rng_and_freeze(monkeypatch):
    rows = tiny_rows(); calls = []
    monkeypatch.setattr(api, 'new_head', lambda seed: Tiny())
    before = api._fingerprint(rows); rng = torch.get_rng_state().clone()
    head, audit, payload = api.fit_head(rows, seed=42, emit=calls.append)
    assert [r['cache_row_index'] for r in calls] == list(range(400))+list(range(100))
    assert audit['optimizer_steps'] == 500 and len(calls) == 500
    assert all(float(s['step']) == 500 for s in payload['optimizer']['state'].values())
    assert not head.training and not any(p.requires_grad for p in head.parameters())
    assert api._fingerprint(rows) == before and torch.equal(rng, torch.get_rng_state())
    assert payload['initial_head']['w'].item() != payload['head']['w'].item()


def test_fit_rejects_holdout_before_head_creation(monkeypatch):
    rows = tiny_rows(); rows[-1] = row(400)
    monkeypatch.setattr(api, 'new_head', lambda seed: pytest.fail('must not construct'))
    with pytest.raises(ValueError): api.fit_head(rows, seed=42)


def test_fit_failure_keeps_original_and_detached_optimizer(monkeypatch):
    live = Tiny(); monkeypatch.setattr(api, 'new_head', lambda seed: live)
    original = RuntimeError('emit failure')
    def emit(record): raise original
    with pytest.raises(RuntimeError) as e: api.fit_head(tiny_rows(), seed=42, emit=emit)
    assert e.value is original
    p = e.value.readout_failure_payload
    assert p['completed_updates'] == 1
    value = p['head']['w'].clone()
    with torch.no_grad(): live.w.add_(10.)
    assert torch.equal(p['head']['w'], value)
    assert all(float(s['step']) == 1 for s in p['optimizer']['state'].values())


def test_fit_detects_emit_input_mutation(monkeypatch):
    rows = tiny_rows(); monkeypatch.setattr(api, 'new_head', lambda seed: Tiny())
    def emit(record):
        if record['step'] == 500: rows[0]['metadata']['protein'] = 'changed'
    with pytest.raises(ValueError, match='mutated inputs') as e: api.fit_head(rows, seed=42, emit=emit)
    assert e.value.readout_failure_payload['completed_updates'] == 500


def test_score_all_roles_labels_negative_predictions_and_no_mutation():
    r = row(); heads = {s: api.new_head(s).eval().requires_grad_(False) for s in api.HEAD_SEEDS}
    with torch.no_grad():
        for head in heads.values():
            for p in head.parameters(): p.zero_()
            head.net[-1].bias.fill_(-.2)
    predictions = dict(latent=r['target_H'].clone(), physical=r['target_H'].clone()+0,
                       raw_initial=r['history_H'][:, -1].clone())
    before = api._tree_hash(dict(row=r, predictions=predictions))
    record, arrays = api.score_row(heads, r, predictions)
    assert len(record['scores']) == 11 and len(record['latent_audits']) == 5
    assert record['latent_audits']['latent']['endpoint_mse'] == 0
    for s in api.HEAD_SEEDS:
        for role in api.ROLES: assert record['scores'][f'{s}/{role}']['negative_count'] == 2
    pred = np.expm1(arrays['seed43__physical__predicted_log'].astype(np.float64))
    assert record['scores']['43/physical']['mae_angstrom'] == pytest.approx(np.abs(pred-arrays['target_distance']).mean())
    assert np.array_equal(arrays['target_log'], r['target_log'].numpy())
    assert api._tree_hash(dict(row=r, predictions=predictions)) == before
    assert all(not v.dtype.hasobject for v in arrays.values())


def test_score_requires_raw_initial_and_correct_head_lineage():
    r = row(); heads = {s: api.new_head(s).eval().requires_grad_(False) for s in api.HEAD_SEEDS}
    p = dict(latent=r['target_H'], physical=r['target_H'])
    with pytest.raises(ValueError): api.score_row(heads, r, p)
    p['raw_initial'] = r['history_H'][:, -1]
    heads[43] = heads[42]
    with pytest.raises(ValueError, match='seed lineage'): api.score_row(heads, r, p)


def test_aggregate_equal_protein_not_pairweighted_or_meanroots():
    rows = records(); out = api.aggregate_scores(rows)
    vals = np.array([r['scores']['43/physical']['mse_angstrom2'] for r in rows])
    assert out['overall']['43/physical']['rmse_angstrom'] == math.sqrt(vals.mean())
    assert not math.isclose(math.sqrt(vals.mean()), np.sqrt(vals).mean(), abs_tol=1e-8)
    assert out['h1']['CA_persistence']['n_proteins'] == 25
    assert len(out) == 5 and all(len(v) == 11 for v in out.values())


def test_bootstrap_fixed_draws_sign_and_rmse_independent_numpy():
    rows = records(); before = np.random.get_state()
    out = api.paired_bootstrap(rows)
    assert out['primary'] == '43/physical-minus-latent/log_mse' and len(out['contrasts']) == 14
    rng = np.random.default_rng(20261102)
    ids = np.concatenate([np.arange(h, 100, 4)[rng.integers(0, 25, (10000, 25))] for h in range(4)], 1)
    a = np.array([r['scores']['43/physical']['mse_angstrom2'] for r in rows])
    b = np.array([r['scores']['43/latent']['mse_angstrom2'] for r in rows])
    ref = np.sqrt(a[ids].mean(1))-np.sqrt(b[ids].mean(1))
    got = out['contrasts']['43/physical-minus-latent']['rmse_angstrom']
    assert got['difference'] == math.sqrt(a.mean())-math.sqrt(b.mean())
    np.testing.assert_array_equal(got['interval95'], np.quantile(ref, [.025, .975], method='linear'))
    assert got['difference'] > 0
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize('fault', ['order', 'duplicate_protein', 'metric', 'roles', 'pairs'])
def test_record_faults(fault):
    rows = records()
    if fault == 'order': rows.reverse()
    elif fault == 'duplicate_protein': rows[1]['metadata']['protein'] = rows[0]['metadata']['protein']
    elif fault == 'metric': rows[0]['scores']['43/physical']['log_mse'] = float('inf')
    elif fault == 'roles': del rows[0]['scores']['42/raw_initial']
    else: rows[0]['scores']['43/physical']['n_pairs'] += 1
    with pytest.raises(ValueError): api.aggregate_scores(rows)


def test_new_head_rejects_non_fp32_default_before_construction():
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match='default CPU FP32'):
            api.new_head(42)
    finally:
        torch.set_default_dtype(original)
