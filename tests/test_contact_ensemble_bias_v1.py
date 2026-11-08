"""Synthetic-only checks; no frozen TRAIN32 tensor deserialization."""
import json

import pytest
import torch

from run_scripts import diagnose_contact_ensemble_bias_v1 as diagnostic


def synthetic():
    generator = torch.Generator().manual_seed(731)
    anchor = torch.randn(9, 3, generator=generator, dtype=torch.float64) * 4
    samples = anchor[None] + torch.randn(12, 9, 3, generator=generator, dtype=torch.float64) * .8
    target = anchor + torch.randn(9, 3, generator=generator, dtype=torch.float64) * .4
    mask = torch.tensor([True] * 8 + [False])
    residue_index = torch.tensor([0, 1, 2, 7, 8, 9, 10, 11, 12])
    chain_id = torch.tensor([0] * 6 + [1] * 3)
    return samples, target, anchor, mask, residue_index, chain_id


def test_canonical_pairs_nonlocal_gap_and_cross_chain():
    mask = torch.tensor([True, True, True, True, False])
    ri = torch.tensor([10, 11, 20, 11, 99])
    ci = torch.tensor([0, 0, 0, 1, 0])
    pairs = diagnostic.canonical_pairs(mask, ri, ci)
    expected = torch.zeros(5, 5, dtype=torch.bool)
    for left, right in ((0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        expected[left, right] = True
    assert torch.equal(pairs, expected)
    with pytest.raises(ValueError, match='no valid'):
        diagnostic.canonical_pairs(torch.ones(2, dtype=torch.bool), torch.arange(2), torch.zeros(2))


def test_cross_member_identity_values_and_gradients():
    from protein_jepa_frame.training.contact_ensemble_objective_v1 import contact_ensemble_brier
    z = torch.tensor([[.2, .9], [.6, .3], [.8, .7]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([.5, .8], dtype=torch.float64)
    plugin, penalty, corrected = contact_ensemble_brier(z, q)
    cross = diagnostic.cross_member_brier(z, q)
    torch.testing.assert_close(plugin, corrected + penalty, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(corrected, cross, rtol=1e-13, atol=1e-13)
    grad = lambda value: torch.autograd.grad(value, z, retain_graph=True)[0]
    torch.testing.assert_close(grad(plugin), grad(corrected) + grad(penalty), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(grad(corrected), grad(cross), rtol=1e-13, atol=1e-13)


def test_null_cosines_and_directional_sign():
    zero, one = torch.zeros(4), torch.ones(4)
    assert diagnostic.gradient_comparison(zero, one)['cosine'] is None
    assert diagnostic.gradient_comparison(one, zero)['norm_ratio'] is None
    assert diagnostic.gradient_comparison(zero, one)['change_right_along_unit_minus_left'] is None
    assert diagnostic.gradient_comparison(one, -one)['change_right_along_minus_left'] == 4
    json.dumps(diagnostic.gradient_comparison(zero, zero), allow_nan=False)


def test_endpoint_parity_input_immutability_and_weights():
    args = synthetic()
    originals = [value.clone() for value in args]
    result = diagnostic.diagnose_endpoint(*args)
    s = result['scalars']
    assert result['n_members'] == 12 and result['n_pairs'] > 0
    assert all(s[key] < 1e-10 for key in s if key.startswith('parity/'))
    assert s['contact_plugin_20'] == pytest.approx(20 * s['contact_plugin'])
    assert s['clash_40'] == pytest.approx(40 * s['clash'])
    assert s['total'] == pytest.approx(s['coordinate_u_es'] + s['contact_plugin_20'] + s['clash_40'])
    assert s['corrected_total'] == pytest.approx(s['total'] - s['variance_penalty_20'])
    for old, new in zip(originals, args):
        assert torch.equal(old, new)
        assert new.grad is None
    json.dumps(result, allow_nan=False)


def test_masked_coordinate_invariance():
    args = synthetic()
    before = diagnostic.diagnose_endpoint(*args)
    after_args = [value.clone() for value in args]
    after_args[0][:, -1] = torch.tensor([1e5, 2e5, -3e5])
    after_args[1][-1] = torch.tensor([-4e5, 5e5, 6e5])
    after_args[2][-1] = torch.tensor([7e5, -8e5, 9e5])
    after = diagnostic.diagnose_endpoint(*after_args)
    assert before['n_pairs'] == after['n_pairs']
    for key, value in before['scalars'].items():
        if value is None:
            assert after['scalars'][key] is None
        else:
            assert after['scalars'][key] == pytest.approx(value, abs=1e-9, rel=1e-8), key


def test_identical_members_zero_penalty_null_gradient_cosines():
    args = list(synthetic())
    # Saturated zero contact probabilities make the zero denominator exact.
    args[0] = args[2][None].expand(12, -1, -1).clone() * 1e4
    s = diagnostic.diagnose_endpoint(*args)['scalars']
    assert s['variance_penalty'] == 0
    assert s['gradient_norm/variance_penalty_20'] == 0
    assert s['penalty_vs/total/cosine'] is None


def test_shape_failure_closed():
    args = list(synthetic())
    args[0] = args[0][:11]
    with pytest.raises(ValueError, match='12 endpoint members'):
        diagnostic.diagnose_endpoint(*args)
    with pytest.raises(ValueError, match='metadata shape'):
        diagnostic.canonical_pairs(torch.ones(4, dtype=torch.bool), torch.arange(3), torch.arange(4))


def test_aggregate_complete_and_null_support():
    rows = [dict(metadata=dict(protein='a'), scalars=dict(x=1., z=None)),
            dict(metadata=dict(protein='a'), scalars=dict(x=3., z=2.)),
            dict(metadata=dict(protein='b'), scalars=dict(x=8., z=4.))]
    stats = diagnostic.summarize(rows)
    assert stats['x']['equal_protein_mean'] == 5
    assert stats['x']['n_windows'] == 3 and stats['x']['n_proteins'] == 2
    assert stats['z']['equal_protein_mean'] is None
    assert stats['z']['available_subset_mean'] == 3
    assert stats['z']['n_windows'] == 2 and not stats['z']['complete']


def test_registered_input_and_snapshot_mutations_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, 'REPO', tmp_path)
    monkeypatch.setattr(diagnostic, 'OUT', tmp_path/'out')
    diagnostic.OUT.mkdir()
    (tmp_path/'raw.pt').write_bytes(b'frozen raw fixture, never deserialized')
    (tmp_path/'source.py').write_text('# source\n')
    snapshot = diagnostic.OUT/'source_snapshot'
    snapshot.mkdir()
    (snapshot/'source.py').write_bytes((tmp_path/'source.py').read_bytes())
    protocol = dict(input_hashes={'raw.pt':diagnostic.sha(tmp_path/'raw.pt')},
                    source_hashes={'source.py':diagnostic.sha(tmp_path/'source.py')}, historical_source_hashes={})
    diagnostic.write_new(diagnostic.OUT/'protocol.json', protocol)
    digest = diagnostic.sha(diagnostic.OUT/'protocol.json')
    diagnostic.verify_registered(protocol, digest)
    (tmp_path/'raw.pt').write_bytes(b'changed raw')
    with pytest.raises(ValueError, match='registered file changed: raw.pt'):
        diagnostic.verify_registered(protocol, digest)
    (tmp_path/'raw.pt').write_bytes(b'frozen raw fixture, never deserialized')
    (snapshot/'source.py').write_text('# changed snapshot\n')
    with pytest.raises(ValueError, match='snapshot changed'):
        diagnostic.verify_registered(protocol, digest)


def test_registration_rejects_changed_raw_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, 'REPO', tmp_path)
    monkeypatch.setattr(diagnostic, 'OUT', tmp_path/'out')
    monkeypatch.setattr(diagnostic, 'cpu_setup', lambda: None)
    raw = tmp_path/'raw.pt'
    raw.write_bytes(b'raw')
    row = dict(metadata=dict(step=1), raw_tensors=dict(path='raw.pt', sha256='0'*64))
    monkeypatch.setattr(diagnostic, 'historical_metadata', lambda: ([row], {'source_hashes':{}}))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('registration must never deserialize tensors'))
    with pytest.raises(ValueError, match='historical raw changed'):
        diagnostic.register()
    assert not diagnostic.OUT.exists()


def test_write_new_refuses_overwrite(tmp_path):
    path = tmp_path/'report.json'
    diagnostic.write_new(path, dict(a=1))
    with pytest.raises(FileExistsError):
        diagnostic.write_new(path, dict(a=2))
    assert json.loads(path.read_text()) == {'a':1}


def test_all_failed_windows_keep_full_null_schema():
    row = diagnostic.failure_row('synthetic nonfinite')
    row['metadata'] = dict(protein='a')
    assert set(row['scalars']) == set(diagnostic.SCALAR_NAMES)
    summary = diagnostic.summarize([row])
    assert set(summary) == set(diagnostic.SCALAR_NAMES)
    assert all(value['equal_protein_mean'] is None and value['n_windows'] == 0
               and value['expected_windows'] == 1 for value in summary.values())
    args = list(synthetic())
    args[0][0, 0, 0] = float('nan')
    with pytest.raises(diagnostic.NumericalFailure, match='nonfinite'):
        diagnostic.diagnose_endpoint(*args)


def test_successful_registration_never_loads_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, 'REPO', tmp_path)
    monkeypatch.setattr(diagnostic, 'OUT', tmp_path/'out')
    monkeypatch.setattr(diagnostic, 'cpu_setup', lambda: None)
    for path in (diagnostic.REPORT, diagnostic.PROTOCOL, *diagnostic.SOURCES, 'raw.pt'):
        destination = tmp_path/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b'synthetic fixture')
    monkeypatch.setattr(diagnostic, 'REPORT_SHA', diagnostic.sha(tmp_path/diagnostic.REPORT))
    monkeypatch.setattr(diagnostic, 'PROTOCOL_SHA', diagnostic.sha(tmp_path/diagnostic.PROTOCOL))
    row = dict(metadata=dict(step=1), raw_tensors=dict(path='raw.pt', sha256=diagnostic.sha(tmp_path/'raw.pt')))
    monkeypatch.setattr(diagnostic, 'historical_metadata', lambda: ([row], {'source_hashes':{}}))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('registration must never deserialize tensors'))
    diagnostic.register()
    assert (diagnostic.OUT/'protocol.json').is_file()
    with pytest.raises(ValueError, match='refuse existing'):
        diagnostic.register()


def test_identical_nonsaturated_members_roundoff_only_penalty():
    args = list(synthetic())
    args[0] = args[0][0:1].expand(12, -1, -1).clone()
    s = diagnostic.diagnose_endpoint(*args)['scalars']
    assert abs(s['variance_penalty']) < 1e-28
    assert s['gradient_norm/variance_penalty_20'] < 1e-14
    # No threshold or clamp is applied to manufacture an exact zero cosine.


def test_run_records_input_failure_before_tensor_load(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, 'REPO', tmp_path)
    monkeypatch.setattr(diagnostic, 'OUT', tmp_path/'out')
    monkeypatch.setattr(diagnostic, 'cpu_setup', lambda: None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    diagnostic.OUT.mkdir()
    (tmp_path/'raw.pt').write_bytes(b'changed')
    protocol = dict(schema_version=diagnostic.VERSION,
                    input_hashes={'raw.pt':'0'*64}, source_hashes={}, historical_source_hashes={})
    diagnostic.write_new(diagnostic.OUT/'protocol.json', protocol)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('must verify before tensor load'))
    with pytest.raises(ValueError, match='registered file changed'):
        diagnostic.run()
    execution = json.loads((diagnostic.OUT/'execution.json').read_text())
    started = json.loads((diagnostic.OUT/'execution.started.json').read_text())
    assert execution['returncode'] == 1 and not execution['frozen_after_failure']['passed']
    assert execution['pid'] == started['pid'] and started['command']
    with pytest.raises(ValueError, match='no automatic restart'):
        diagnostic.run()
