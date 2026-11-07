"""CPU mathematical and bookkeeping checks; no research cache/results are read."""
import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.diagnose_fixed_ca_capacity import (
    ABSOLUTE_MARGIN_ANGSTROM, CACHE, REPO, ROUTES, aggregate_rows,
    official_template_radii, peptide_lower_bound, prepare_edges,
    protocol, score_ca, score_row, source_hashes, verify_official_reconstruction, write_json_exclusive,
)


def row(length=3):
    ca = torch.zeros(1, length, 3, dtype=torch.float32)
    ca[0, :, 0] = torch.arange(length) * 6.
    bb = ca[..., None, :].expand(1, length, 4, 3).clone()
    return dict(metadata=dict(protein='toy', horizon=1),
                observed_mask=torch.ones(length, dtype=torch.bool),
                chemical_aatype=torch.zeros(length, dtype=torch.long),
                geometry_row_ids=torch.arange(length)+100,
                peptide_edges=torch.stack((torch.arange(length-1), torch.arange(1, length)), -1),
                atom_mask=torch.ones(length, 4, dtype=torch.bool),
                target_atom_mask=torch.ones(length, 4, dtype=torch.bool),
                observed_R=torch.eye(3).expand(1, length, 3, 3).clone(),
                oracle_CA=ca, forecast_CA=ca.expand(12, length, 3).clone(),
                observed_backbone=bb, target_backbone=bb.clone(), topology=dict(sha256='synthetic'))


def test_two_point_bound_is_attained_for_facing_atoms_and_never_exceeds_penalty():
    # Independent residue rotations can align these two radius vectors exactly.
    left = torch.tensor([0., 0., 0.], dtype=torch.float64)
    right = torch.tensor([6., 0., 0.], dtype=torch.float64)
    r_c, r_n, mu, std = 1.5, 1.4, 1.329, .014
    upper = mu + 12*std
    bound = peptide_lower_bound(left, right, r_c, r_n, upper)
    c, n = left+left.new_tensor([r_c, 0., 0.]), right-left.new_tensor([r_n, 0., 0.])
    penalty = torch.relu((n-c).norm()-upper)
    assert torch.allclose(bound['raw'], penalty, atol=1e-14, rtol=0)
    assert bound['impossible'] and bound['conservative'] < bound['raw']
    generator = torch.Generator().manual_seed(4)
    u = torch.randn(200, 3, generator=generator, dtype=torch.float64)
    v = torch.randn(200, 3, generator=generator, dtype=torch.float64)
    c = left+r_c*u/u.norm(dim=-1, keepdim=True)
    n = right+r_n*v/v.norm(dim=-1, keepdim=True)
    exact_penalty = torch.relu(((n-c).norm(dim=-1)-mu).abs()-12*std)
    assert bool((exact_penalty >= bound['raw']-1e-13).all())
    assert peptide_lower_bound(left, left+left.new_tensor([3., 0., 0.]), r_c, r_n, upper)['raw'] == 0


def test_common_rotation_preserves_raw_bound_and_margin_and_translation_only_changes_allowance():
    left = torch.tensor([[2., -1., .5], [0., 3., 1.]], dtype=torch.float64)
    right = left+torch.tensor([7., 2., -1.], dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.tensor([[1., 2., 4.], [2., 1., 1.], [3., 3., 1.]], dtype=torch.float64))
    first = peptide_lower_bound(left, right, 1.5, 1.4, 1.497)
    rotated = peptide_lower_bound(left@q, right@q, 1.5, 1.4, 1.497)
    translated = peptide_lower_bound(left+100., right+100., 1.5, 1.4, 1.497)
    assert torch.allclose(first['raw'], rotated['raw'], atol=1e-14, rtol=0)
    assert torch.allclose(first['margin'], rotated['margin'], atol=1e-16, rtol=0)
    assert torch.allclose(first['raw'], translated['raw'], atol=1e-14, rtol=0)
    assert bool((translated['margin'] > first['margin']).all())


def test_official_identity_template_and_rotated_reconstruction_preserve_ca_and_radii():
    r_c, r_n = official_template_radii()
    assert r_c.dtype == r_n.dtype == torch.float64
    assert r_c.shape == r_n.shape == (21,)
    # UNK invokes the official GLY fallback, never an invented unknown prior.
    assert r_c[20] == r_c[7] and r_n[20] == r_n[7]
    verification = verify_official_reconstruction(r_c, r_n)
    assert verification['ca_bitwise_preserved'] is True
    assert verification['radius_max_absolute_error_angstrom'] < ABSOLUTE_MARGIN_ANGSTROM
    json.dumps(protocol(REPO/CACHE, {}, verification), allow_nan=False)


def test_next_unknown_excluded_proline_threshold_correct_and_missing_atoms_not_bridged():
    item = row(4)
    item['chemical_aatype'] = torch.tensor([20, 14, 20, 0])
    aa, edges, counts = prepare_edges(item)
    assert edges.tolist() == [[0, 1], [2, 3]]
    assert counts['n_unknown_next_edges_excluded'] == 1
    assert counts['n_eligible_previous_unknown_edges'] == 2
    assert counts['n_eligible_next_proline_edges'] == 1
    rc, rn = torch.full((21,), 1.5), torch.full((21,), 1.4)
    scored = score_ca(item['oracle_CA'], aa, edges, counts, rc, rn)
    expected = ((6-1.5-1.4-1.533)+(6-1.5-1.4-1.497))/2
    assert scored['mean_lower_bound_angstrom'] == pytest.approx(expected, abs=1e-7)
    item['atom_mask'][1, 0] = False
    _, edges, counts = prepare_edges(item)
    assert edges.tolist() == [[2, 3]]
    assert counts['n_masked_or_missing_edges'] == 1


def test_zero_eligible_edges_are_undefined_not_zero_and_input_unchanged():
    item = row(2)
    item['chemical_aatype'][1] = 20
    before = copy.deepcopy(item)
    r_c, r_n = official_template_radii()
    result = score_row(item, r_c, r_n)
    for route in ROUTES:
        scored = result['routes'][route]
        assert scored['mean_lower_bound_angstrom'] is None
        assert scored['impossible_edge_fraction'] is None
        assert scored['counts']['n_eligible_edge_members'] == 0
    overall = aggregate_rows([result])
    assert overall['forecast_CA']['equal_protein']['n_valid_proteins'] == 0
    assert overall['forecast_CA']['all_members']['counts']['n_unknown_next_edge_members_excluded'] == 12
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, before[key])


def test_member_window_protein_weights_are_distinct_and_explicit():
    r_c, r_n = official_template_radii()
    a, b, c = row(2), row(4), row(4)
    a['metadata']['protein'] = 'A'
    for value in (b, c):
        value['metadata']['protein'] = 'B'
        value['oracle_CA'].zero_()
        value['forecast_CA'].zero_()
        value['observed_backbone'].zero_()
        value['target_backbone'].zero_()
    records = [score_row(x, r_c, r_n) for x in (a, b, c)]
    total = aggregate_rows(records)
    for route in ROUTES:
        assert total[route]['all_members']['impossible_edge_fraction'] == pytest.approx(1/7)
        assert total[route]['equal_window']['impossible_edge_fraction'] == pytest.approx(1/3)
        assert total[route]['equal_protein']['impossible_edge_fraction'] == pytest.approx(1/2)
    assert total['forecast_CA']['all_members']['n_members'] == 36
    assert total['oracle_CA']['all_members']['n_members'] == 3


@pytest.mark.parametrize('edges', [torch.tensor([[0, 0]]), torch.tensor([[0, 1], [0, 1]]),
                                   torch.tensor([[0, 2]]), torch.tensor([[0., 1.]])])
def test_invalid_explicit_edges_rejected(edges):
    item = row(2)
    item['peptide_edges'] = edges
    with pytest.raises(ValueError):
        prepare_edges(item)


def test_numerical_margin_strict_boundary_and_exclusive_output(tmp_path):
    left = torch.zeros(3, dtype=torch.float64)
    # A raw positive residual below the fixed absolute allowance is not called impossible.
    right = left.new_tensor([1.5+1.4+1.497+1e-5, 0., 0.])
    bound = peptide_lower_bound(left, right, 1.5, 1.4, 1.497)
    assert 0 < bound['raw'] < bound['margin']
    assert not bound['impossible'] and bound['conservative'] == 0
    target = tmp_path/'evidence.json'
    write_json_exclusive(target, dict(meaning='original'))
    with pytest.raises(FileExistsError):
        write_json_exclusive(target, dict(meaning='changed'))
    assert json.loads(target.read_text()) == dict(meaning='original')


def test_source_snapshot_ignores_virtual_module_filenames(monkeypatch):
    monkeypatch.setitem(sys.modules, '_capacity_virtual_module', SimpleNamespace(__file__=str(REPO/'_not_a_real_source.py')))
    sources = source_hashes()
    assert '_not_a_real_source.py' not in sources
    assert 'scripts/diagnose_fixed_ca_capacity.py' in sources
    assert 'tests/test_fixed_ca_capacity.py' in sources
