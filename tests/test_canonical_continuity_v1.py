"""Synthetic-only geometry diagnostics; no model, trajectory, or checkpoint IO."""
from itertools import product

import numpy as np
import pytest
import torch

from protein_jepa_frame.models.encoder import _within_frame_canonical_positions
from protein_jepa_frame.models.stage_c.canonical_continuity_v1 import (
    canonical_frame_audit, signed_permutations, transition_audit,
)


def backbone(ca):
    return ca[:, None, :].repeat(1, 4, 1).float()


def cloud(seed=34, n=31):
    generator = torch.Generator().manual_seed(seed)
    ca = torch.randn(n, 3, generator=generator) * torch.tensor([3., 1., .3])
    return backbone(ca)


def run(coords0, coords1, mask=None, H0=None, H1=None):
    if mask is None:
        mask = torch.ones(len(coords0), dtype=torch.bool)
    if H0 is None:
        H0 = torch.randn(len(coords0), 9, generator=torch.Generator().manual_seed(31))
    if H1 is None:
        H1 = H0 + .2
    frames = [canonical_frame_audit(c, mask) for c in (coords0, coords1)]
    return transition_audit(*frames, H0, H1, mask)


@pytest.mark.parametrize('seed', [0, 1, 2, 23])
@pytest.mark.parametrize('masked', [False, True])
def test_exact_source_float32_B1_parity(seed, masked):
    coords = cloud(seed)
    mask = torch.ones(len(coords), dtype=torch.bool)
    if masked:
        mask[::4] = False
        coords[~mask] = float('nan')
    original = coords.clone()
    frame = canonical_frame_audit(coords, mask)
    safe = torch.where(mask[:, None, None], coords, torch.zeros_like(coords))
    source = _within_frame_canonical_positions(safe[None], mask[None])[0]
    assert torch.equal(frame['canonical'], source)
    assert torch.allclose(coords, original, equal_nan=True)
    assert torch.equal(frame['basis'], frame['raw_basis'] * frame['sign'][None])
    assert frame['normalized_eigengaps'].shape == (2,)


def test_known_proper_rotation_row_convention_and_translation():
    coords = cloud()
    theta = .72
    Q = torch.tensor([[np.cos(theta), -np.sin(theta), 0.],
                      [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]], dtype=torch.float64)
    t = torch.tensor([4., -2., 1.], dtype=torch.float64)
    moved = (coords.double() @ Q + t).float()
    out = run(coords, moved)
    a, s = out['arrays'], out['summary']
    torch.testing.assert_close(a['R'], Q.T, atol=2e-7, rtol=0)
    torch.testing.assert_close(a['translation'], -t @ Q.T, atol=4e-7, rtol=0)
    torch.testing.assert_close(a['full_overlap'], torch.eye(3, dtype=torch.float64), atol=2e-6, rtol=0)
    assert s['kabsch_rmsd'] < 5e-7
    assert s['all_pair_distance_rms_change'] < 5e-7
    assert s['kabsch_determinant'] == pytest.approx(1., abs=2e-15)
    assert s['analysis_jump_rms'] < 3e-6
    assert s['decomposition_max_abs_error'] < 3e-15


def test_reflection_retains_improper_overlap_without_angle():
    coords = cloud(12)
    reflected = coords * torch.tensor([-1., 1., 1.])
    out = run(coords, reflected)
    s = out['summary']
    assert s['all_pair_distance_rms_change'] == 0
    assert s['kabsch_determinant'] == pytest.approx(1., abs=2e-15)
    assert s['kabsch_rmsd'] > .1
    assert s['overlap_determinant'] < -.99999
    assert all('angle' not in key for key in s)
    assert out['arrays']['candidate_costs'].shape == (48,)


def test_all48_order_orthogonality_unique_and_first_exact_minimum():
    P = signed_permutations()
    assert P.shape == (48, 3, 3)
    assert len({tuple(x.flatten().tolist()) for x in P}) == 48
    assert torch.equal(P[0], -torch.eye(3, dtype=torch.float64))
    assert torch.equal(P[7], torch.eye(3, dtype=torch.float64))
    assert torch.equal(P @ P.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(48, -1, -1))
    assert int((torch.linalg.det(P) > 0).sum()) == 24
    out = run(cloud(), cloud())
    a, s = out['arrays'], out['summary']
    assert s['selected_candidate_index'] == int(a['candidate_costs'].argmin())
    assert s['selected_candidate_index'] == 7
    assert s['candidate_near_tie_count'] >= s['candidate_exact_tie_count']


def test_skew_crossing_large_feature_change_small_physical_motion():
    def crossing(eps):
        return backbone(torch.tensor(list(product([-3., -2., -1., 1., 2., 3.+eps],
                                                  [-.7, -.2, .9], [-.3, -.05, .35]))))
    c0, c1 = crossing(-1e-4), crossing(1e-4)
    out = run(c0, c1)
    s, a = out['summary'], out['arrays']
    assert s['kabsch_rmsd'] < 1e-4
    assert s['all_pair_distance_rms_change'] < 1e-4
    assert s['analysis_jump_rms'] > 4.
    assert s['matched_jump_rms'] < 1e-4
    assert s['original_basis_cost'] > 3.9
    assert s['selected_candidate_cost'] < 1e-10
    assert a['transported_prior_basis_third_moment0'][0] * a['transported_prior_basis_third_moment1'][0] < 0
    assert s['squared_decomposition_max_abs_error'] < 1e-13


def test_source_centering_roundoff_retained_not_recentered_for_decomposition():
    c0 = cloud() + 10000.
    c1 = c0.clone()
    c1[:, 1, 0] += torch.linspace(0, .05, len(c1))
    out = run(c0, c1)
    a, s = out['arrays'], out['summary']
    frame = canonical_frame_audit(c0, torch.ones(len(c0), dtype=torch.bool))
    assert torch.equal(a['analysis_Z0'], frame['centered'].double() @ frame['basis'].double())
    assert s['source_centering0_norm'] > 1e-6
    assert s['decomposition_max_abs_error'] < 2e-15
    assert s['source_reconstruction0_max_abs_error'] > 0
    assert not torch.equal(a['analysis_jump'], a['source_jump'])
    # Physical residual uses translation-optimal double native centering.
    assert a['physical_residual'].mean(0).abs().max() < 1e-11


@pytest.mark.parametrize('kind', ['collapsed', 'collinear', 'isotropic', 'near_isotropic'])
def test_degenerate_geometry_kept_without_event_selection(kind):
    if kind == 'collapsed':
        ca = torch.ones(7, 3)
    elif kind == 'collinear':
        ca = torch.stack([torch.arange(7.), torch.zeros(7), torch.zeros(7)], -1)
    else:
        ca = torch.tensor(list(product([-1., 1.], repeat=3)))
        if kind == 'near_isotropic':
            ca[:, 0] *= 1. + 1e-6
    coords = backbone(ca)
    out = run(coords, coords)
    s = out['summary']
    assert s['residue_count'] == len(ca)
    assert s['pair_count'] == len(ca) * (len(ca)-1) // 2
    assert s['analysis_jump_rms'] == 0
    assert out['arrays']['candidate_costs'].shape == (48,)
    assert s['kabsch_rank_below_two'] == (kind in ('collapsed', 'collinear'))
    assert s['kabsch_rmsd'] < 1e-14
    if kind == 'collapsed':
        f = canonical_frame_audit(coords, torch.ones(len(ca), dtype=torch.bool))
        assert not f['skew_defined'].any()
        assert torch.isnan(f['abs_standardized_skew']).all()
    if kind == 'isotropic':
        f = canonical_frame_audit(coords, torch.ones(len(ca), dtype=torch.bool))
        assert torch.equal(f['normalized_eigengaps'], torch.zeros(2, dtype=torch.float64))


def test_masked_rows_do_not_affect_geometry_or_H_and_allvalid_pairs_used():
    c0, c1 = cloud(), cloud(44)
    H0 = torch.randn(len(c0), 7, generator=torch.Generator().manual_seed(7))
    H1 = H0 * 1.3
    mask = torch.arange(len(c0)) % 3 != 0
    full = run(c0, c1, mask, H0, H1)
    c0[~mask] = float('nan'); c1[~mask] = float('inf')
    H0[~mask] = float('nan'); H1[~mask] = float('nan')
    changed = run(c0, c1, mask, H0, H1)
    assert full['summary'] == changed['summary']
    assert torch.equal(full['arrays']['pair_indices'], changed['arrays']['pair_indices'])
    assert mask[full['arrays']['pair_indices']].all()
    assert full['summary']['pair_count'] == int(mask.sum()) * (int(mask.sum())-1)//2


def test_H_projectors_match_independent_numpy_and_finite_chord():
    coords = cloud(n=13)
    gen = torch.Generator().manual_seed(58)
    H0 = torch.randn(13, 5, generator=gen) + 3
    H1 = torch.randn(13, 5, generator=gen) - 2
    a = run(coords, coords, H0=H0, H1=H1)['arrays']
    J = np.eye(5) - np.ones((5, 5))/5
    h0, h1 = H0.numpy().astype(float), H1.numpy().astype(float)
    tangent, radial = [], []
    for x, y in zip(h0, h1):
        z = J @ x
        u = z / np.linalg.norm(z)
        T = J - np.outer(u, u)
        tangent.append(np.sum((T @ (y-x))**2))
        radial.append(u @ (y-x))
    np.testing.assert_allclose(a['H_tangent_increment_squared'].numpy(), tangent, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(a['H_radial_increment'].numpy(), radial, atol=1e-13, rtol=1e-13)
    assert a['H_finite_chord_closure'].abs().max() < 1e-13
    torch.testing.assert_close(a['H_raw_increment_squared'], a['H_centered_increment_squared'] + a['H_channel_increment_squared'], atol=1e-13, rtol=1e-13)


def test_undefined_tangent_preserves_raw_centered_support_and_null_summary():
    coords = cloud(n=6)
    H0 = torch.ones(6, 4)
    H1 = torch.arange(24.).reshape(6, 4)
    out = run(coords, coords, H0=H0, H1=H1)
    a, s = out['arrays'], out['summary']
    assert not a['tangent_valid'].any()
    assert torch.isnan(a['H_tangent_increment_squared']).all()
    assert s['H_tangent_increment_mean_squared'] is None
    assert s['H_tangent_undefined_count'] == 6
    assert s['H_raw_increment_mean_squared'] > 0
    assert s['H_centered_increment_mean_squared'] > 0
    assert s['H_finite_chord_closure_max_abs_error'] is None


def test_single_residue_keeps_row_but_has_null_pair_metric():
    out = run(backbone(torch.ones(1, 3)), backbone(torch.ones(1, 3)))
    assert out['summary']['pair_count'] == 0
    assert out['summary']['all_pair_distance_rms_change'] is None
    assert out['arrays']['pair_indices'].shape == (0, 2)


@pytest.mark.parametrize('failure', ['float64', 'empty', 'badshape', 'nonfinite', 'badmask'])
def test_source_validation_fails_closed(failure):
    coords = cloud()
    mask = torch.ones(len(coords), dtype=torch.bool)
    if failure == 'float64': coords = coords.double()
    if failure == 'empty': mask[:] = False
    if failure == 'badshape': coords = coords[:, :3]
    if failure == 'nonfinite': coords[0, 1, 0] = float('nan')
    if failure == 'badmask': mask = mask.float()
    with pytest.raises(ValueError):
        canonical_frame_audit(coords, mask)


def test_transition_rejects_mask_drift_and_nonfinite_H():
    c = cloud()
    mask = torch.ones(len(c), dtype=torch.bool)
    f0, f1 = canonical_frame_audit(c, mask), canonical_frame_audit(c, mask)
    H = torch.ones(len(c), 3)
    f1['mask'][0] = False
    with pytest.raises(ValueError, match='mask differs'):
        transition_audit(f0, f1, H, H, mask)
    f1['mask'][0] = True
    H[0, 1] = float('nan')
    with pytest.raises(ValueError, match='nonfinite H'):
        transition_audit(f0, f1, H, H, mask)


def test_full_rank_reflected_isotropic_fit_has_declared_near_tie_diagnostic():
    coords = backbone(torch.tensor(list(product([-1., 1.], repeat=3))))
    out = run(coords, coords * torch.tensor([-1., 1., 1.]))
    s = out['summary']
    assert s['kabsch_rank'] == 3
    assert not s['kabsch_rank_below_two']
    assert s['kabsch_reflection_correction']
    assert s['kabsch_reflected_smallest_singular_near_tie']
    assert s['kabsch_determinant'] == pytest.approx(1., abs=2e-15)
    assert s['residue_count'] == 8
    assert s['pair_count'] == 28


def test_exact_candidate_tie_selects_first_without_tolerance_group_selection():
    coords = cloud()
    mask = torch.ones(len(coords), dtype=torch.bool)
    f0, f1 = canonical_frame_audit(coords, mask), canonical_frame_audit(coords, mask)
    # Synthetic source-frame witnesses isolate matching from eigensolver gauge:
    # two axes lie exactly halfway between signed-permutation matches.
    q = torch.tensor(2. ** -.5, dtype=torch.float32)
    f0['basis'] = torch.eye(3, dtype=torch.float32)
    f1['basis'] = torch.tensor([[q, -q, 0.], [q, q, 0.], [0., 0., 1.]])
    for f in (f0, f1):
        f['canonical'] = f['centered'] @ f['basis']
    # Collapsed native clouds yield exact identity Kabsch, while distinct source
    # frames are intentionally synthetic inputs for the tie-selection routine.
    for f in (f0, f1):
        f['ca'] = torch.zeros_like(f['ca'])
    H = torch.ones(len(coords), 4)
    out = transition_audit(f0, f1, H, H, mask)
    a, s = out['arrays'], out['summary']
    minima = torch.nonzero(a['candidate_costs'] == a['candidate_costs'].min()).flatten()
    assert len(minima) == 2
    assert s['candidate_exact_tie_count'] == 2
    assert s['candidate_cost_margin'] == 0
    assert s['selected_candidate_index'] == int(minima[0])
    assert s['candidate_near_tie_atol'] == 1e-12


def test_constant_radius_finite_chord_has_negative_radial_component():
    coords = cloud(n=2)
    H0 = torch.tensor([[1., -1., 0.], [1., -1., 0.]])
    H1 = -H0
    out = run(coords, coords, H0=H0, H1=H1)
    a, s = out['arrays'], out['summary']
    assert torch.equal(a['H_centered_norm_squared_change'], torch.zeros(2, dtype=torch.float64))
    assert (a['H_radial_increment'] < 0).all()
    assert s['H_tangent_increment_mean_squared'] < 1e-28
    assert s['H_finite_chord_closure_max_abs_error'] < 1e-14


def test_two_variances_cross_with_nonzero_skew_and_axis_permutation_jump():
    # Independent Cartesian factors have nonzero third moments. Normalize x/y
    # to the same variance, then perturb their scales in opposite directions;
    # each endpoint has distinct eigenvalues, but the leading axis swaps.
    x = torch.tensor([-2., -1., 0., 3.], dtype=torch.float64)
    y = torch.tensor([-3., -.5, 1., 2.5], dtype=torch.float64)
    x = x / x.square().mean().sqrt()
    y = y / y.square().mean().sqrt()
    def crossing(eps):
        return backbone(torch.tensor(list(product((x*(1+eps)).tolist(),
                                                   (y*(1-eps)).tolist(),
                                                   [-.4, -.1, .5])), dtype=torch.float64))
    c0, c1 = crossing(-1e-4), crossing(1e-4)
    mask = torch.ones(len(c0), dtype=torch.bool)
    f0, f1 = canonical_frame_audit(c0, mask), canonical_frame_audit(c1, mask)
    assert (f0['abs_standardized_skew'] > .1).all()
    assert (f1['abs_standardized_skew'] > .1).all()
    assert (f0['normalized_eigengaps'] > 1e-5).all()
    assert (f1['normalized_eigengaps'] > 1e-5).all()
    assert f0['basis'].abs().argmax(dim=0).tolist() == [1, 0, 2]
    assert f1['basis'].abs().argmax(dim=0).tolist() == [0, 1, 2]
    out = run(c0, c1)
    a, s = out['arrays'], out['summary']
    assert s['kabsch_rmsd'] < 3e-4
    assert s['all_pair_distance_rms_change'] < 4e-4
    assert s['analysis_jump_rms'] > 1.9
    assert s['matched_jump_rms'] < 3e-4
    assert a['selected_permutation'].abs().argmax(dim=0).tolist() == [1, 0, 2]
    assert s['selected_candidate_cost'] < 1e-8
    assert s['squared_decomposition_max_abs_error'] < 1e-13
