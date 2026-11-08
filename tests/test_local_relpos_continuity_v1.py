"""Synthetic raw-feature diagnostics only; no model or saved-data IO."""
import numpy as np
import pytest
import torch

import protein_jepa_frame.models.stage_c.local_relpos_continuity_v1 as v
from protein_jepa_frame.models.encoder import _local_backbone_frames


def coords(seed=5, length=17):
    g = torch.Generator().manual_seed(seed)
    ca = torch.randn(length, 3, generator=g)*3
    x = ca[:, None].expand(-1, 4, -1).clone()
    x[:, 0] += torch.tensor([-.4, 1.2, .1])
    x[:, 2] += torch.tensor([1.4, .1, .2])
    x[:, 3] += torch.tensor([1.9, .3, .1])
    return x


def source_center(x, mask):
    safe = torch.where(mask[:, None, None], x, torch.zeros_like(x))[None]
    ca = safe[:, :, 1]
    m = mask[None].float().unsqueeze(-1)
    centroid = (ca*m).sum(1, keepdim=True)/m.sum(1, keepdim=True).clamp_min(1)
    return ((ca-centroid)*m)[0]


def audit(x, mask=None, pca_basis=None):
    if mask is None: mask = torch.ones(len(x), dtype=torch.bool)
    centered = source_center(x, mask)
    if pca_basis is None: pca_basis = torch.eye(3)
    # Caller-supplied baseline; no synthetic eigensystem is needed to test the
    # local-frame algebra or the contract forbidding a new PCA/Kabsch fit.
    return v.frame_audit(x, mask, centered, centered@pca_basis, pca_basis)


def np_frames(x):
    c = x[:, 2]-x[:, 1]; n = x[:, 0]-x[:, 1]
    normalize = lambda z: z/np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    axis = normalize(c); direction = normalize(n)
    z = normalize(np.cross(axis, direction)); y = normalize(np.cross(z, axis))
    return np.stack([axis, y, z], axis=-2)


def test_source_frames_exact_B1_and_independent_numpy_frame_math(monkeypatch):
    x = coords(); mask = torch.ones(len(x), dtype=torch.bool)
    original = _local_backbone_frames
    calls = []
    def record(z):
        calls.append(z.shape)
        return original(z)
    monkeypatch.setattr(v, '_local_backbone_frames', record)
    out = audit(x, mask)
    assert calls == [(1, len(x), 4, 3)]
    assert torch.equal(out['arrays']['local_frames'].view(torch.int32), original(x[None])[0].view(torch.int32))
    expected = np_frames(x.numpy().astype(float))
    np.testing.assert_allclose(out['arrays']['local_frames'].numpy(), expected, atol=2e-7, rtol=2e-7)
    expected_feature = np.einsum('lkc,lc->lk', out['arrays']['local_frames'].double().numpy(), out['arrays']['centered'].double().numpy())
    np.testing.assert_allclose(out['arrays']['analysis_local'].numpy(), expected_feature, atol=1e-14, rtol=1e-14)


def test_no_pca_or_kabsch_refit(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('forbidden decomposition')
    monkeypatch.setattr(torch.linalg, 'eigh', forbidden)
    monkeypatch.setattr(torch.linalg, 'svd', forbidden)
    x = coords(); mask = torch.ones(len(x), dtype=torch.bool)
    a = audit(x, mask); b = audit(x+.1, mask)
    v.pair_audit(a, b, torch.eye(3, dtype=torch.float64), mask)


def test_proper_SE3_invariance_row_transport_and_roundoff():
    x = coords()
    angle = .67
    Q = torch.tensor([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float64)
    moved = (x.double()@Q + torch.tensor([5., -3., 1.])).float()
    a, b = audit(x), audit(moved)
    torch.testing.assert_close(a['arrays']['local_canonical'], b['arrays']['local_canonical'], atol=4e-6, rtol=1e-6)
    out = v.pair_audit(a, b, Q.T, a['arrays']['mask'])
    assert out['summary']['local']['source_jump_rms'] < 4e-6
    assert out['summary']['local_frame_change_frobenius']['max'] < 2e-6
    assert out['summary']['local_decomposition_max_abs_error'] < 2e-15
    assert out['summary']['local_transport_roundoff_max_abs_error'] < 3e-15


def test_reflection_flips_normal_component_preserving_chirality_sensitivity():
    x = coords()
    a, b = audit(x), audit(x*torch.tensor([-1., 1., 1.]))
    f0, f1 = a['arrays']['local_canonical'], b['arrays']['local_canonical']
    torch.testing.assert_close(f1, f0*torch.tensor([1., 1., -1.]), atol=1e-6, rtol=1e-6)
    assert a['summary']['frame_determinant']['min'] > .999999
    assert b['summary']['frame_determinant']['min'] > .999999


def test_fixed_mask_centroid_and_invalid_padding_robustness():
    x = coords(); mask = torch.arange(len(x))%3 != 0
    a = audit(x, mask)
    corrupted = x.clone(); corrupted[~mask] = float('nan')
    b = audit(corrupted, mask)
    assert a['summary'] == b['summary']
    assert torch.equal(a['arrays']['local_canonical'], b['arrays']['local_canonical'])
    assert torch.equal(b['arrays']['source_backbone'][~mask], torch.zeros_like(b['arrays']['source_backbone'][~mask]))
    expected_centroid = x[mask, 1].double().mean(0)
    torch.testing.assert_close(b['arrays']['centroid'].double(), expected_centroid, atol=5e-7, rtol=1e-6)
    assert not b['arrays']['bond_angle_defined'][~mask].any()
    assert b['summary']['residue_count'] == int(mask.sum())
    changed = audit(x, torch.ones(len(x), dtype=torch.bool))
    assert not torch.equal(a['arrays']['centroid'], changed['arrays']['centroid'])


def test_supplied_centered_requires_bits_not_approximate_equality():
    x = coords(); mask = torch.ones(len(x), dtype=torch.bool)
    center = source_center(x, mask); canonical = center.clone()
    center[0, 0] = torch.nextafter(center[0, 0], torch.tensor(float('inf')))
    with pytest.raises(ValueError, match='centering'):
        v.frame_audit(x, mask, center, canonical, torch.eye(3))


@pytest.mark.parametrize('kind', ['zero_c', 'zero_n', 'collapsed', 'parallel', 'antiparallel', 'near_collinear'])
def test_degenerate_rows_retained_no_fallback_or_hidden_mask(kind):
    x = coords(length=5)
    if kind in ('zero_c', 'collapsed'): x[:, 2] = x[:, 1]
    if kind in ('zero_n', 'collapsed'): x[:, 0] = x[:, 1]
    if kind in ('parallel', 'antiparallel', 'near_collinear'):
        # Use a zero CA origin so the tiny near-collinear component remains
        # representable in float32 instead of vanishing during subtraction.
        x[:] = 0
        x[:, 2, 0] = 1
        x[:, 0, 0] = -1 if kind == 'antiparallel' else 1
        if kind == 'near_collinear': x[:, 0, 1] = 1e-10
    out = audit(x); a, s = out['arrays'], out['summary']
    assert s['residue_count'] == 5
    assert s['normalized_cross_at_or_below_eps_count'] == 5
    assert s['orthogonality_max_abs_error']['max'] > .99
    if kind in ('zero_c', 'zero_n', 'collapsed'):
        assert s['bond_angle_undefined_count'] == 5
        assert torch.isnan(a['bond_angle_sine']).all()
    else:
        assert s['bond_angle_defined_count'] == 5
    if kind == 'near_collinear':
        assert s['exactly_degenerate_count'] == 0
        assert (a['frame_determinant'] > 0).all()
        torch.testing.assert_close(a['frame_determinant'], torch.full((5,), .01, dtype=torch.float64), atol=1e-8, rtol=0)
    else:
        assert s['exactly_degenerate_count'] == 5
    pair = v.pair_audit(out, out, torch.eye(3, dtype=torch.float64), a['mask'])
    assert pair['summary']['local']['source_jump_rms'] == 0
    assert pair['summary']['local_decomposition_max_abs_error'] == 0


def test_local_rotation_at_fixed_CA_and_lever_arm_amplification():
    x = coords(length=5)
    y = x.clone()
    theta = .1
    Q = torch.tensor([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]], dtype=torch.float32)
    y = (y-y[:, 1:2])@Q+y[:, 1:2]
    a, b = audit(x), audit(y)
    out = v.pair_audit(a, b, torch.eye(3, dtype=torch.float64), a['arrays']['mask'])
    assert out['summary']['local_shape_mean_squared'] == 0
    assert out['summary']['local_basis_mean_squared'] > .01
    assert out['summary']['local_frame_change_frobenius']['mean'] > .1
    assert out['summary']['local']['source_jump_rms'] > .1
    assert out['summary']['pca']['source_jump_rms'] == 0


def test_decomposition_matches_independent_numpy_signed_cross_and_source_residual():
    x = coords(); y = coords(seed=8)
    y[:, 0] += torch.linspace(-.3, .3, len(y))[:, None]
    mask = torch.ones(len(x), dtype=torch.bool)
    a, b = audit(x, mask), audit(y, mask)
    R = torch.eye(3, dtype=torch.float64)
    out = v.pair_audit(a, b, R, mask)
    X, Y = a['arrays']['centered'].double().numpy(), b['arrays']['centered'].double().numpy()
    F0, F1 = a['arrays']['local_frames'].double().numpy(), b['arrays']['local_frames'].double().numpy()
    shape = np.einsum('lkc,lc->lk', F0, Y-X)
    basis = np.einsum('lkc,lc->lk', F1-F0, Y)
    cross = 2*np.sum(shape*basis, axis=-1)
    np.testing.assert_allclose(out['arrays']['local_shape_term'].numpy(), shape, atol=1e-14)
    np.testing.assert_allclose(out['arrays']['local_basis_term'].numpy(), basis, atol=1e-14)
    np.testing.assert_allclose(out['arrays']['local_cross'].numpy(), cross, atol=1e-13)
    assert out['summary']['local_squared_decomposition_max_abs_error'] < 1e-12
    assert a['summary']['local']['source_reconstruction_max_abs_error'] > 0
    assert out['summary']['local']['analysis_minus_source_jump_max_abs_error'] > 0


def test_zero_branch_exact_and_identity_pair_exact():
    a = audit(coords()); out = v.pair_audit(a, a, torch.eye(3, dtype=torch.float64), a['arrays']['mask'])
    for value in (a['arrays']['zero_canonical'], a['arrays']['analysis_zero'], out['arrays']['zero__source_jump'], out['arrays']['zero__analysis_jump']):
        assert torch.count_nonzero(value) == 0
    for name in ('pca', 'zero', 'local'):
        assert out['summary'][name]['source_jump_rms'] == 0
        assert out['summary'][name]['analysis_jump_rms'] == 0


def test_masks_and_invalid_or_improper_R_rejected():
    a = audit(coords()); mask = a['arrays']['mask'].clone()
    for R in (torch.diag(torch.tensor([-1., 1., 1.], dtype=torch.float64)), torch.eye(3, dtype=torch.float64)*2):
        with pytest.raises(ValueError, match='proper'):
            v.pair_audit(a, a, R, mask)
    mask[0] = False
    with pytest.raises(ValueError, match='masks'):
        v.pair_audit(a, a, torch.eye(3, dtype=torch.float64), mask)


@pytest.mark.parametrize('kind', ['nonfinite', 'double', 'empty', 'bad_pca_padding'])
def test_frame_validation(kind):
    x = coords(); mask = torch.ones(len(x), dtype=torch.bool)
    if kind == 'nonfinite': x[0, 0, 0] = float('nan')
    if kind == 'double': x = x.double()
    if kind == 'empty': mask[:] = False
    if kind == 'bad_pca_padding': mask[-1] = False
    center = source_center(x, mask).float(); canonical = center.clone()
    if kind == 'bad_pca_padding': canonical[-1] = 1
    with pytest.raises(ValueError): v.frame_audit(x, mask, center, canonical, torch.eye(3))


@pytest.mark.parametrize('tiny_atom,flag', [(2, 'c_norm_at_or_below_eps_count'), (0, 'n_norm_at_or_below_eps_count')])
def test_nonzero_bond_below_source_epsilon_is_reported_without_drop(tiny_atom, flag):
    x = torch.zeros(4, 4, 3)
    x[:, 2, 0] = 1.
    x[:, 0, 1] = 1.
    x[:, tiny_atom] *= 1e-10
    out = audit(x)
    assert out['summary'][flag] == 4
    assert out['summary']['exactly_degenerate_count'] == 0
    assert out['summary']['residue_count'] == 4
    torch.testing.assert_close(out['arrays']['bond_angle_sine'], torch.ones(4, dtype=torch.float64))


def test_finite_extreme_inputs_fail_if_source_arithmetic_overflows():
    x = coords(length=3)
    x[:, 1, 0] = -3e38
    x[:, 2, 0] = 3e38
    # Centering itself can overflow, so either the source-centered guard or the
    # derived-geometry guard must fail; no finite-looking fallback is accepted.
    with pytest.raises(ValueError): audit(x)
