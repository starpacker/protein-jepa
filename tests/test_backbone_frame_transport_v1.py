"""Synthetic CPU contracts for the separate, label-free frame transport."""
import math

import pytest
import torch

from protein_jepa_frame.models.backbone_frame_transport_v1 import transport_observed_frames


def rotation(axis=(1., 2., -1.), angle=.73, dtype=torch.float64):
    axis = torch.tensor(axis, dtype=dtype)
    axis = axis / axis.norm()
    x, y, z = axis.unbind()
    zero = axis.new_zeros(())
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return torch.eye(3, dtype=dtype) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def inputs(length=7, dtype=torch.float64):
    g = torch.Generator().manual_seed(933)
    obs = torch.randn(1, length, 3, dtype=dtype, generator=g)
    R = torch.stack([rotation(angle=.11 * i, dtype=dtype) for i in range(length)])[None]
    metadata = dict(residue_mask=torch.ones(length, dtype=torch.bool),
                    peptide_edges=torch.stack((torch.arange(length - 1), torch.arange(1, length)), -1),
                    geometry_row_ids=torch.arange(length) * 17 + 100)
    return obs, R, metadata


@pytest.mark.parametrize('dtype,atol', [(torch.float64, 2e-12), (torch.float32, 2e-6)])
def test_global_rigid_motion_recovers_rotation_and_frame(dtype, atol):
    obs, R, kw = inputs(dtype=dtype)
    Q = rotation(dtype=dtype)
    cur = obs @ Q.T + obs.new_tensor([4., -8., 1.3])
    before = [value.clone() for value in (obs, cur, R)]
    result = transport_observed_frames(obs, cur, R, **kw)
    assert result['rotation'].dtype == dtype
    assert torch.allclose(result['rotation'], Q.expand_as(result['rotation']), atol=atol, rtol=0)
    assert torch.allclose(result['frame_seed'], Q @ R, atol=atol, rtol=0)
    audit = result['audit']
    assert not audit['fallback'].any()
    assert audit['neighbor_count'].tolist() == [[3, 4, 5, 5, 5, 4, 3]]
    assert (audit['rank'] >= 2).all()
    assert audit['fit_rmsd'].max() < atol * 4
    assert torch.allclose(audit['det'], torch.ones_like(audit['det']), atol=atol, rtol=0)
    assert audit['orthogonality'].max() < atol * 4
    assert all(torch.equal(a, b) for a, b in zip((obs, cur, R), before))


def test_combined_gauge_equivariance_with_nonrigid_current_geometry():
    obs, R, kw = inputs()
    cur = obs @ rotation().T
    cur[:, 2] += torch.tensor([.1, -.3, .2])
    base = transport_observed_frames(obs, cur, R, **kw)
    G = rotation(axis=(2., -1., 4.), angle=1.3)
    shift = torch.tensor([8., 2., -7.])
    moved = transport_observed_frames(obs @ G.T + shift, cur @ G.T + shift, G @ R, **kw)
    assert torch.allclose(moved['rotation'], G @ base['rotation'] @ G.T, atol=2e-12, rtol=0)
    assert torch.allclose(moved['frame_seed'], G @ base['frame_seed'], atol=2e-12, rtol=0)
    assert torch.allclose(moved['audit']['fit_rmsd'], base['audit']['fit_rmsd'], atol=2e-12, rtol=0)
    assert torch.equal(moved['audit']['rank'], base['audit']['rank'])


def test_vectors_are_centered_at_residue_not_neighborhood_centroid():
    obs = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], dtype=torch.float64)
    cur = torch.tensor([[[0., 0., 0.], [1., 1., 0.], [0., 1., 0.]]], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64).expand(1, 3, 3, 3)
    out = transport_observed_frames(obs, cur, R, residue_mask=torch.ones(3, dtype=torch.bool),
        peptide_edges=torch.tensor([[0, 1], [0, 2]]), geometry_row_ids=torch.arange(3))
    expected = rotation(axis=(0., 0., 1.), angle=math.atan2(1, 2))
    assert torch.allclose(out['rotation'][0, 0], expected, atol=1e-14, rtol=0)
    residual = obs[0] @ expected.T - cur[0]
    assert out['audit']['fit_rmsd'][0, 0] == pytest.approx(residual.square().sum(-1).mean().sqrt().item())


def test_topology_gaps_and_invalid_vertices_do_not_bridge():
    obs, R, kw = inputs(length=7)
    # No edge 2--3; no inference from contiguous rows or geometry IDs.
    kw['peptide_edges'] = torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5], [5, 6]])
    cur = obs @ rotation().T + 2
    out = transport_observed_frames(obs, cur, R, **kw)
    assert out['audit']['neighbor_count'].tolist() == [[3, 3, 3, 3, 4, 4, 3]]
    kw['residue_mask'][4] = False
    obs[:, 4] = torch.nan
    cur[:, 4] = torch.nan
    R[:, 4] = torch.nan
    out = transport_observed_frames(obs, cur, R, **kw)
    assert out['audit']['neighbor_count'].tolist() == [[3, 3, 3, 1, 0, 2, 2]]
    assert out['audit']['fallback'].tolist() == [[False, False, False, True, True, True, True]]
    assert torch.equal(out['frame_seed'][0, 3], R[0, 3])
    assert torch.equal(out['frame_seed'][0, 4], torch.eye(3, dtype=R.dtype))
    assert all(torch.isfinite(value).all() for value in out['audit'].values())


def test_complete_two_hop_neighborhood_has_no_five_point_cap():
    obs, R, kw = inputs(length=8)
    kw['peptide_edges'] = torch.stack((torch.zeros(7, dtype=torch.long), torch.arange(1, 8)), -1)
    out = transport_observed_frames(obs, obs @ rotation().T, R, **kw)
    assert out['audit']['neighbor_count'].tolist() == [[8] * 8]
    assert not out['audit']['fallback'].any()


def test_rank_one_fallback_preserves_observed_frame_and_reports_residual():
    obs, R, kw = inputs(length=4)
    obs[:] = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]])
    cur = obs @ rotation().T
    out = transport_observed_frames(obs, cur, R, **kw)
    assert out['audit']['fallback'].all()
    assert (out['audit']['rank'] == 1).all()
    assert torch.equal(out['frame_seed'], R)
    assert torch.equal(out['rotation'], torch.eye(3, dtype=R.dtype).expand_as(R))
    assert (out['audit']['fit_rmsd'] > 0).all()


@pytest.mark.parametrize('epsilon,scale,rank', [(1e-4, 1., 1), (2e-3, 1., 2), (1., 1e-6, 0)])
def test_fixed_relative_and_absolute_rank_thresholds(epsilon, scale, rank):
    obs = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., epsilon, 0.]]], dtype=torch.float64) * scale
    R = torch.eye(3, dtype=torch.float64).expand(1, 3, 3, 3)
    out = transport_observed_frames(obs, obs @ rotation().T, R,
        residue_mask=torch.ones(3, dtype=torch.bool), peptide_edges=torch.tensor([[0, 1], [0, 2]]),
        geometry_row_ids=torch.arange(3))
    assert out['audit']['rank'][0, 0] == rank
    assert out['audit']['fallback'][0, 0].item() == (rank < 2)


def test_exact_identity_is_bitwise_and_no_grad_with_twelve_samples():
    obs, R, kw = inputs(dtype=torch.float32)
    obs.requires_grad_()
    R.requires_grad_()
    cur = obs.detach().expand(12, -1, -1).clone()
    cur[1:] = cur[1:] @ rotation(dtype=torch.float32).T + 1
    cur.requires_grad_()
    state = torch.random.get_rng_state().clone()
    out = transport_observed_frames(obs, cur, R, **kw)
    assert out['frame_seed'].shape == (12, 7, 3, 3)
    assert torch.equal(out['rotation'][0], torch.eye(3).expand(7, 3, 3))
    assert torch.equal(out['frame_seed'][0], R[0])
    assert out['audit']['exact_identity'][0].all()
    assert not out['audit']['exact_identity'][1:].any()
    for i in (0, 3, 11):
        single = transport_observed_frames(obs, cur[i:i+1], R, **kw)
        assert torch.equal(single['frame_seed'][0], out['frame_seed'][i])
    assert all(not value.requires_grad for value in (out['frame_seed'], out['rotation'], *out['audit'].values()))
    assert torch.equal(state, torch.random.get_rng_state())


def test_reflection_input_is_fitted_with_proper_rotation():
    obs, R, kw = inputs()
    kw['peptide_edges'] = torch.combinations(torch.arange(7))
    cur = obs.clone()
    cur[..., 0] *= -1
    out = transport_observed_frames(obs, cur, R, **kw)
    assert (out['audit']['rank'] == 3).all()
    assert torch.allclose(out['audit']['det'], torch.ones(1, 7, dtype=torch.float64), atol=2e-14, rtol=0)
    assert (out['audit']['fit_rmsd'] > .01).all()


def test_row_permutation_edge_direction_and_identity_renaming_do_not_change_geometry():
    obs, R, kw = inputs()
    cur = obs @ rotation().T
    base = transport_observed_frames(obs, cur, R, **kw)
    p = torch.tensor([5, 1, 0, 6, 2, 3, 4])
    inverse = torch.argsort(p)
    edges = inverse[kw['peptide_edges'].flip(-1)]
    # Both directed versions may refer to the same undirected PSF link.
    edges = torch.cat((edges, edges.flip(-1)))
    moved = transport_observed_frames(obs[:, p], cur[:, p], R[:, p],
        residue_mask=kw['residue_mask'][p], peptide_edges=edges, geometry_row_ids=torch.arange(7) + 803)
    assert torch.allclose(moved['frame_seed'], base['frame_seed'][:, p], atol=2e-14, rtol=0)
    assert torch.equal(moved['audit']['neighbor_count'], base['audit']['neighbor_count'][:, p])


def test_all_masked_nan_and_empty_edges_are_finite_safe():
    obs, R, kw = inputs(length=1)
    obs[:] = torch.nan
    R[:] = torch.nan
    kw['residue_mask'].zero_()
    out = transport_observed_frames(obs, obs.expand(12, -1, -1), R, **kw)
    assert torch.equal(out['frame_seed'], torch.eye(3, dtype=R.dtype).expand(12, 1, 3, 3))
    assert out['audit']['fallback'].all()
    assert not out['audit']['valid'].any()
    for key in ('neighbor_count', 'rank', 'fit_rmsd', 'orthogonality'):
        assert torch.count_nonzero(out['audit'][key]) == 0


def test_batch_specific_observed_masks_remove_bridges_independently():
    obs, R, kw = inputs(length=5)
    cur = (obs @ rotation().T).expand(2, -1, -1).clone()
    kw['residue_mask'] = torch.ones(2, 5, dtype=torch.bool)
    kw['residue_mask'][1, 2] = False
    cur[1, 2] = torch.nan
    out = transport_observed_frames(obs, cur, R, **kw)
    assert out['audit']['neighbor_count'].tolist() == [[3, 4, 5, 4, 3], [2, 2, 0, 2, 2]]
    assert not out['audit']['fallback'][0].any()
    assert out['audit']['fallback'][1].all()
    assert torch.equal(out['frame_seed'][1, [0, 1, 3, 4]], R[0, [0, 1, 3, 4]])


def test_no_edges_keeps_valid_observed_frames_without_inferred_adjacency():
    obs, R, kw = inputs()
    kw['peptide_edges'] = torch.empty(0, 2, dtype=torch.long)
    out = transport_observed_frames(obs, obs @ rotation().T + 3, R, **kw)
    assert torch.equal(out['frame_seed'], R)
    assert (out['audit']['neighbor_count'] == 1).all()
    assert (out['audit']['rank'] == 0).all()
    assert out['audit']['fallback'].all()


@pytest.mark.parametrize('which', ['observed_ca', 'current_ca', 'observed_R'])
def test_nonfinite_valid_inputs_fail_closed(which):
    obs, R, kw = inputs()
    values = dict(observed_ca=obs, current_ca=obs.clone(), observed_R=R)
    values[which].flatten()[0] = torch.inf
    with pytest.raises(ValueError, match='nonfinite valid'):
        transport_observed_frames(**values, **kw)


@pytest.mark.parametrize('edges', [torch.tensor([[0, 0]]), torch.tensor([[0, 1], [0, 1]]),
    torch.tensor([[0, 7]]), torch.tensor([[-1, 0]]), torch.tensor([[0., 1.]]),
    torch.tensor([[False, True]]), torch.tensor([0, 1]), torch.empty(0, dtype=torch.long)])
def test_invalid_edges_fail_closed(edges):
    obs, R, kw = inputs()
    kw['peptide_edges'] = edges
    with pytest.raises(ValueError):
        transport_observed_frames(obs, obs.clone(), R, **kw)


@pytest.mark.parametrize('ids', [torch.arange(6), torch.zeros(7, dtype=torch.long),
    torch.arange(7).float(), torch.ones(7, dtype=torch.bool), torch.arange(7)[None]])
def test_invalid_row_ids_fail_closed(ids):
    obs, R, kw = inputs()
    kw['geometry_row_ids'] = ids
    with pytest.raises(ValueError):
        transport_observed_frames(obs, obs.clone(), R, **kw)


def test_shape_dtype_and_mask_contracts():
    obs, R, kw = inputs()
    for bad_obs, bad_current, bad_R, bad_mask in (
        (obs.float(), obs, R, kw['residue_mask']),
        (obs, obs[0], R, kw['residue_mask']),
        (obs, obs, R.expand(2, -1, -1, -1), kw['residue_mask']),
        (obs, obs, R, kw['residue_mask'].float()),
        (obs, obs, R, torch.ones(2, 7, dtype=torch.bool)),
    ):
        with pytest.raises(ValueError):
            transport_observed_frames(bad_obs, bad_current, bad_R,
                **{**kw, 'residue_mask': bad_mask})
