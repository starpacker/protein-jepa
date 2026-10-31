import numpy as np
import pytest
import torch
from scripts.diag_lasp_components import correlation, rmsd
from scripts.diag_lasp_ca_ceiling import (
    _development_manifest, _same_component, endpoint_probe, latent_transition_terms,
)


def test_diagnostics_reject_unapproved_manifest_before_reading(tmp_path):
    manifest = tmp_path / 'development.jsonl'
    manifest.symlink_to(tmp_path / 'test.jsonl')
    with pytest.raises(ValueError, match='approved development'):
        _development_manifest(manifest)
    with pytest.raises(ValueError, match='approved development'):
        _development_manifest(tmp_path / 'test.jsonl')


def test_geometry_screen_ignores_gaps_and_locates_bad_sample():
    from scripts.diagnose_lasp_bounded_energy import geometry_summary
    ca = torch.tensor([[[0., 0., 0.], [3.8, 0., 0.], [100., 0., 0.]],
                       [[0., 0., 0.], [5.2, 0., 0.], [100., 0., 0.]]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    ri = torch.tensor([[10, 11, 20], [10, 11, 20]])
    ci = torch.zeros_like(ri)
    result = geometry_summary(ca, mask, ri, ci)
    assert result['bond_count'] == 2 and result['invalid_bonds'] == 1
    assert result['worst_sample'] == 1
    assert result['worst_residue_indices'] == [10, 11]
    assert result['bond_max'] == pytest.approx(5.2)


def test_bond_probe_identity_length_mismatch_and_manifest_guard(tmp_path):
    from scripts.diagnose_lasp_bounded_energy import bond_length_probe, run_bond_representation_probe
    anchor = torch.tensor([[[0., 0., 0.], [3.8, 0., 0.], [7.6, 0., 0.]]], dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    ri = torch.arange(3)[None]
    ci = torch.zeros_like(ri)
    result = bond_length_probe(anchor, anchor, mask, ri, ci)
    assert result['finite'] and result['oracle_coordinate_rms_shift'] < 1e-12
    result = bond_length_probe(anchor, anchor*1.01, mask, ri, ci)
    assert result['bond_mismatch_rms'] == pytest.approx(.038)
    assert result['oracle_coordinate_rms_shift'] > 0
    with pytest.raises(ValueError, match='approved train/development'):
        run_bond_representation_probe(tmp_path/'test.jsonl', tmp_path/'out.json')


def test_retraction_gate_blocks_horizon_clashes_and_endpoint_regression():
    from copy import deepcopy
    from scripts.diagnose_lasp_bounded_energy import retraction_acceptance
    baseline = dict(bonds=dict(invalid_bonds=0, finite=True), gradient_finite=True,
                    coordinate_endpoint_es=1., clash_fraction=0.,
                    ca_angle_cos_mae=.3, ca_clash_fraction=0.)
    rows = [dict(horizon=h, rms_finest_pair=1e-5,
                 modes={k: deepcopy(baseline) for k in ('32', '256', '512')})
            for h in (1, 4, 16, 64)]
    assert retraction_acceptance(rows, '256', '512')['passed']
    rows[-1]['modes']['256']['ca_clash_fraction'] = .00011
    result = retraction_acceptance(rows, '256', '512')
    assert not result['passed'] and not result['checks']['calibration_clashes']
    assert result['geometry_deltas']['overall']['ca_clash_fraction'] < .0001
    rows[-1]['modes']['256']['ca_clash_fraction'] = 0.
    rows[0]['modes']['256']['coordinate_endpoint_es'] = 1.021
    assert not retraction_acceptance(rows, '256', '512')['checks']['endpoint_score']


def test_component_comparison_checks_nested_weights_and_metadata():
    component = {'encoder': {'weight': torch.tensor([1., 2.])}, 'mode': 'ca'}
    identical = {'encoder': {'weight': torch.tensor([1., 2.])}, 'mode': 'ca'}
    assert _same_component(component, identical)
    identical['encoder']['weight'][0] = 3.
    assert not _same_component(component, identical)
    assert not _same_component(component, None)
    assert _same_component({'rng': [np.arange(3)]}, {'rng': [np.arange(3)]})
    assert not _same_component(np.arange(3), np.arange(4))



def test_endpoint_probe_masks_padding_and_keeps_best_fit():
    class IdentityBundle:
        def decode_ca(self, h, *args):
            return h

    h = torch.zeros(1, 3, 3)
    target = torch.ones_like(h)
    target[:, -1] = 1000.
    mask = torch.tensor([[True, True, False]])
    args = (IdentityBundle(), h, h, None, None, None, None, None, mask, target)
    initial = endpoint_probe(*args, steps=0)
    result = endpoint_probe(*args, steps=20, lr=.1)
    assert abs(initial['encoded_endpoint_rmsd']-3**.5) < 1e-6
    assert result['optimized_endpoint_rmsd'] < initial['encoded_endpoint_rmsd']*.3
    assert result['latent_change_rms'] > 0
    assert torch.equal(h, torch.zeros_like(h))


def test_latent_transition_terms_masks_padding_and_normalizes_collective_dimension():
    anchor = torch.zeros(1, 3, 2)
    target = anchor + torch.tensor([[[1., 0.], [0., 2.], [100., 100.]]])
    generated = torch.stack([
        anchor + torch.tensor([[[1., 0.], [0., 2.], [999., 999.]]]),
        anchor + torch.tensor([[[1., 0.], [0., 2.], [-999., -999.]]]),
    ], dim=1)
    mask = torch.tensor([[True, True, False]])
    terms = latent_transition_terms(generated, anchor, target, mask)
    assert terms['latent_dimension'] == 4
    assert terms['true_delta_norm'] == pytest.approx(5**.5)
    assert terms['latent_displacement_cosine_mean'] > .9
    assert terms['latent_displacement_norm_ratio_mean'] == pytest.approx(1.)
    assert all(torch.isfinite(torch.tensor(value)) for value in terms.values()
               if isinstance(value, float))


def test_collective_projection_exposes_internal_motion_cancellation():
    anchor = torch.zeros(1, 2, 1)
    target = torch.tensor([[[1.], [-1.]]])
    generated = torch.zeros(1, 2, 2, 1)
    terms = latent_transition_terms(generated, anchor, target, torch.ones(1, 2, dtype=torch.bool))
    assert terms['true_collective_energy_fraction'] == 0.
    assert terms['collective_displacement_energy_score'] == 0.
    assert terms['latent_endpoint_energy_score'] == pytest.approx(1.)
    shifted = latent_transition_terms(generated + 7., anchor + 7., target + 7.,
                                      torch.ones(1, 2, dtype=torch.bool))
    assert shifted['latent_displacement_energy_score'] == pytest.approx(1.)


def test_collective_fraction_is_one_for_uniform_residue_motion():
    anchor = torch.zeros(1, 3, 2)
    target = torch.ones_like(anchor)
    generated = target[:, None].expand(1, 2, 3, 2).clone()
    terms = latent_transition_terms(generated, anchor, target, torch.ones(1, 3, dtype=torch.bool))
    assert terms['true_collective_energy_fraction'] == pytest.approx(1.)
    assert terms['generated_collective_energy_fraction'] == pytest.approx(1.)
    assert terms['collective_displacement_energy_score'] == pytest.approx(0.)
    assert terms['latent_persistence_energy_score'] == pytest.approx(1.)


def test_latent_diagnostics_reject_empty_mask_and_single_sample():
    anchor = torch.zeros(1, 2, 1)
    generated = torch.zeros(1, 2, 2, 1)
    with pytest.raises(ValueError, match='valid residues'):
        latent_transition_terms(generated, anchor, anchor, torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match='two samples'):
        latent_transition_terms(generated[:, :1], anchor, anchor, torch.ones(1, 2, dtype=torch.bool))


def test_component_rmsd_removes_rigid_motion_per_frame():
    torch.manual_seed(2)
    true = torch.randn(3, 8, 3, dtype=torch.float64)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],dtype=true.dtype)
    pred = true @ rotation + torch.tensor([4., 6., 2.])
    assert rmsd(pred,true) < 1e-10
    pred[:,0] += 1
    assert rmsd(pred,true) > 0.1


def test_degenerate_representation_correlation_is_explicit():
    assert correlation(torch.zeros(5), torch.arange(5.)) is None
    assert abs(correlation(torch.arange(5.),torch.arange(5.) * 3) - 1) < 1e-10
