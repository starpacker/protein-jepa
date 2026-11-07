import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.eval_ca_backbone_diagnostic import (
    align_target_backbone, continuous_segments, lift_segmented, score_window, summarize, audit_legacy_basis,
    chemical_aatype,
)
from protein_jepa_frame.utils.contracts import state_sha256


def fixture():
    ca = torch.tensor([[0., 0., 0.], [3.8, 0., 0.], [6., 3., 0.],
                       [5., 6., 2.], [2., 8., 1.], [-1., 6., 2.]])
    mask, aa, ri, ci = torch.ones(6, dtype=torch.bool), torch.zeros(6, dtype=torch.long), torch.arange(6), torch.zeros(6, dtype=torch.long)
    bb = ca[:, None] + torch.tensor([[-.5, 1.3, 0.], [0., 0., 0.], [1.5, 0., 0.], [2., 1., 0.]])
    return ca, bb, mask, aa, ri, ci


def test_target_CA_fit_applies_same_transform_to_all_atoms_and_ignores_masked_nan():
    ca, bb, mask, *_ = fixture()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transformed = bb @ rotation + torch.tensor([5., -3., 7.])
    assert torch.allclose(align_target_backbone(transformed, ca, mask), bb, atol=1e-6)
    mask[-1] = False
    transformed[-1] = float('nan')
    aligned = align_target_backbone(transformed, ca, mask)
    assert torch.allclose(aligned[mask], bb[mask], atol=1e-6)
    assert torch.equal(aligned[~mask], torch.zeros_like(aligned[~mask]))
    transformed[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        align_target_backbone(transformed, ca, mask)


def test_lift_segments_never_cross_mask_chain_or_index_gaps_and_preserve_CA():
    ca, _, mask, aa, ri, ci = fixture()
    ri[3:] += 7
    ci[-1] = 1
    assert continuous_segments(mask, ri, ci) == [[0,1,2], [3,4], [5]]
    mask[1] = False
    assert continuous_segments(mask, ri, ci) == [[0], [2], [3,4], [5]]
    samples = ca[None].repeat(2, 1, 1)
    samples[1, 3, 2] += 1
    samples[:, 1] = float('nan')
    result, report = lift_segmented(samples, aa, mask, ri, ci)
    assert torch.equal(result[:, mask, 1], samples[:, mask])
    assert report['segment_lengths'] == [1,1,2,1]
    assert report['terminal_residue_fraction'] == 1.
    for i, segment in enumerate(continuous_segments(mask, ri, ci)):
        from protein_jepa_frame.models.stage_c.openfold_ca_lifter import lift_ca_to_backbone
        expected, _, _ = lift_ca_to_backbone(samples[:, segment], aatype=aa[segment][None].repeat(2, 1))
        assert torch.equal(result[:, segment], expected)
    aa[0] = 20
    unknown_bb, info = lift_segmented(samples, aa, mask, ri, ci)
    assert torch.equal(unknown_bb[:,mask,1], samples[:,mask])
    assert info['unknown_gly_backbone_template_residues'] == 1


def fake_window_and_models():
    ca, bb, mask, aa, ri, ci = fixture()
    observed = bb[None].repeat(8, 1, 1, 1)
    target = bb.clone()
    target[3] += torch.tensor([0., 0., 2.])
    window = SimpleNamespace(context=dict(coords=observed, residue_mask=mask[None].repeat(8,1),
        aatype=aa, residue_index=ri, chain_id=ci),
        target=dict(coords=target[None], residue_mask=mask[None]),
        future_path=object(), protein_id='p', trajectory_id='r', temperature_k=320.,
        anchor_frame=7, target_frame=11, horizon=4, horizon_ps=4000., context_frames=np.arange(8))
    class Bundle:
        def encode(self, coords, *metadata):
            return coords[...,1,:]
        def decode_ca(self, h, anchor_h, aa, ri, ci, rotation, anchor, mask):
            return anchor + .2*(h-anchor_h)
    class Model:
        training = False
        prediction_kind = 'endpoint'
        kernel = SimpleNamespace(cfg=SimpleNamespace(step_ps=1000.))
        bundle = Bundle()
        def sample(self, context, actual_mask, aatype, residue_index, chain_id, temp, steps, count, generator):
            assert torch.equal(context[0], observed)
            self.last_observation_steps = [steps]*count
            self.last_samples = context[:,-1,:,1][:,None].repeat(1,count,1,1)
            self.last_samples += .1*torch.randn(self.last_samples.shape, generator=generator)
            return self.last_samples, self.last_samples*.1
    class Legacy:
        def decode_structure(self, h, aa, rotation, anchor, mask):
            offsets = bb-ca[:,None]
            output = anchor[:,:,None]+.1*h[:,:,None]+offsets
            return dict(backbone_coords=output, atom37_mask=torch.ones((*h.shape[:2],37),dtype=torch.bool))
    def original(coords, mask, aa, ri, ci, token_mask):
        return coords[...,1,:]*.2
    return window, Model(), original, Legacy()


def test_future_label_changes_scores_and_oracles_but_never_forecast_geometry_or_samples():
    window, model, original, legacy = fake_window_and_models()
    first = score_window(model, original, legacy, window)
    samples = model.last_samples.clone()
    window.target['coords'][0, 3, :, 2] += 4
    second = score_window(model, original, legacy, window)
    assert torch.equal(samples, model.last_samples)
    assert first['sampling_seed'] == second['sampling_seed']
    for route in ('forecast_ca_ideal_lift', 'forecast_H_legacy_b1a', 'persistence_MD'):
        assert first['routes'][route]['geometry'] == second['routes'][route]['geometry']
        assert first['routes'][route]['error'] != second['routes'][route]['error']
    assert first['routes']['oracle_current_H_ca_ideal_lift']['geometry'] != second['routes']['oracle_current_H_ca_ideal_lift']['geometry']
    assert len(first['routes']['forecast_ca_ideal_lift']['geometry']['per_sample']) == 12
    assert first['routes']['target_MD']['error']['mean']['coordinate_rmsd_angstrom'] == 0.
    assert len(first['legacy_vs_current_forecast_CA_rmsd_per_sample']) == 12
    assert any(v>0 for v in first['legacy_vs_current_forecast_CA_rmsd_per_sample'])


def test_wrong_horizon_or_training_mode_is_rejected():
    window, model, original, legacy = fake_window_and_models()
    model.training = True
    with pytest.raises(ValueError, match='eval-mode'):
        score_window(model, original, legacy, window)
    model.training = False
    window.horizon_ps = 1000.
    with pytest.raises(ValueError, match='physical'):
        score_window(model, original, legacy, window)


def test_chemical_order_conversion_and_unknown_coverage_do_not_change_sampler_mask():
    # Dataset A/C/D/.../P/.../X versus OpenFold A/R/N/D/C/.../P/.../X.
    assert chemical_aatype(torch.tensor([0, 1, 2, 12, 20])).tolist() == [0, 4, 3, 14, 20]
    window, model, original, legacy = fake_window_and_models()
    window.context['aatype'][2] = 20
    result = score_window(model, original, legacy, window)
    assert result['valid_residues'] == result['scored_residues'] == 6
    assert result['unknown_residues_excluded'] == 0 and result['chemical_unknown_residues'] == 1
    assert result['unknown_residue_index'] == [2]
    for route, value in result['routes'].items():
        assert all(sample['n_atoms'] == 24 for sample in value['error']['per_sample'])
        assert all(sample['openfold_bond_N_CA_n'] == 5 for sample in value['geometry']['per_sample'])
    assert result['routes']['forecast_ca_ideal_lift']['lift']['segment_lengths'] == [6]


def test_aggregation_equal_proteins_and_duplicate_rejection():
    window, model, original, legacy = fake_window_and_models()
    a = score_window(model, original, legacy, window)
    b, c = copy.deepcopy(a), copy.deepcopy(a)
    b['anchor_frame'] += 1
    c['protein'] = 'q'
    for r, value in ((a,1.), (b,1.), (c,3.)):
        r['routes']['forecast_ca_ideal_lift']['error']['mean']['coordinate_rmsd_angstrom'] = value
    result = summarize([a,b,c])['4']
    assert result['means']['forecast_ca_ideal_lift']['error']['coordinate_rmsd_angstrom'] == dict(mean=2., n_proteins=2)
    with pytest.raises(ValueError, match='duplicate'):
        summarize([a,a])


def test_native_basis_requires_tensor_and_checkpoint_contracts_not_just_width():
    encoder = torch.nn.Linear(3, 3)
    decoder = SimpleNamespace(decoder=torch.nn.Linear(3, 4))
    checkpoint = dict(encoder=copy.deepcopy(encoder.state_dict()), representation_contract=dict(
        decoder_input='raw_H', encoder_sha256=state_sha256(encoder), decoder_sha256=state_sha256(decoder.decoder),
        source=dict(stage_a_checkpoint_sha256='source')))
    assert audit_legacy_basis(encoder, decoder, checkpoint, 'source')['native_basis_verified']
    with pytest.raises(ValueError, match='contract mismatch'):
        audit_legacy_basis(encoder, decoder, checkpoint, 'other-source')
    with torch.no_grad():
        encoder.weight.add_(.01)
    with pytest.raises(ValueError, match='contract mismatch'):
        audit_legacy_basis(encoder, decoder, checkpoint, 'source')
