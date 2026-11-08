import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.geometry import kabsch_align, physical_distance_energy_score
from scripts.eval_ca_endpoint_scores import canonical_pairs, endpoint_scores, score_window, summarize


def geometry():
    anchor = torch.tensor([[0., 0., 0.], [3.8, 0., 0.], [6., 3., 0.],
                           [5., 6., 2.], [2., 8., 1.], [-1., 6., 2.]], dtype=torch.float64)
    return anchor, torch.ones(6, dtype=torch.bool), torch.arange(6), torch.zeros(6, dtype=torch.long)


def test_oracle_and_persistence_coordinate_units_and_anchor_gauge():
    anchor, mask, ri, ci = geometry()
    endpoint = anchor.clone()
    endpoint[3, 2] += 2
    aligned = kabsch_align(endpoint, anchor, mask)
    oracle = endpoint_scores(aligned[None].expand(12, -1, -1), endpoint, anchor, mask, ri, ci)
    assert all(abs(value) < 1e-12 for value in oracle.values())
    persistence = endpoint_scores(anchor[None].expand(12, -1, -1), endpoint, anchor, mask, ri, ci)
    expected = float((anchor - aligned).square().mean().sqrt())
    assert persistence['coordinate_energy_score'] == pytest.approx(expected, abs=1e-12)
    shifted = endpoint_scores(aligned[None].expand(12, -1, -1) + 5, endpoint, anchor, mask, ri, ci)
    assert shifted['coordinate_energy_score'] == pytest.approx(5.)
    assert shifted['physical_distance_energy_score'] < 1e-12


def test_scores_preserve_u_and_v_statistics_and_hard_contact_brier():
    anchor, mask, ri, ci = geometry()
    samples = anchor[None].repeat(3, 1, 1)
    samples[1, 4, 1] -= 4
    samples[2, 4, 1] -= 8
    result = endpoint_scores(samples, anchor, anchor, mask, ri, ci)
    flat = samples.flatten(1)
    accuracy = (flat - anchor.flatten()).norm(dim=-1).mean()
    offdiagonal = torch.stack([(flat[i] - flat[j]).norm() for i in range(3) for j in range(i)]).mean()
    assert result['coordinate_energy_score'] == pytest.approx(float((accuracy - .5 * offdiagonal) / 18**.5))
    expected_distance = physical_distance_energy_score(samples, anchor, mask)
    assert result['physical_distance_energy_score'] == pytest.approx(expected_distance)
    i, j = canonical_pairs(mask, ri, ci)
    distances = (samples[:, i] - samples[:, j]).norm(dim=-1)
    truth_distances = (anchor[i] - anchor[j]).norm(dim=-1)
    v_score = ((distances - truth_distances).norm(dim=-1).mean()
               - .5 * torch.cdist(distances, distances).mean()) / len(i)**.5
    assert result['physical_distance_energy_score'] == pytest.approx(float(v_score))
    probability = ((samples[:, i] - samples[:, j]).norm(dim=-1) < 8).double().mean(0)
    observed = ((anchor[i] - anchor[j]).norm(dim=-1) < 8).double()
    assert result['contact_brier'] == pytest.approx(float(((probability - observed)**2).mean()))
    assert result['contact_brier'] > 0


def test_physical_pair_mismatch_and_nonfinite_values_fail_closed():
    anchor, mask, ri, ci = geometry()
    ci[-1] = 1
    with pytest.raises(ValueError, match='physical'):
        canonical_pairs(mask, ri, ci)
    ci[:] = 0
    samples = anchor[None].repeat(2, 1, 1)
    samples[0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        endpoint_scores(samples, anchor, anchor, mask, ri, ci)


def window_and_model():
    anchor, mask, ri, ci = geometry()
    coords = anchor[:, None].expand(-1, 4, -1).clone()
    context = coords[None].repeat(8, 1, 1, 1)
    window = SimpleNamespace(context=dict(coords=context, residue_mask=mask[None].repeat(8, 1),
        aatype=torch.zeros(6, dtype=torch.long), residue_index=ri, chain_id=ci),
        target=dict(coords=coords[None].clone(), residue_mask=mask[None]),
        future_path=object(), protein_id='p', trajectory_id='r', temperature_k=320.,
        anchor_frame=7, target_frame=11, horizon=4, horizon_ps=4000., context_frames=np.arange(8))
    class Model:
        prediction_kind = 'endpoint'
        kernel = SimpleNamespace(cfg=SimpleNamespace(step_ps=1000.))
        def sample(self, observed, actual_mask, aa, actual_ri, actual_ci, temp, steps, count, generator):
            assert torch.equal(observed[0], context)
            assert torch.equal(actual_mask[0], mask)
            self.last_observation_steps = [steps] * count
            # Actual stochastic draws test stateless seed reuse when labels change.
            self.last_samples = observed[:, -1, :, 1][:, None].repeat(1, count, 1, 1)
            self.last_samples += .1 * torch.randn(self.last_samples.shape, generator=generator)
            return self.last_samples, None
    return window, Model()


def test_sampler_has_context_only_stateless_endpoints_and_canonical_sample_geometry():
    window, model = window_and_model()
    first = score_window(model, window)
    original_samples = model.last_samples.clone()
    window.target['coords'][0, 3, :, 2] += 3
    second = score_window(model, window)
    assert torch.equal(original_samples, model.last_samples)
    assert first['sampling_seed'] == second['sampling_seed']
    assert first['scores'] != second['scores']
    assert first['actual_n_samples'] == 12
    assert first['observation_steps'] == [4] * 12
    assert len(first['sample_geometry']) == 12
    assert 'ca_bond_invalid_fraction' in first['mean_sample_geometry']
    assert 'ca_nonlocal_clash_fraction' in first['target_geometry']
    assert first['persistence_geometry'] == second['persistence_geometry']


def test_wrong_physical_horizon_and_mixture_sampling_are_rejected():
    window, model = window_and_model()
    window.horizon_ps = 1000.
    with pytest.raises(ValueError, match='physical horizon'):
        score_window(model, window)
    window.horizon_ps = 4000.
    model.prediction_kind = 'time_mixture'
    with pytest.raises(ValueError, match='endpoint sampling'):
        score_window(model, window)


def test_summary_is_equal_protein_weighted_and_duplicate_windows_fail():
    window, model = window_and_model()
    first = score_window(model, window)
    first['scores']['model'] = {key: 1. for key in first['scores']['model']}
    second = copy.deepcopy(first)
    second['protein'] = 'q'
    second['scores']['model'] = {key: 3. for key in second['scores']['model']}
    extra = copy.deepcopy(first)
    extra['anchor_frame'] += 1
    result = summarize([first, second, extra])['4']
    assert result['means']['model']['coordinate_energy_score'] == 2.
    assert result['n_windows'] == 3 and result['n_proteins'] == 2
    with pytest.raises(ValueError, match='duplicate'):
        summarize([first, first])


def test_loader_checks_embedded_decoder_identity(monkeypatch):
    import scripts.eval_ca_endpoint_scores as evaluator
    model = SimpleNamespace(kernel=torch.nn.Linear(1, 1), bundle=SimpleNamespace(
        encoder=torch.nn.Linear(1, 1), decoder=torch.nn.Linear(1, 1)))
    model.eval = lambda: model
    checkpoint = dict(checkpoint_kind='lasp_ca_dynamics', args={},
        kernel=copy.deepcopy(model.kernel.state_dict()), component_checkpoint=dict(
            encoder=copy.deepcopy(model.bundle.encoder.state_dict()),
            delta_ca_decoder=copy.deepcopy(model.bundle.decoder.state_dict())))
    monkeypatch.setattr(evaluator, 'CADynamics', lambda *args, **kwargs: model)
    assert evaluator.load_model('synthetic.pt', checkpoint, 'cpu') is model
    with torch.no_grad():
        model.bundle.decoder.weight.add_(1.)
    with pytest.raises(ValueError, match='decoder'):
        evaluator.load_model('synthetic.pt', checkpoint, 'cpu')
