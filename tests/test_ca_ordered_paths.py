"""CPU-only contracts for the ordered-path development harness; no data reads."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.eval_ca_ordered_paths import (
    DEFAULT_HORIZONS, _snapshot_sources, _write_json_exclusive, panel_spec,
    repository_sources, run, score_window, summarize, temporal_controls, validate_panel,
)
from evaluation.loaders import sha256_file


class FakeBundle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoded = []
        self.decoded = []

    def encode(self, bb, mask, aa, ri, ci):
        self.encoded.append(bb.clone())
        return bb[..., 1, :].clone()

    def decode_ca(self, h, h0, aa, ri, ci, rotation, anchor, mask):
        self.decoded.append((h0.clone(), anchor.clone()))
        return anchor + .3 * (h-h0) * mask[..., None]


class FakeKernel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(step_ps=1000.)
        self.s0_mode = 'exact_spatial'
        self.calls = 0
        self.transitions = []

    def rollout(self, h0, mask, temperature, steps, *, generator=None, return_bath=False):
        self.calls += 1
        h = h0.clone()
        s = torch.randn((*h.shape[:-1], 1), dtype=h.dtype, generator=generator)
        hs, ss = [h], [s]
        self.transitions = []
        for _ in range(steps):
            next_s = .8*s + .1*torch.randn(s.shape, dtype=s.dtype, generator=generator)
            next_h = h + .2*next_s.expand_as(h)
            self.transitions.append((h.clone(), s.clone(), next_h.clone(), next_s.clone()))
            h, s = next_h, next_s
            hs.append(h)
            ss.append(s)
        self.last_h, self.last_s = torch.stack(hs, 1), torch.stack(ss, 1)
        return (self.last_h, self.last_s) if return_bath else self.last_h


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bundle, self.kernel = FakeBundle(), FakeKernel()
        self.history_conditioning = True
        self.prediction_kind = 'endpoint'
        self.sample_calls = 0
        self.sampled_inputs = []

    def sample(self, observed, mask, aa, ri, ci, temperature, steps, n_samples, generator=None):
        self.sample_calls += 1
        self.sampled_inputs.append(tuple(value.clone() for value in (observed, mask, aa, ri, ci)))
        b, _, length = observed.shape[:3]
        repeat = lambda x: x[:, None].expand(b, n_samples, *x.shape[1:]).reshape(b*n_samples, *x.shape[1:])
        h0 = self.bundle.encode(observed[:, -1], mask, aa, ri, ci)
        start = repeat(h0)
        path = self.kernel.rollout(start, repeat(mask), temperature, steps, generator=generator)
        anchor = repeat(observed[:, -1, :, 1])
        rotation = torch.eye(3, dtype=observed.dtype).expand(b*n_samples, length, 3, 3)
        end_ca = self.bundle.decode_ca(path[:, -1], start, repeat(aa), repeat(ri), repeat(ci),
                                      rotation, anchor, repeat(mask))
        self.last_observation_steps = [steps]*(b*n_samples)
        return end_ca.reshape(b, n_samples, length, 3), path[:, -1].reshape(b, n_samples, length, 3)


def fixture(horizon=4):
    ca = torch.tensor([[0.,0.,0.], [3.8,0.,0.], [6.,3.,0.],
                       [5.,6.,2.], [2.,8.,1.], [-1.,6.,2.]], dtype=torch.float64)
    bb = ca[:, None] + ca.new_tensor([[-.5,1.3,0.], [0.,0.,0.], [1.5,0.,0.], [2.,1.,0.]])
    mask = torch.ones(6, dtype=torch.bool)
    observed = bb[None].repeat(8, 1, 1, 1)
    future = bb[None].repeat(horizon+1, 1, 1, 1)
    flex = torch.arange(6, dtype=torch.float64)[:, None, None] * bb.new_tensor([.03, -.01, .04])
    future += torch.arange(horizon+1, dtype=torch.float64)[:, None, None, None] * flex
    time = torch.arange(8, dtype=torch.float64)*1000.
    future_time = time[-1] + torch.arange(horizon+1, dtype=torch.float64)*1000.
    context = dict(coords=observed, residue_mask=mask[None].expand(8,-1).clone(),
        aatype=torch.tensor([0,1,2,3,20,5]), residue_index=torch.arange(6),
        chain_id=torch.zeros(6, dtype=torch.long), time_ps=time)
    window = SimpleNamespace(context=context,
        future_path=dict(coords=future, residue_mask=mask[None].expand(horizon+1,-1).clone(), time_ps=future_time),
        target=dict(coords=future[-1:].clone(), residue_mask=mask[None].clone(), time_ps=future_time[-1:].clone()),
        context_frames=torch.arange(8), protein_id='p', trajectory_id='r', temperature_k=320.,
        anchor_frame=7, target_frame=7+horizon, horizon=horizon, horizon_ps=horizon*1000.)
    return FakeModel().eval(), window


def test_one_continuous_rollout_no_reencoding_or_bath_restart():
    model, window = fixture()
    row = score_window(model, window, n_samples=3, decode_chunk_size=2)
    assert model.sample_calls == model.kernel.calls == len(model.bundle.encoded) == 1
    assert torch.equal(model.bundle.encoded[0], window.context['coords'][-1:])
    assert len(model.kernel.transitions) == 4
    for t, (h, s, next_h, next_s) in enumerate(model.kernel.transitions):
        assert torch.equal(h, model.kernel.last_h[:, t])
        assert torch.equal(next_h, model.kernel.last_h[:, t+1])
        assert torch.equal(s, model.kernel.last_s[:, t])
        assert torch.equal(next_s, model.kernel.last_s[:, t+1])
    for h0, anchor in model.bundle.decoded:
        assert torch.equal(h0, window.context['coords'][-1, :, 1].expand_as(h0))
        assert torch.equal(anchor, window.context['coords'][-1, :, 1].expand_as(anchor))
    assert row['continuous_path']['n_sample_calls'] == 1
    assert row['continuous_path']['future_reencodings'] == row['continuous_path']['bath_resets'] == 0
    assert row['continuous_path']['bath_shape'] == [1,3,5,6,1]
    assert 'rollout' not in model.kernel.__dict__
    assert all(parameter.grad is None for parameter in model.parameters())
    assert row['arms']['reference_MD']['n_paths'] == 1
    assert row['arms']['reference_MD']['metrics']['scalars']['coordinate_path_energy_score'] is None
    assert row['arms']['reference_MD']['metrics']['scalars']['physical_distance_path_energy_score'] is None
    assert row['arms']['reference_MD']['metrics']['scalars']['per_step_rmsd_mean'] < 1e-12
    assert len(row['arms']['forecast']['metrics']['per_path']) == 3
    assert len(row['arms']['forecast']['metrics']['lag_statistics']['reference']['per_path']) == 1


def test_future_change_alters_only_targets_oracle_and_scores_not_forecast():
    model, window = fixture()
    first = score_window(model, window, n_samples=3, decode_chunk_size=2)
    first_inputs = model.sampled_inputs[-1]
    window.future_path['coords'][1:, 3, :, 2] += 4.
    window.target['coords'] = window.future_path['coords'][-1:].clone()
    second = score_window(model, window, n_samples=3, decode_chunk_size=2)
    assert all(torch.equal(a,b) for a,b in zip(first_inputs, model.sampled_inputs[-1]))
    assert first['input_fingerprints']['forecast_CA_path'] == second['input_fingerprints']['forecast_CA_path']
    assert first['input_fingerprints']['observed'] == second['input_fingerprints']['observed']
    assert first['input_fingerprints']['reference'] != second['input_fingerprints']['reference']
    assert first['temporal_controls'] == second['temporal_controls']
    for arm in ('forecast','reversed_forecast','shuffled_forecast','persistence'):
        assert first['arms'][arm]['metrics']['per_path'] == second['arms'][arm]['metrics']['per_path']
        assert first['arms'][arm]['metrics']['lag_statistics']['prediction'] == second['arms'][arm]['metrics']['lag_statistics']['prediction']
    assert first['arms']['forecast']['metrics']['scalars']['coordinate_path_energy_score'] != second['arms']['forecast']['metrics']['scalars']['coordinate_path_energy_score']


def test_future_labels_are_not_read_until_the_single_forecast_is_complete():
    model, window = fixture()
    class Guarded:
        def __getattr__(self, key):
            if key in ('target','future_path'):
                assert model.sample_calls == 1
                assert len(model.kernel.transitions) == window.horizon
                # Original endpoint plus intermediate decoding must both finish.
                assert len(model.bundle.decoded) > 1
            return getattr(window, key)
    score_window(model, Guarded(), n_samples=2)


@pytest.mark.parametrize('change', ('observed_time','context_frames','horizon'))
def test_invalid_observed_or_requested_grid_fails_before_sampling(change):
    model, window = fixture()
    if change == 'observed_time':
        window.context['time_ps'][2] += 1.
    elif change == 'context_frames':
        window.context_frames[2] += 1
    else:
        window.horizon_ps += 1.
    with pytest.raises(ValueError, match='grid|horizon'):
        score_window(model, window, n_samples=2)
    assert model.sample_calls == 0


@pytest.mark.parametrize('change', ('future_time','missing_residue','endpoint','anchor'))
def test_invalid_real_path_fails_without_reselecting_mask_or_restarting(change):
    model, window = fixture()
    if change == 'future_time':
        window.future_path['time_ps'][2] += 1.
    elif change == 'missing_residue':
        window.future_path['residue_mask'][2, 1] = False
    elif change == 'endpoint':
        window.target['coords'][0, 1, 1, 2] += 1.
    else:
        window.future_path['coords'][0, 1, 1, 2] += 1.
    with pytest.raises(ValueError, match='grid|missing|endpoint|anchor'):
        score_window(model, window, n_samples=2)
    assert model.sample_calls == model.kernel.calls == len(model.bundle.encoded) == 1


def test_temporal_controls_keep_t0_future_bag_membership_and_rng_state():
    paths = torch.arange(2*8*5*3, dtype=torch.float64).reshape(2,8,5,3)
    before = torch.get_rng_state().clone()
    controls, meta = temporal_controls(paths, 73)
    assert torch.equal(torch.get_rng_state(), before)
    assert meta == temporal_controls(paths, 73)[1]
    assert meta['reverse_time_indices'] == [0,7,6,5,4,3,2,1]
    for control in controls.values():
        assert torch.equal(control[:,0], paths[:,0])
        assert torch.equal(control[:,1:].sort(dim=1).values, paths[:,1:].sort(dim=1).values)
    assert meta['shuffle_time_indices'] == [0, *sorted(meta['shuffle_time_indices'][1:])] or meta['shuffle_changes_order']
    short, short_meta = temporal_controls(paths[:,:2], 73)
    assert not short_meta['reverse_changes_order'] and not short_meta['shuffle_changes_order']
    assert all(torch.equal(x,paths[:,:2]) for x in short.values())


def test_unknown_residue_is_not_excluded_from_CA_path_scores():
    model, window = fixture()
    row = score_window(model, window, n_samples=2)
    assert row['valid_residues'] == 6
    assert model.sampled_inputs[0][2][0,4] == 20
    assert model.sampled_inputs[0][1].all()


def test_observed_nonprefix_mask_is_fixed_even_when_future_has_more_valid_residues():
    model, window = fixture()
    window.context['residue_mask'][2, 1] = False
    row = score_window(model, window, n_samples=2)
    assert row['valid_residues'] == 5
    assert model.sampled_inputs[0][1].tolist() == [[True,False,True,True,True,True]]
    for arm in row['arms'].values():
        assert arm['metrics']['conventions']['n_valid_residues'] == 5


def test_samples_then_windows_then_proteins_and_counts_not_performance():
    model, window = fixture()
    a = score_window(model, window, n_samples=2)
    b, c = copy.deepcopy(a), copy.deepcopy(a)
    b['anchor_frame'] += 1
    c['protein'] = 'q'
    for row, value in ((a,1.),(b,1.),(c,3.)):
        row['arms']['forecast']['metrics']['scalars']['coordinate_path_energy_score'] = value
    result = summarize([a,b,c])['4']
    assert result['means']['forecast']['coordinate_path_energy_score'] == dict(mean=2.,n_proteins=2)
    assert result['n_windows'] == 3
    assert 'distance_acf_corr_n_lags' in result['count_fields_retained_per_window']
    assert 'distance_acf_corr_n_lags' not in result['means']['forecast']
    with pytest.raises(ValueError,match='duplicate'):
        summarize([a,a])


def test_fixed_panels_and_smoke_prefixes_are_explicit_and_cannot_be_selected():
    assert panel_spec(DEFAULT_HORIZONS)['max_windows'] == 200
    assert panel_spec(DEFAULT_HORIZONS)['full_panel']
    assert panel_spec((192,))['max_windows'] == 50
    assert not panel_spec(DEFAULT_HORIZONS,4)['full_panel']
    assert not panel_spec((192,),1)['full_panel']
    for horizons, count in (((1,4),None),(DEFAULT_HORIZONS,5),(DEFAULT_HORIZONS,204),((192,),51)):
        with pytest.raises(ValueError):
            panel_spec(horizons,count)


def test_run_refuses_artifact_overwrite_before_any_checkpoint_or_data_read(tmp_path):
    output = tmp_path/'scores.json'
    output.write_text('preserved')
    args = SimpleNamespace(manifest='artifacts/mdcath_manifest_full/val.jsonl',
        horizons=DEFAULT_HORIZONS, max_windows=4, torch_threads=1, decode_chunk_size=2,
        checkpoint=str(tmp_path/'missing.pt'), out=str(output), device='cpu')
    with pytest.raises(FileExistsError,match='overwrite'):
        run(args)
    assert output.read_text() == 'preserved'


def test_source_snapshot_includes_loaded_adapter_metrics_and_data_reader_dependencies():
    sources = repository_sources()
    assert len(sources) == len(set(sources))
    for path in ('evaluation/ca_trajectory_adapter.py','evaluation/trajectory_metrics_v2.py',
                 'protein_jepa_frame/data/dataset.py','protein_jepa_frame/data/manifest.py',
                 'protein_jepa_frame/models/stage_c/geometry.py'):
        assert path in sources
    assert not any(path.startswith('tests/') for path in sources)


def test_full_panel_enforces_per_protein_condition_horizons_and_temperature_counts():
    def rows_for(horizons):
        return [dict(protein=str(p), trajectory='r', anchor_frame=7,
                     temperature_k=(320.,348.,379.,413.,450.)[p%5], horizon=h)
                for p in range(50) for h in horizons]
    standard = rows_for(DEFAULT_HORIZONS)
    validate_panel(standard, panel_spec(DEFAULT_HORIZONS))
    validate_panel(rows_for((192,)), panel_spec((192,)))
    validate_panel(standard[:4], panel_spec(DEFAULT_HORIZONS,4))
    for key, value in (('horizon', 1), ('anchor_frame', 8), ('temperature_k', 348.)):
        wrong = copy.deepcopy(standard)
        wrong[1][key] = value
        with pytest.raises(RuntimeError):
            validate_panel(wrong, panel_spec(DEFAULT_HORIZONS))
    wrong = copy.deepcopy(standard)
    for row in wrong[:4]:
        row['temperature_k'] = 348.
    with pytest.raises(RuntimeError, match='allocation'):
        validate_panel(wrong, panel_spec(DEFAULT_HORIZONS))


def test_exclusive_writes_and_snapshot_verify_registered_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = tmp_path/'score.json'
    _write_json_exclusive(output, {'value': 1})
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        _write_json_exclusive(output, {'value': 2})
    assert output.read_bytes() == before
    source = tmp_path/'source.py'
    source.write_text('original source')
    expected = sha256_file(source)
    snapshot = tmp_path/'snapshot'
    _snapshot_sources(snapshot, {'source.py': expected})
    assert sha256_file(snapshot/'source.py') == expected
    with pytest.raises(FileExistsError):
        _snapshot_sources(snapshot, {'source.py': expected})
    source.write_text('changed after registration')
    with pytest.raises(RuntimeError, match='snapshot differs'):
        _snapshot_sources(tmp_path/'changed_snapshot', {'source.py': expected})
