"""Focused CPU contract tests; these never run a training loop."""
from dataclasses import asdict
import json

import pytest
import torch

from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training import train_lasp_ca_dynamics as base
from protein_jepa_frame.training import train_lasp_bounded_endpoint as bounded


class FakeBundle(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.ones(()), requires_grad=False)

    def encode(self, observed, mask, aa, ri, ci):
        return observed[:, :, 1, :2].detach()

    def decode_ca(self, h, start, aa, ri, ci, rotation, anchor, mask):
        return anchor + torch.nn.functional.pad(h-start, (0, 1))


@pytest.fixture
def model(tmp_path, monkeypatch):
    cfg = StageB1bConfig(latent_dim=2, bath_dim=1, energy_heads=1,
                        energy_hidden=4, energy_depth=1, history_hidden=4,
                        step_ps=1.)
    old = StageB1b(cfg)
    path = tmp_path / 'checkpoint_best.pt'
    torch.save(dict(checkpoint_kind='lasp_ca_dynamics', kernel=old.state_dict(),
                    kernel_config=asdict(cfg), args={'noise_mix': .25, 'noise_length': 7.}), path)
    monkeypatch.setattr(base, 'CAComponentBundle', FakeBundle)
    monkeypatch.setattr(bounded, '_extract_backbone_frames',
                        lambda x: (torch.eye(3).expand(x.shape[0], x.shape[1], 3, 3), x[:, :, 1]))
    result = bounded.BoundedEndpointDynamics('unused', 'cpu', path, microsteps=2)
    return result, old, path


def test_fresh_energy_retains_bath_and_legacy_rejects(model, tmp_path):
    new, old, path = model
    for k, value in old.bath.state_dict().items():
        assert torch.equal(value, new.kernel.bath.state_dict()[k])
    assert isinstance(new.kernel.energy, bounded.BoundedRidgeEnergy)
    with pytest.raises(RuntimeError):
        old.load_state_dict(new.kernel.state_dict(), strict=True)
    assert not new.history_conditioning
    assert all(not p.requires_grad for p in new.bundle.parameters())
    assert new.integrator_metadata()['full_bptt']
    assert new.kernel.cfg.step_ps == 1.
    assert new.noise_mix == .25 and new.noise_length == 7.
    assert new.integrator_metadata()['noise_mix'] == .25
    again = bounded.BoundedEndpointDynamics('unused', 'cpu', path, microsteps=2)
    for k, value in new.kernel.energy.state_dict().items():
        assert torch.equal(value, again.kernel.energy.state_dict()[k])
    checkpoint = tmp_path / 'bounded.pt'
    torch.save(dict(checkpoint_kind=bounded.CHECKPOINT_KIND,
                    schema_version=bounded.SCHEMA_VERSION, test_opened=False,
                    energy_seed=new.energy_seed, integrator=new.integrator_metadata(),
                    kernel=new.kernel.state_dict(), kernel_config=asdict(new.kernel.cfg),
                    args={'noise_mix': new.noise_mix, 'noise_length': new.noise_length}), checkpoint)
    loaded, saved = bounded.load_bounded_checkpoint(checkpoint, 'unused', 'cpu')
    assert saved['checkpoint_kind'] == bounded.CHECKPOINT_KIND
    assert loaded.noise_mix == new.noise_mix and loaded.noise_length == new.noise_length
    for k, value in loaded.kernel.state_dict().items():
        assert torch.equal(value, new.kernel.state_dict()[k])


def test_sample_generator_clock_and_full_bptt(model, monkeypatch):
    new, _, _ = model
    new.train()
    observed = torch.randn(1, 4, 4, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    aa = ci = torch.zeros(1, 4, dtype=torch.long)
    ri = torch.arange(4)[None]
    original = new.kernel.rollout_microstep_v1
    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        captured['steps'] = args[3]
        args[0].requires_grad_(True)
        captured['initial'] = args[0]
        return original(*args, **kwargs)

    monkeypatch.setattr(new.kernel, 'rollout_microstep_v1', spy)
    def sample():
        return new.sample(observed, mask, aa, ri, ci, torch.tensor([348.]), 3, 2,
                          torch.Generator().manual_seed(5))
    ca, latent = sample()
    initial = captured['initial']
    latent.square().sum().backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    assert any(p.grad is not None for p in new.kernel.energy.parameters())
    assert any(p.grad is not None for p in new.kernel.bath.parameters())
    assert captured['steps'] == 3
    assert captured['grad_steps'] is None and captured['detach'] is False
    assert captured['seq_feature'] is captured['initial']
    assert captured['H_hist'] is None
    assert captured['integrator'] == 'coupled_midpoint_v1'
    assert captured['microsteps'] == 2
    ca2, latent2 = sample()
    assert torch.equal(ca, ca2) and torch.equal(latent, latent2)
    source = torch.load(model[2], map_location='cpu', weights_only=False)
    checkpoint = model[2].parent / 'bounded.pt'
    torch.save(dict(checkpoint_kind=bounded.CHECKPOINT_KIND,
                    schema_version=bounded.SCHEMA_VERSION, test_opened=False,
                    energy_seed=new.energy_seed, integrator=new.integrator_metadata(),
                    kernel=new.kernel.state_dict(), kernel_config=asdict(new.kernel.cfg),
                    args=source['args']), checkpoint)
    loaded, _ = bounded.load_bounded_checkpoint(checkpoint, 'unused', 'cpu')
    assert loaded.noise_mix == new.noise_mix and loaded.noise_length == new.noise_length
    loaded.eval()
    new.eval()
    with torch.no_grad():
        loaded_result = loaded.sample(observed, mask, aa, ri, ci, torch.tensor([348.]), 3, 2,
                                      torch.Generator().manual_seed(19))
        original_result = new.sample(observed, mask, aa, ri, ci, torch.tensor([348.]), 3, 2,
                                     torch.Generator().manual_seed(19))
    assert all(torch.equal(x, y) for x, y in zip(loaded_result, original_result))
    with pytest.raises(ValueError, match='full BPTT'):
        new.sample(observed, mask, aa, ri, ci, None, 3, 2, grad_steps=1)


def test_matched_mixture_times_noise_and_one_step_equivalence(model):
    new, _, _ = model
    new.eval()
    observed = torch.randn(1, 4, 4, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    aa = ci = torch.zeros(1, 4, dtype=torch.long)
    ri = torch.arange(4)[None]
    def sample(steps, mixture):
        new._sample_time_mixture = mixture
        rng = torch.Generator().manual_seed(17)
        result = new.sample(observed, mask, aa, ri, ci, torch.tensor([348.]), steps, 12, rng)
        return result, rng.get_state(), new.last_sample_diagnostics['observation_steps']
    endpoint, state, _ = sample(1, False)
    mixture, other_state, times = sample(1, True)
    assert all(torch.equal(x, y) for x, y in zip(endpoint, mixture))
    assert torch.equal(state, other_state) and times == [1]*12
    endpoint, state, _ = sample(4, False)
    mixture, other_state, times = sample(4, True)
    assert torch.equal(state, other_state)
    assert all(1 <= t <= 4 for t in times) and len(set(times)) > 1
    new._sample_time_mixture = False


def test_float64_norm_and_explicit_clip():
    kernel = torch.nn.Module()
    kernel.energy = torch.nn.Linear(2, 1, bias=False)
    kernel.energy.weight.grad = torch.full_like(kernel.energy.weight, 1e30)
    assert torch.isinf(kernel.energy.weight.grad.norm())
    audit = bounded.gradient_audit_and_clip(kernel)
    assert audit['gradient_elements_finite'] and audit['norm64_finite']
    assert audit['preclip_norm64'] > 1e30
    assert float(kernel.energy.weight.grad.double().norm()) == pytest.approx(1., abs=1e-6)
    kernel.energy.weight.grad[0, 0] = float('inf')
    audit = bounded.gradient_audit_and_clip(kernel)
    assert not audit['gradient_elements_finite']
    assert not audit['norm64_finite']


def test_gate_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / 'numerical_gate.json'
    monkeypatch.setattr(bounded, 'GATE_PATH', path)
    checkpoint = tmp_path / 'checkpoint_best.pt'
    manifest = tmp_path / 'val.jsonl'
    mode = dict(forward_finite=True, gradient_elements_finite=True,
                joint_backward_gain=1., endpoint_ca_rms_to_microsteps_max=.01)
    gate = dict(screening_passed=True, test_opened=False, optimizer_steps=0,
                protocol=dict(checkpoint=str(checkpoint), manifest=str(manifest),
                              horizons=[1], microsteps=[4, 8, 16], temperatures=[348.]),
                windows=[dict(horizon=1, temperature=348., modes={str(k): dict(mode) for k in [4, 8, 16]})])
    path.write_text(json.dumps(gate))
    assert bounded.verify_numerical_gate(path, checkpoint, manifest, [1], 8)['full_promotion'] is False
    with pytest.raises(ValueError):
        bounded.verify_numerical_gate(path, checkpoint, manifest, [64], 8)
    gate['windows'] = []
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        bounded.verify_numerical_gate(path, checkpoint, manifest, [1], 8)
    gate['screening_passed'] = False
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        bounded.verify_numerical_gate(path, checkpoint, manifest, [1], 8)


def test_component_gate_rejects_missing_content_provenance(tmp_path, monkeypatch):
    root = tmp_path
    monkeypatch.setattr(bounded, 'REPO', root)
    gate_path = root/'results/lasp_bounded_v3/numerical_gate.json'
    gate_path.parent.mkdir(parents=True)
    component = root/'component.pt'
    checkpoint = root/'checkpoint.pt'
    component.write_bytes(b'component')
    checkpoint.write_bytes(b'checkpoint')
    gate_path.write_text(json.dumps(dict(screening_passed=True, test_opened=False,
                                        optimizer_steps=0, protocol={})))
    with pytest.raises(ValueError, match='provenance mismatch'):
        bounded.verify_numerical_gate(gate_path, checkpoint, root/'val.jsonl', [1], 8, component)


def test_component_gate_accepts_new_location_and_detects_manifest_change(tmp_path):
    import hashlib
    component, checkpoint, manifest = [tmp_path/name for name in ('component.pt','init.pt','val.jsonl')]
    for path in (component, checkpoint, manifest):
        path.write_bytes(path.name.encode())
    mode = dict(forward_finite=True, gradient_elements_finite=True,
                joint_backward_gain=1., endpoint_ca_rms_to_microsteps_max=.01)
    gate = dict(screening_passed=True, test_opened=False, optimizer_steps=0,
                protocol=dict(component=str(component),checkpoint=str(checkpoint),manifest=str(manifest),
                              horizons=[1],microsteps=[4,8,16],temperatures=[348.]),
                windows=[dict(horizon=1,temperature=348.,modes={str(n):dict(mode) for n in (4,8,16)})])
    for key,path in (('component',component),('checkpoint',checkpoint),('manifest',manifest)):
        gate[key+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = tmp_path/'new_gate.json'
    path.write_text(json.dumps(gate))
    assert bounded.verify_numerical_gate(path,checkpoint,manifest,[1],8,component)['screening_only']
    manifest.write_bytes(b'changed')
    with pytest.raises(ValueError,match='provenance mismatch'):
        bounded.verify_numerical_gate(path,checkpoint,manifest,[1],8,component)


def test_train_scaled_objective_retains_common_validation(model, monkeypatch):
    from argparse import Namespace
    new, _, _ = model
    value = torch.tensor(2., requires_grad=True)
    def losses(*args):
        return dict(latent_es=value, coord_es=3*value, clash=.5*value,
                    loss=7*value)
    monkeypatch.setattr(base.CADynamics, 'losses', losses)
    terms = new.losses(None, 1, Namespace(target='endpoint', score_recipe='train_scaled_v2'), None)
    assert float(terms['training_loss'].detach()) == 14.
    assert float(terms['loss'].detach()) == pytest.approx(6.21)
    terms['training_loss'].backward()
    assert float(value.grad) == 7.


def test_calibration_eligibility_and_fixed_initial_regression():
    import copy
    initial = {'by_horizon': {str(h): dict(latent_es=1., coord_es=2., loss=2.1)
                               for h in (1, 4, 16, 64)}}
    row = copy.deepcopy(initial)
    assert not bounded.calibration_eligible(row, initial)
    row['by_horizon']['64']['latent_es'] = .97
    assert bounded.calibration_eligible(row, initial)
    row['by_horizon']['1']['coord_es'] = 2.05
    assert not bounded.calibration_eligible(row, initial)
    row['by_horizon']['64']['loss'] = 2.21
    assert bounded.calibration_regressed(row, initial)


def test_displacement_scale_masks_padding_and_matches_rms():
    h = torch.zeros(3, 2)
    x = torch.zeros(3, 3)
    target_h = torch.ones_like(h)*2
    target_x = torch.ones_like(x)*3
    target_h[-1] = float('nan')
    target_x[-1] = float('nan')
    hs, xs = base.displacement_scales(h, target_h, x, target_x, torch.tensor([True, True, False]))
    assert float(hs) == pytest.approx(2.)
    assert float(xs) == pytest.approx(3.)
    assert not hs.requires_grad


def test_training_scales_preserve_rng_and_equal_horizon_weights(monkeypatch):
    from protein_jepa_frame.training.lasp_run_state import rng_state
    class ScaleModel:
        training = True
        bundle = torch.nn.Identity()
        kernel = type('Kernel', (), {'cfg': type('Config', (), {'step_ps': 1.})()})()
        def eval(self): self.training = False
        def train(self, value): self.training = value
        def encode_clip(self, batch):
            coords = batch['coords']
            return coords[:, :, :, 1, :2], batch['residue_mask'].all(1)
    def batch(horizon):
        coords = torch.zeros(1, 4+horizon, 4, 4, 3)
        coords[0, -1, :, 1] = 2.
        return dict(coords=coords, residue_mask=torch.ones(1, 4+horizon, 4, dtype=torch.bool),
                    time_ps=torch.arange(4+horizon)[None].float(), temperature_k=torch.tensor([348.]),
                    protein_id=['p'], trajectory_id=['t'])
    monkeypatch.setattr(bounded, '_move_batch', lambda b, d: b)
    import evaluation.ensemble_metrics_ext as metrics
    monkeypatch.setattr(metrics, '_align_to', lambda x, a, m: x)
    before = rng_state()
    model = ScaleModel()
    scales = bounded.training_score_scales(model, {1: [batch(1)], 4: [batch(4), batch(4)]}, 'cpu')
    assert scales['window_count'] == 3 and scales['latent_weight'] == pytest.approx(1.)
    assert scales['latent'] == pytest.approx(2.)
    assert model.training
    assert torch.equal(before['torch'], rng_state()['torch'])


def test_cli_defaults_and_required_gate():
    args = bounded.parser().parse_args(['--init-checkpoint', 'best.pt', '--output', 'new',
                                       '--numerical-gate', 'gate.json'])
    assert args.steps == 2000 and args.val_every == 250
    assert args.n_samples == 12 and args.microsteps == 8
    assert args.horizons == [1, 4, 16, 64]
    assert args.grad_steps is None and args.stateless_data
    assert args.target == 'endpoint'
    assert (args.coord_weight, args.latent_weight, args.clash_weight) == (1., .1, .01)
    assert (args.rmsf_weight, args.contact_weight, args.mode_weight, args.distance_weight) == (0., 0., 0., 0.)
    with pytest.raises(SystemExit):
        bounded.parser().parse_args(['--init-checkpoint', 'best.pt', '--output', 'new'])
