"""CPU contracts for the registered target-only intervention and its runner."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from run_scripts import run_hs_target_pilot as pilot


class TinyBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 8)
        self.decoder = nn.Linear(8, 3, bias=False)
        self.requires_grad_(False)

    def encode(self, coords, mask, aa, ri, ci):
        return self.encoder(coords[..., 1, :]) * mask[..., None]

    def decode_ca(self, h, h0, aa, ri, ci, rotation, anchor, mask):
        return anchor + self.decoder(h - h0) * mask[..., None]


def tiny():
    torch.manual_seed(39)
    model = CADynamics.__new__(CADynamics)
    nn.Module.__init__(model)
    model.bundle = TinyBundle().double()
    model.kernel = StageB1b(StageB1bConfig(
        latent_dim=8, energy_heads=2, energy_hidden=16, energy_depth=1,
        bath_dim=2, step_ps=.1, diffusion_scale=.1, coupling_scale=.2,
        gamma_scale=.2, gamma_floor=.01, max_gamma_step=None, force_clip=None)).double()
    model.kernel.s0_mode = 'exact_spatial'
    model.collective_mobility = True
    model.history_conditioning = True
    model.noise_mix = .8
    model.noise_kernel = 'gaussian'
    model.noise_length = 8.
    model.noise_cutoff = 10.
    model.train_encoder = False
    model.target_encoder = None
    aa = torch.zeros(1, 6, dtype=torch.long)
    batch = dict(coords=torch.randn(1, 6, 6, 4, 3, dtype=torch.float64),
        residue_mask=torch.ones(1, 6, 6, dtype=torch.bool), aatype=aa,
        residue_index=torch.arange(6)[None], chain_id=aa,
        temperature_k=torch.tensor([320.], dtype=torch.float64))
    return model.train(), batch


def objective(model, batch, horizon, target):
    args = SimpleNamespace(**dict(pilot.EXPECTED, target=target))
    args.n_samples = 3
    args.grad_steps = horizon
    generator = torch.Generator().manual_seed(20261101)
    model.zero_grad(set_to_none=True)
    result = model.losses(batch, horizon, args, generator)
    result['loss'].backward()
    grads = {name: None if p.grad is None else p.grad.detach().clone()
             for name, p in model.named_parameters()}
    assert all(p.grad is None for p in model.bundle.parameters())
    assert any(value is not None and bool(value.abs().sum()) for value in grads.values())
    return {k: v.detach().clone() for k, v in result.items()}, grads, generator.get_state()


def equal_gradients(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if a[key] is None:
            assert b[key] is None
        else:
            assert torch.equal(a[key], b[key]), key


def test_horizon_one_targets_have_identical_effective_loss_gradients_and_rng():
    model, batch = tiny()
    batch['coords'] = batch['coords'][:, :5]
    batch['residue_mask'] = batch['residue_mask'][:, :5]
    a, ga, ra = objective(model, batch, 1, 'endpoint')
    b, gb, rb = objective(model, batch, 1, 'path_mixture')
    assert all(torch.equal(a[key], b[key]) for key in a)
    equal_gradients(ga, gb)
    assert torch.equal(ra, rb)


def test_intermediate_future_is_label_only_and_only_affects_path_target():
    model, batch = tiny()
    changed = copy.deepcopy(batch)
    # Observations, endpoint and all masks remain bitwise identical. Change
    # just the intermediate conformation, including non-rigid CA displacement.
    changed['coords'][:, 4, :, :, :] += torch.arange(6, dtype=torch.float64)[None, :, None, None] * 1.7
    endpoint, eg, er = objective(model, batch, 2, 'endpoint')
    endpoint2, eg2, er2 = objective(model, changed, 2, 'endpoint')
    assert all(torch.equal(endpoint[key], endpoint2[key]) for key in endpoint)
    equal_gradients(eg, eg2)
    assert torch.equal(er, er2)
    path, pg, pr = objective(model, batch, 2, 'path_mixture')
    path2, pg2, pr2 = objective(model, changed, 2, 'path_mixture')
    for name in ('latent_es', 'coord_es', 'contact', 'loss'):
        assert not torch.equal(path[name], path2[name]), name
    assert any(pg[key] is not None and not torch.equal(pg[key], pg2[key]) for key in pg)
    assert torch.equal(pr, pr2) and torch.equal(er, pr)
    # The unweighted RMSF diagnostic may differ; it is not an objective term.
    for terms in (endpoint, path):
        expected = terms['coord_es'] + .1 * terms['latent_es'] + 20 * terms['contact'] + 40 * terms['clash']
        torch.testing.assert_close(terms['loss'], expected, rtol=0, atol=0)


def test_commands_isolate_target_and_include_both_standard_suites():
    left, right = [pilot.commands(arm) for arm in pilot.ARMS]
    a, b = list(left['train']), list(right['train'])
    for flag in ('--target', '--output'):
        a[a.index(flag) + 1] = b[b.index(flag) + 1] = '<varies>'
    assert a == b
    assert '--train-encoder' not in a and '--resume' not in a
    for flag in ('--distance-weight', '--rmsf-weight', '--mode-weight', '--mode-distribution-weight'):
        assert a[a.index(flag) + 1] == '0'
    assert tuple(left) == pilot.JOBS
    assert left['suite'][left['suite'].index('--modes') + 1:left['suite'].index('--modes') + 4] == ['correct', 'wrong', 'none']
    assert left['suite'][left['suite'].index('--proteins-per-temp') + 1] == '12'
    assert left['matched8'][left['matched8'].index('--proteins-per-temp') + 1] == '8'
    for kind in ('suite', 'matched8'):
        assert '--t2a' in left[kind]
    pilot.assert_endpoint_default()
    with pytest.raises(ValueError, match='endpoint default'):
        pilot.assert_endpoint_default("kind=getattr(self,'prediction_kind','time_mixture')")


def checkpoint():
    component = dict(checkpoint_kind='ca_component_probe', decoder_mode='geometric_ca',
        calibration_args=dict(bond_projection_steps=32, joint=True),
        encoder={'weight': torch.tensor([1.])}, delta_ca_decoder={'weight': torch.tensor([2.])})
    args = dict(pilot.EXPECTED, target='endpoint', output=str(pilot.ARTIFACTS / 'endpoint'),
                component=str(pilot.INITIAL), init_kernel=str(pilot.INITIAL))
    return dict(checkpoint_kind='lasp_ca_dynamics', step=500, args=args,
                component_checkpoint=component, kernel_config={'latent_dim': 128},
                kernel={'weight': torch.tensor([3.])}, target_encoder=None)


@pytest.mark.parametrize('mutation', ['step', 'target', 'encoder', 'decoder', 'projection', 'teacher'])
def test_checkpoint_rejects_wrong_selection_or_changed_frozen_component(mutation):
    ck = checkpoint()
    identity = pilot.component_identity(ck['component_checkpoint'])
    assert pilot.audit_checkpoint(ck, 'endpoint', identity)['frozen_component_verified']
    if mutation == 'step':
        ck['step'] = 400
    elif mutation == 'target':
        ck['args']['target'] = 'path_mixture'
    elif mutation in ('encoder', 'decoder'):
        ck['component_checkpoint']['encoder' if mutation == 'encoder' else 'delta_ca_decoder']['weight'] += 1
    elif mutation == 'projection':
        ck['component_checkpoint']['calibration_args']['bond_projection_steps'] = 0
    else:
        ck['target_encoder'] = {'weight': torch.zeros(1)}
    with pytest.raises(ValueError):
        pilot.audit_checkpoint(ck, 'endpoint', identity)


def test_input_changes_and_exclusive_outputs_fail_closed(tmp_path):
    f = tmp_path / 'file'
    f.write_text('initial')
    hashes = {str(f): pilot.sha256_file(f)}
    assert pilot.verify_files(hashes) == hashes
    f.write_text('changed')
    with pytest.raises(ValueError, match='immutable'):
        pilot.verify_files(hashes)
    output = tmp_path / 'out.json'
    pilot.write_new(output, {'original': True})
    with pytest.raises(FileExistsError):
        pilot.write_new(output, {'overwritten': True})
    assert json.loads(output.read_text()) == {'original': True}


def test_failed_subprocess_retains_actual_pid_exit_and_does_not_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'RESULTS', tmp_path)
    monkeypatch.setattr(pilot, 'ARTIFACTS', tmp_path / 'artifacts')
    calls = []
    class FailedProcess:
        pid = 123456
        def __init__(self, command, **kwargs):
            calls.append(command)
        def wait(self):
            return 9
    monkeypatch.setattr(pilot.subprocess, 'Popen', FailedProcess)
    protocol = dict(input_hashes={}, source_hashes={}, per_arm_commands={'endpoint': {'train': ['fixed-command']}})
    with pytest.raises(RuntimeError, match='exited 9'):
        pilot.run_arm('endpoint', protocol)
    saved = json.loads((tmp_path / 'endpoint.execution.json').read_text())
    assert saved['status'] == 'failed'
    assert saved['jobs'][0]['pid'] == 123456 and saved['jobs'][0]['returncode'] == 9
    assert len(calls) == 1 and len(saved['jobs']) == 1
    with pytest.raises(FileExistsError):
        pilot.run_arm('endpoint', protocol)
