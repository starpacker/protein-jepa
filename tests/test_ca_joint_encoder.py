"""Causal and gradient contracts for autonomous H+S representation feedback."""
from types import SimpleNamespace
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)

    def forward(self, coords, mask, aa, ri, ci, token_mask=None):
        return self.projection(coords[..., 1, :]) * mask[..., None]


class TinyBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.decoder = nn.Linear(8, 3, bias=False)
        self.requires_grad_(False)

    def encode(self, coords, mask, aa, ri, ci):
        return self.encoder(coords[:, None], mask[:, None], aa, ri, ci)[:, 0]

    def decode_ca(self, h, h0, aa, ri, ci, rotation, anchor, mask):
        return anchor + self.decoder(h - h0) * mask[..., None]


def fixture():
    torch.manual_seed(39)
    model = CADynamics.__new__(CADynamics)
    nn.Module.__init__(model)
    model.bundle = TinyBundle().double()
    model.kernel = StageB1b(StageB1bConfig(
        latent_dim=8, energy_heads=2, energy_hidden=16, energy_depth=1,
        bath_dim=2, step_ps=.1, diffusion_scale=.1, coupling_scale=.2,
        gamma_scale=.2, gamma_floor=.01, max_gamma_step=None,
        force_clip=None)).double()
    model.kernel.s0_mode = 'exact'
    model.collective_mobility = True
    model.history_conditioning = True
    model.noise_mix = 0.
    model.train_encoder = False
    model.target_encoder = None
    coords = torch.randn(1, 6, 6, 4, 3, dtype=torch.float64)
    aa = torch.zeros(1, 6, dtype=torch.long)
    batch = dict(coords=coords, residue_mask=torch.ones(1, 6, 6, dtype=torch.bool),
                 aatype=aa, residue_index=torch.arange(6)[None], chain_id=aa,
                 temperature_k=torch.tensor([320.], dtype=torch.float64))
    return model, batch


def predict(model, batch):
    return model.sample(batch['coords'][:, :4], batch['residue_mask'][:, :4].all(1),
                        *[batch[k] for k in ('aatype', 'residue_index', 'chain_id')],
                        batch['temperature_k'], 2, 3,
                        torch.Generator().manual_seed(12), grad_steps=2)


def test_joint_forecast_encoder_gradient_matches_finite_difference():
    model, batch = fixture()
    model.enable_encoder_training()
    model.train()
    # No teacher/reconstruction objective is involved in this check: the only
    # encoder signal comes through the autonomous H+S prediction.
    def score():
        ca, _ = predict(model, batch)
        return (ca - batch['coords'][:, -1, :, 1][:, None]).square().mean()
    value = score()
    weight = model.bundle.encoder.projection.weight
    value.backward()
    assert torch.isfinite(weight.grad).all() and weight.grad.norm() > 1e-7
    assert all(p.grad is None for p in model.bundle.decoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    direction = torch.randn_like(weight)
    direction /= direction.norm()
    analytic = (weight.grad * direction).sum()
    original = weight.detach().clone()
    eps = 1e-5
    with torch.no_grad():
        weight.copy_(original + eps * direction)
    plus = score().detach()
    with torch.no_grad():
        weight.copy_(original - eps * direction)
    minus = score().detach()
    with torch.no_grad():
        weight.copy_(original)
    torch.testing.assert_close(analytic, (plus - minus) / (2 * eps), atol=2e-7, rtol=2e-4)


def test_fixed_teacher_targets_and_observed_retention_ignore_future_changes():
    model, batch = fixture()
    model.enable_encoder_training()
    before, _ = model.encode_clip(batch)
    with torch.no_grad():
        model.bundle.encoder.projection.weight.mul_(1.1)
    after, _ = model.encode_clip(batch)
    torch.testing.assert_close(before, after)
    assert not after.requires_grad
    absolute, delta = model.representation_retention(batch)
    assert absolute > 0 and delta > 0
    changed = {**batch, 'coords': batch['coords'].clone()}
    changed['coords'][:, 4:] += 1000
    altered = model.representation_retention(changed)
    torch.testing.assert_close(absolute, altered[0])
    torch.testing.assert_close(delta, altered[1])
    torch.testing.assert_close(predict(model, batch)[0], predict(model, changed)[0])


def test_joint_switch_preserves_values_and_checkpoint_saves_live_encoder():
    model, batch = fixture()
    frozen, _ = predict(model, batch)
    model.enable_encoder_training()
    model.train()
    joint, _ = predict(model, batch)
    torch.testing.assert_close(joint, frozen)
    source = dict(encoder={k: v.clone() for k, v in model.target_encoder.state_dict().items()},
                  checkpoint_kind='ca_component_probe', sentinel='preserved')
    with torch.no_grad():
        model.bundle.encoder.projection.bias.add_(.25)
    saved = model.component_checkpoint(source)
    assert saved['sentinel'] == source['sentinel']
    for name, parameter in model.bundle.encoder.state_dict().items():
        torch.testing.assert_close(saved['encoder'][name], parameter)
    assert not torch.equal(saved['encoder']['projection.bias'], source['encoder']['projection.bias'])
    # Snapshots must not be aliases of subsequently updated parameters.
    prior = saved['encoder']['projection.bias'].clone()
    with torch.no_grad():
        model.bundle.encoder.projection.bias.add_(.5)
    torch.testing.assert_close(saved['encoder']['projection.bias'], prior)
    assert not model.target_encoder.training and not model.bundle.decoder.training


def test_joint_loss_updates_encoder_but_keeps_future_target_detached():
    model, batch = fixture()
    model.enable_encoder_training()
    args = SimpleNamespace(target='endpoint', n_samples=3, grad_steps=2,
                           latent_weight=.1, contact_weight=1., rmsf_weight=1.,
                           distance_weight=0., clash_weight=0., mode_weight=0.,
                           contact_objective='brier', teacher_weight=1., teacher_delta_weight=.1)
    terms = model.losses(batch, 2, args, torch.Generator().manual_seed(51))
    terms['loss'].backward()
    assert all(torch.isfinite(value) for value in terms.values())
    assert model.bundle.encoder.projection.weight.grad.norm() > 0
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert all(p.grad is None for p in model.bundle.decoder.parameters())


def test_future_validity_does_not_change_sampler_inputs():
    model, batch = fixture()
    model.enable_encoder_training()
    seen = []
    sample = model.sample
    def capture(observed, mask, *args, **kwargs):
        result = sample(observed, mask, *args, **kwargs)
        seen.append((mask.clone(), result[0].detach().clone()))
        return result
    model.sample = capture
    args = SimpleNamespace(target='endpoint', n_samples=3, grad_steps=2,
                           latent_weight=.1, contact_weight=1., rmsf_weight=1.,
                           distance_weight=0., clash_weight=0., mode_weight=0.,
                           contact_objective='brier', teacher_weight=1., teacher_delta_weight=.1)
    model.losses(batch, 2, args, torch.Generator().manual_seed(51))
    changed = {**batch, 'residue_mask': batch['residue_mask'].clone()}
    changed['residue_mask'][:, 4:, -1] = False
    model.losses(changed, 2, args, torch.Generator().manual_seed(51))
    torch.testing.assert_close(seen[0][0], seen[1][0])
    torch.testing.assert_close(seen[0][1], seen[1][1])


def test_warm_start_restores_embedded_encoder_and_decoder(tmp_path, monkeypatch):
    import protein_jepa_frame.training.train_lasp_ca_dynamics as training
    model, _ = fixture()
    component = dict(encoder=model.bundle.encoder.state_dict(),
                     decoder=model.bundle.decoder.state_dict(), marker='paired')
    checkpoint = tmp_path / 'dynamics.pt'
    torch.save(dict(kernel_config=asdict(model.kernel.cfg), kernel=model.kernel.state_dict(),
                    component_checkpoint=component, args=dict(s0_mode='exact', collective_mobility=True)),
               checkpoint)
    def load_embedded(path, device):
        source = torch.load(path, weights_only=False)
        bundle = TinyBundle().double()
        bundle.encoder.load_state_dict(source['encoder'])
        bundle.decoder.load_state_dict(source['decoder'])
        return bundle
    monkeypatch.setattr(training, 'CAComponentBundle', load_embedded)
    restored = CADynamics(str(tmp_path / 'missing_external_component.pt'), 'cpu', str(checkpoint))
    for name, tensor in model.bundle.state_dict().items():
        torch.testing.assert_close(restored.bundle.state_dict()[name], tensor)
    assert restored.component_source['marker'] == 'paired'
    assert restored.kernel.s0_mode == 'exact'


def test_spatial_posterior_shared_across_ensemble_matches_repeated_filter_gradients(monkeypatch):
    """One filter per window preserves full-covariance draws and their gradients."""
    model, original = fixture()
    model.enable_encoder_training()
    model.kernel.s0_mode = 'exact_spatial'
    model.noise_mix = .6
    # Different validity layouts require per-window packing inside the filter.
    batch = {key: value.repeat(2, *([1] * (value.ndim - 1))) for key, value in original.items()}
    batch['coords'][1] += .2
    batch['residue_mask'][1, :, -1] = False
    observed = batch['coords'][:, :4]
    mask = batch['residue_mask'][:, :4].all(1)
    aa, ri, ci = [batch[key] for key in ('aatype', 'residue_index', 'chain_id')]
    temp = batch['temperature_k']
    b, n, length = observed.shape[:3]
    members = 3
    covariance = .4 * torch.eye(length, dtype=observed.dtype) + .6 * torch.ones(length, length, dtype=observed.dtype)
    factor = torch.linalg.cholesky(covariance).expand(b, -1, -1)
    monkeypatch.setattr(model, 'anchor_kernel', lambda anchor, mask: factor)
    filtered_batches = []
    original_filter = model.kernel.exact_spatial_bath_posterior
    def capture(history, *args, **kwargs):
        filtered_batches.append(history.shape[0])
        return original_filter(history, *args, **kwargs)
    monkeypatch.setattr(model.kernel, 'exact_spatial_bath_posterior', capture)
    generator = torch.Generator().manual_seed(73)
    _, efficient = model.sample(observed, mask, aa, ri, ci, temp, 2, members, generator, grad_steps=2)
    final_rng = generator.get_state()
    assert filtered_batches == [b]
    parameters = (model.bundle.encoder.projection.weight, model.kernel.energy.in_proj.weight)
    efficient_gradients = torch.autograd.grad(efficient.square().mean(), parameters)

    # Independent reference: let rollout filter each repeated ensemble member.
    repeat = lambda x: x[:, None].expand(b, members, *x.shape[1:]).reshape(b * members, *x.shape[1:])
    expand = lambda x: x[:, None].expand(b, n, *x.shape[1:]).reshape(b * n, *x.shape[1:])
    history = model.bundle.encode(observed.reshape(b * n, length, 4, 3),
                                  expand(mask), expand(aa), expand(ri), expand(ci)).reshape(b, n, length, -1)
    anchor = model.bundle.encode(observed[:, -1], mask, aa, ri, ci)
    generator = torch.Generator().manual_seed(73)
    with sdpa_kernel(SDPBackend.MATH):
        trajectory = model.kernel.rollout(repeat(anchor), repeat(mask), repeat(temp), 2,
            H_hist=repeat(history), dt_hist=torch.full((b * members, n), model.kernel.cfg.step_ps),
            seq_feature=repeat(anchor), generator=generator, detach=False, grad_steps=2,
            noise_factor=repeat(factor), collective_mobility=True)
    naive = trajectory[:, -1].reshape_as(efficient)
    naive_gradients = torch.autograd.grad(naive.square().mean(), parameters)
    assert filtered_batches == [b, b * members]
    assert torch.equal(final_rng, generator.get_state())
    torch.testing.assert_close(efficient, naive, atol=1e-10, rtol=1e-9)
    for actual, expected in zip(efficient_gradients, naive_gradients):
        assert torch.isfinite(actual).all() and actual.norm() > 0
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-8)


def readout_args(**overrides):
    values=dict(target='endpoint',n_samples=3,grad_steps=2,latent_weight=.1,
                contact_weight=1.,rmsf_weight=1.,distance_weight=0.,clash_weight=0.,
                mode_weight=0.,contact_objective='brier',teacher_weight=1.,teacher_delta_weight=.1)
    return SimpleNamespace(**{**values,**overrides})


def test_observed_readout_matches_diagnostic_and_ignores_future_coords_and_mask():
    from evaluation.geometry import kabsch_align
    from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
    from scripts.diagnose_ca_readout_retention import losses as diagnostic_losses
    model,batch=fixture()
    model.enable_encoder_training()
    batch['residue_mask'][:,:,0]=False
    observed=batch['coords'][0,:4]
    future=batch['coords'][0,4:]
    mask=batch['residue_mask'][0,:4].all(0)
    rotation,anchor=_extract_backbone_frames(observed[-1:])
    metadata={key:batch[key][0] for key in ('aatype','residue_index','chain_id')}
    with torch.no_grad():
        teacher=model.bundle.encode(observed,mask[None].expand(4,-1),
            *[metadata[key][None].expand(4,-1) for key in ('aatype','residue_index','chain_id')])
    group=dict(observed=observed,future=future,mask=mask,rotation=rotation,anchor=anchor,
        teacher_h=teacher,baseline_readout=torch.zeros(5,6,3,dtype=observed.dtype),**metadata,
        aligned_truth=torch.stack([kabsch_align(frame[:,1],anchor[0],mask)
                                   for frame in torch.cat((observed[:3],future))]))
    expected=diagnostic_losses(model.bundle,group,(1.,.1))['observed_readout_mse']
    actual=model.observed_readout_retention(batch)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    changed={**batch,'coords':batch['coords'].clone(),'residue_mask':batch['residue_mask'].clone()}
    changed['coords'][:,4:]=float('nan')
    changed['residue_mask'][:,4:]=False
    torch.testing.assert_close(actual,model.observed_readout_retention(changed),atol=0,rtol=0)
    actual.backward()
    assert model.bundle.encoder.projection.weight.grad.norm()>0
    assert all(p.grad is None and not p.requires_grad for p in model.bundle.decoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.target_encoder.parameters())
    assert all(p.grad is None for p in model.kernel.parameters())


def test_observed_readout_uses_bundle_projection_and_equal_clip_means(monkeypatch):
    from protein_jepa_frame.models.ca_geometry import project_ca_bonds
    model,original=fixture()
    model.enable_encoder_training()
    batch={key:value.repeat(2,*([1]*(value.ndim-1))) for key,value in original.items()}
    batch['coords'][1,0,2,1,0]+=2
    batch['residue_mask'][1,:,-1]=False
    decode=model.bundle.decode_ca
    calls=[]
    def projected(h,h0,aa,ri,ci,rotation,anchor,mask):
        calls.append(h.shape[0])
        return project_ca_bonds(decode(h,h0,aa,ri,ci,rotation,anchor,mask),anchor,mask,ri,ci,2)
    monkeypatch.setattr(model.bundle,'decode_ca',projected)
    together=model.observed_readout_retention(batch)
    separate=[model.observed_readout_retention({key:value[i:i+1] for key,value in batch.items()})
              for i in range(2)]
    assert calls==[6,3,3]
    torch.testing.assert_close(together,torch.stack(separate).mean())
    together.backward()
    assert torch.isfinite(model.bundle.encoder.projection.weight.grad).all()
    assert model.bundle.encoder.projection.weight.grad.norm()>0


def test_readout_zero_preserves_legacy_terms_and_rng_and_positive_adds_only_weighted_term(monkeypatch):
    model,batch=fixture()
    model.enable_encoder_training()
    args=readout_args()
    state=torch.get_rng_state()
    generator=torch.Generator().manual_seed(51)
    legacy=model.losses(batch,2,args,generator)
    final_rng=generator.get_state()
    global_final=torch.get_rng_state()
    readout=model.observed_readout_retention
    def forbidden(*_args):
        raise AssertionError('zero weight must not evaluate the readout term')
    monkeypatch.setattr(model,'observed_readout_retention',forbidden)
    torch.set_rng_state(state)
    generator=torch.Generator().manual_seed(51)
    disabled=model.losses(batch,2,readout_args(observed_readout_weight=0.),generator)
    assert legacy.keys()==disabled.keys()
    for key in legacy:
        torch.testing.assert_close(legacy[key],disabled[key],atol=0,rtol=0)
    assert torch.equal(final_rng,generator.get_state())
    assert torch.equal(global_final,torch.get_rng_state())
    monkeypatch.setattr(model,'observed_readout_retention',readout)
    torch.set_rng_state(state)
    generator=torch.Generator().manual_seed(51)
    enabled=model.losses(batch,2,readout_args(observed_readout_weight=.25),generator)
    assert set(enabled)==set(legacy)|{'observed_readout_mse'}
    for key in legacy.keys()-{'loss'}:
        torch.testing.assert_close(legacy[key],enabled[key],atol=0,rtol=0)
    torch.testing.assert_close(enabled['loss'],legacy['loss']+.25*enabled['observed_readout_mse'],atol=0,rtol=0)
    assert torch.equal(final_rng,generator.get_state())
    assert torch.equal(global_final,torch.get_rng_state())


def test_readout_weight_validation_and_legacy_resume_contract(monkeypatch):
    import protein_jepa_frame.training.train_lasp_ca_dynamics as training
    model,batch=fixture()
    for weight in (-.1,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='finite and nonnegative'):
            model.losses(batch,2,readout_args(observed_readout_weight=weight),None)
    with pytest.raises(ValueError,match='requires --train-encoder'):
        model.losses(batch,2,readout_args(observed_readout_weight=.1),None)
    # CLI rejects an ineffective positive weight before accessing data or CUDA.
    monkeypatch.setattr('sys.argv',['trainer','--component','unused','--output','unused',
                                  '--observed-readout-weight','.1'])
    with pytest.raises(ValueError,match='requires --train-encoder'):
        training.main()
    legacy=dict(train_encoder=True,noise_kernel=None,noise_cutoff=None,resume=False)
    args=SimpleNamespace(train_encoder=True,noise_kernel='gaussian',noise_cutoff=10.,
                         resume=True,observed_readout_weight=0.)
    training._validate_resume_arguments(args,legacy)
    args.observed_readout_weight=.25
    with pytest.raises(ValueError,match='resume argument differs: observed_readout_weight'):
        training._validate_resume_arguments(args,legacy)
    training._validate_resume_arguments(args,{**legacy,'observed_readout_weight':.25})
    args.observed_readout_weight=0.
    with pytest.raises(ValueError,match='resume argument differs: observed_readout_weight'):
        training._validate_resume_arguments(args,{**legacy,'observed_readout_weight':.25})
