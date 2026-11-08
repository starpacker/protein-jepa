"""Synthetic sampler/gradient contracts; no actual checkpoint or trajectory loads."""
import copy

import pytest
import torch

from protein_jepa_frame.training import contact_corrected_path_objective_v1 as corrected
from protein_jepa_frame.training import ordered_path_objective_v1 as frozen
from test_hs_target_pilot import equal_gradients
from test_ordered_path_objective_v1 import setup, gradient


def fixture(horizon=3, batch_size=1):
    model,batch,args = setup(horizon,batch_size)
    args.n_samples = 12
    args.coord_objective = 'marginal_path'
    args.contact_estimator = 'plugin'
    batch['coords'] = batch['coords']*3
    return model,batch,args


def rng(seed=88):
    return torch.Generator().manual_seed(seed)


def geometric_p32(model, monkeypatch):
    from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
    from protein_jepa_frame.models.ca_geometry import project_ca_bonds
    model.bundle.decoder = GeometricDeltaCADecoder(latent_dim=8,hidden=8,depth=1).double().requires_grad_(False).eval()
    def decode(h,h0,aa,ri,ci,rotation,anchor,mask):
        raw = model.bundle.decoder(h,h0,aa,ri,ci,rotation,anchor,mask)
        return project_ca_bonds(raw,anchor,mask,ri,ci,iterations=32)
    monkeypatch.setattr(model.bundle,'decode_ca',decode)


@pytest.mark.parametrize('horizon',[1,3])
@pytest.mark.parametrize('batch_size',[1,2])
def test_plugin_control_old_values_gradients_and_rng_bitwise(horizon,batch_size):
    model,batch,args = fixture(horizon,batch_size)
    a_rng,b_rng = rng(),rng()
    global_before = torch.get_rng_state().clone()
    old = frozen.ordered_path_losses(model,batch,horizon,args,a_rng)
    old_grad = gradient(model,old['loss'])
    global_after_old = torch.get_rng_state().clone()
    new = corrected.contact_corrected_path_losses(model,batch,horizon,args,b_rng)
    new_grad = gradient(model,new['loss'])
    assert all(torch.equal(old[key],new[key]) for key in old)
    assert new['loss'] is new['loss_plugin'] and new['contact'] is new['contact_plugin']
    equal_gradients(old_grad,new_grad)
    assert torch.equal(a_rng.get_state(),b_rng.get_state())
    assert torch.equal(global_before,global_after_old) and torch.equal(global_before,torch.get_rng_state())
    assert all(value.ndim == 0 for value in new.values())


@pytest.mark.parametrize('checkpoint_decoder',[False,True])
def test_actual_geometric_p32_control_and_corrected_full_gradients(checkpoint_decoder,monkeypatch):
    model,batch,args = fixture(3)
    args.checkpoint_decoder = checkpoint_decoder
    geometric_p32(model,monkeypatch)
    a_rng,b_rng,c_rng = rng(),rng(),rng()
    old = frozen.ordered_path_losses(model,batch,3,args,a_rng)
    old_grad = gradient(model,old['loss'])
    control = corrected.contact_corrected_path_losses(model,batch,3,args,b_rng)
    equal_gradients(old_grad,gradient(model,control['loss']))
    assert all(torch.equal(old[key],control[key]) for key in old)
    args.contact_estimator = 'population_u'
    arm = corrected.contact_corrected_path_losses(model,batch,3,args,c_rng)
    assert torch.equal(arm['loss_plugin'],control['loss'])
    assert torch.equal(arm['loss'],arm['loss_plugin']-20*arm['contact_variance'])
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    g_old = torch.autograd.grad(arm['loss_plugin'],parameters,retain_graph=True,allow_unused=True)
    g_v = torch.autograd.grad(20*arm['contact_variance'],parameters,retain_graph=True,allow_unused=True)
    g_new = torch.autograd.grad(arm['loss'],parameters,allow_unused=True)
    assert any(g is not None and bool(g.abs().sum()>0) for g in g_v)
    for a,v,b in zip(g_old,g_v,g_new):
        if a is None:
            assert v is None and b is None
        else:
            torch.testing.assert_close(b,a-(torch.zeros_like(a) if v is None else v),rtol=2e-10,atol=2e-11)
    assert all(p.grad is None for p in model.bundle.parameters())
    assert torch.equal(a_rng.get_state(),b_rng.get_state()) and torch.equal(b_rng.get_state(),c_rng.get_state())


def test_variance_pair_mask_batch_average_and_member_gradients():
    generator = rng(32)
    x = (torch.randn(2,12,6,3,generator=generator,dtype=torch.float64)*4).requires_grad_()
    mask = torch.tensor([[True]*5+[False],[True]*6])
    ri = torch.tensor([[0,1,2,7,8,9],[0,1,2,3,4,5]])
    ci = torch.tensor([[0,0,0,0,1,1],[0,0,1,1,1,1]])
    actual = corrected.endpoint_contact_variance(x,mask,ri,ci)
    references = []
    for b in range(2):
        per_pair = []
        for i in range(6):
            for j in range(i+1,6):
                if mask[b,i] and mask[b,j] and (abs(int(ri[b,i]-ri[b,j]))>=3 or ci[b,i]!=ci[b,j]):
                    z = torch.sigmoid((8-(x[b,:,i]-x[b,:,j]).norm(dim=-1))/.5)
                    # Ordered pairwise squared differences: variance/M.
                    per_pair.append(((z[:,None]-z[None,:]).square().sum())/(2*12*12*11))
        references.append(torch.stack(per_pair).mean())
    expected = torch.stack(references).mean()
    torch.testing.assert_close(actual,expected,atol=1e-15,rtol=1e-13)
    ga = torch.autograd.grad(actual,x,retain_graph=True)[0]
    ge = torch.autograd.grad(expected,x)[0]
    torch.testing.assert_close(ga,ge,atol=1e-15,rtol=1e-12)
    assert bool((ga.square().sum((-1,-2))>0).all())
    assert bool((ga[0,:,~mask[0]]==0).all())


def test_one_continuous_sample_future_independence_and_member_gradient(monkeypatch):
    model,batch,args = fixture(3)
    args.contact_estimator = 'population_u'
    changed = copy.deepcopy(batch)
    changed['coords'][:,4:] += torch.arange(6,dtype=torch.float64)[None,None,:,None,None]*.8
    calls = dict(rollout=0,posterior=0)
    for name,label in [('rollout','rollout'),('initial_bath_state','posterior')]:
        original = getattr(model.kernel,name)
        def count(*a,_original=original,_label=label,**kw):
            calls[_label] += 1
            return _original(*a,**kw)
        monkeypatch.setattr(model.kernel,name,count)
    captures = []
    original = corrected.sample_paths
    def capture(*a,**kw):
        assert a[1].shape[1]==4 and kw['observed_time_ps'].shape[1]==4
        value = original(*a,**kw)
        value['endpoint_ca'].retain_grad()
        captures.append(value)
        return value
    monkeypatch.setattr(corrected,'sample_paths',capture)
    a_rng,b_rng = rng(),rng()
    a = corrected.contact_corrected_path_losses(model,batch,3,args,a_rng)
    b = corrected.contact_corrected_path_losses(model,changed,3,args,b_rng)
    assert calls == dict(rollout=2,posterior=2)
    assert torch.equal(captures[0]['ca_paths'],captures[1]['ca_paths'])
    assert torch.equal(captures[0]['h_paths'],captures[1]['h_paths'])
    assert torch.equal(a['contact_variance'],b['contact_variance'])
    assert not torch.equal(a['loss'],b['loss'])
    assert torch.equal(a_rng.get_state(),b_rng.get_state())
    a['contact_variance'].backward()
    member_norms = captures[0]['endpoint_ca'].grad.square().sum((-1,-2))
    assert bool((member_norms>0).all())


@pytest.mark.parametrize('arm',corrected.CONTACT_ESTIMATORS)
def test_masked_nan_padding_extra_future_support_and_gaps(arm):
    model,batch,args = fixture(3)
    args.contact_estimator = arm
    batch['residue_mask'][:,:,-1] = False
    batch['coords'][:,:,-1] = float('nan')
    batch['residue_index'][0] = torch.tensor([0,1,5,6,9,10])
    batch['chain_id'][0,-2:] = 1
    other = copy.deepcopy(batch)
    other['residue_mask'][:,4:,-1] = True
    other['coords'][:,4:,-1] = 1e6
    a_rng,b_rng = rng(),rng()
    a = corrected.contact_corrected_path_losses(model,batch,3,args,a_rng)
    ga = gradient(model,a['loss'])
    b = corrected.contact_corrected_path_losses(model,other,3,args,b_rng)
    gb = gradient(model,b['loss'])
    assert all(torch.equal(a[k],b[k]) for k in a)
    equal_gradients(ga,gb)
    assert torch.equal(a_rng.get_state(),b_rng.get_state())


@pytest.mark.parametrize('change',[('contact_estimator','other'),('coord_objective','joint_path'),
    ('n_samples',3),('contact_weight',19.),('path_weight',1.),('grad_steps',1)])
def test_invalid_arm_or_scope_rejected_before_rng(change):
    model,batch,args = fixture(3)
    setattr(args,*change)
    generator = rng()
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        corrected.contact_corrected_path_losses(model,batch,3,args,generator)
    assert torch.equal(before,generator.get_state())
