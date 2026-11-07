"""Production sampler contracts on synthetic inputs; no checkpoint/data reads."""
import copy

import pytest
import torch

from protein_jepa_frame.training import ordered_path_objective_v1 as frozen
from protein_jepa_frame.training import physical_distance_augmented_objective_v2 as augmented
from test_hs_target_pilot import equal_gradients
from test_ordered_path_objective_v1 import setup, gradient


def fixture(horizon=3, batch_size=1):
    model,batch,args = setup(horizon,batch_size)
    # The reusable tiny kernel uses 0.1 time units. Express its same hD/hGamma
    # products on this objective's physical 1000 ps grid, without changing C.
    rate_scale = model.kernel.cfg.step_ps/1000.
    model.kernel.cfg.step_ps = 1000.
    for name in ('diffusion_scale','gamma_scale','gamma_floor'):
        setattr(model.kernel.bath,name,getattr(model.kernel.bath,name)*rate_scale)
        setattr(model.kernel.cfg,name,getattr(model.kernel.cfg,name)*rate_scale)
    batch['time_ps'] = torch.arange(horizon+4,dtype=torch.float64)[None].expand(batch_size,-1)*1000.
    args.n_samples = 12
    args.coord_objective = 'marginal_path'
    args.distance_path_objective = 'absolute'
    args.distance_path_weight = 1.
    batch['coords'] = batch['coords']*3
    ri,ci=batch['residue_index'],batch['chain_id']
    batch['geometry_row_ids']=torch.arange(ri.shape[1])[None].expand_as(ri).clone()
    batch['physical_pair_eligibility']=(ci[:,:,None]!=ci[:,None,:])|((ri[:,:,None]-ri[:,None,:]).abs()>=3)
    return model,batch,args


def rng(seed=71):
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
def test_base_old_values_full_gradients_and_rng_bitwise(horizon,batch_size):
    model,batch,args = fixture(horizon,batch_size)
    old_rng,new_rng = rng(),rng()
    global_before = torch.get_rng_state().clone()
    old = frozen.ordered_path_losses(model,batch,horizon,args,old_rng)
    old_grad = gradient(model,old['loss'])
    new = augmented.physical_distance_augmented_path_losses(model,batch,horizon,args,new_rng)
    new_grad = gradient(model,new['loss_base'])
    assert all(torch.equal(value,new['loss_base' if key=='loss' else key]) for key,value in old.items())
    equal_gradients(old_grad,new_grad)
    assert torch.equal(old_rng.get_state(),new_rng.get_state())
    assert torch.equal(global_before,torch.get_rng_state())
    assert new['loss'] is new['loss_absolute']
    assert all(value.ndim==0 and bool(torch.isfinite(value)) for value in new.values())


@pytest.mark.parametrize('checkpoint_decoder',[False,True])
def test_real_geometric_p32_same_forward_additive_gradients(checkpoint_decoder,monkeypatch):
    model,batch,args = fixture(3)
    args.checkpoint_decoder = checkpoint_decoder
    geometric_p32(model,monkeypatch)
    old_rng,new_rng = rng(),rng()
    old = frozen.ordered_path_losses(model,batch,3,args,old_rng)
    old_grad = gradient(model,old['loss'])
    model.zero_grad(set_to_none=True)
    terms = augmented.physical_distance_augmented_path_losses(model,batch,3,args,new_rng)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    gradients = {name:torch.autograd.grad(terms[name],parameters,retain_graph=True,allow_unused=True)
                 for name in ('loss_base','distance_absolute_es','distance_increment_es','loss_absolute','loss_increment')}
    base_by_name = {name:None for name,_ in model.named_parameters()}
    base_by_name.update({name:g for (name,p),g in zip(
        [(n,p) for n,p in model.named_parameters() if p.requires_grad],gradients['loss_base'])})
    equal_gradients(old_grad,base_by_name)
    for objective in augmented.DISTANCE_OBJECTIVES:
        auxiliary = gradients['distance_'+objective+'_es']
        total = gradients['loss_'+objective]
        assert any(g is not None and bool(g.abs().sum()>0) for g in auxiliary)
        for base,extra,actual in zip(gradients['loss_base'],auxiliary,total):
            if base is None and extra is None:
                assert actual is None
            else:
                expected = (torch.zeros_like(actual) if base is None else base)+(torch.zeros_like(actual) if extra is None else extra)
                torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-11)
    assert any(a is not None and b is not None and not torch.allclose(a,b,rtol=1e-5,atol=1e-7)
               for a,b in zip(gradients['distance_absolute_es'],gradients['distance_increment_es']))
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(old_rng.get_state(),new_rng.get_state())


@pytest.mark.parametrize('checkpoint_decoder',[False,True])
def test_h1_arm_loss_and_full_kernel_gradient_identity(checkpoint_decoder,monkeypatch):
    model,batch,args = fixture(1,2)
    args.checkpoint_decoder = checkpoint_decoder
    geometric_p32(model,monkeypatch)
    a_rng,b_rng = rng(),rng()
    absolute = augmented.physical_distance_augmented_path_losses(model,batch,1,args,a_rng)
    ga = gradient(model,absolute['loss'])
    args.distance_path_objective = 'increment'
    increment = augmented.physical_distance_augmented_path_losses(model,batch,1,args,b_rng)
    gb = gradient(model,increment['loss'])
    torch.testing.assert_close(absolute['loss'],increment['loss'],rtol=2e-12,atol=2e-12)
    torch.testing.assert_close(absolute['distance_absolute_es'],increment['distance_increment_es'],rtol=2e-12,atol=2e-12)
    assert increment['loss'] is increment['loss_increment']
    for name,a in ga.items():
        if a is None:
            assert gb[name] is None
        else:
            torch.testing.assert_close(a,gb[name],rtol=2e-10,atol=2e-11)
    assert torch.equal(a_rng.get_state(),b_rng.get_state())


def test_future_isolation_one_draw_and_each_future_member_gets_gradient(monkeypatch):
    model,batch,args = fixture(3)
    args.distance_path_objective = 'increment'
    other = copy.deepcopy(batch)
    other['coords'][:,4:] += torch.arange(6,dtype=torch.float64)[None,None,:,None,None]*.8
    calls = dict(rollout=0,posterior=0)
    for name,label in [('rollout','rollout'),('initial_bath_state','posterior')]:
        original = getattr(model.kernel,name)
        def count(*a,_original=original,_label=label,**kw):
            calls[_label] += 1
            return _original(*a,**kw)
        monkeypatch.setattr(model.kernel,name,count)
    captured = []
    original = augmented.sample_paths
    def capture(*a,**kw):
        assert a[1].shape[1]==4 and kw['observed_time_ps'].shape[1]==4
        paths = original(*a,**kw)
        paths['ca_paths'].retain_grad()
        captured.append(paths)
        return paths
    monkeypatch.setattr(augmented,'sample_paths',capture)
    a_rng,b_rng = rng(),rng()
    a = augmented.physical_distance_augmented_path_losses(model,batch,3,args,a_rng)
    b = augmented.physical_distance_augmented_path_losses(model,other,3,args,b_rng)
    assert calls==dict(rollout=2,posterior=2)
    assert torch.equal(captured[0]['ca_paths'],captured[1]['ca_paths'])
    assert torch.equal(captured[0]['h_paths'],captured[1]['h_paths'])
    assert not torch.equal(a['distance_increment_es'],b['distance_increment_es'])
    assert torch.equal(a_rng.get_state(),b_rng.get_state())
    a['distance_increment_es'].backward()
    gradients = captured[0]['ca_paths'].grad[:,:,1:]
    assert bool(torch.isfinite(gradients).all())
    assert bool((gradients.square().sum((-1,-2))>0).all())
    assert all(p.grad is None for p in model.bundle.parameters())


@pytest.mark.parametrize('objective',augmented.DISTANCE_OBJECTIVES)
def test_masked_nan_padding_future_extra_support_and_physical_gaps(objective):
    model,batch,args = fixture(3)
    args.distance_path_objective = objective
    batch['residue_mask'][:,:,-1] = False
    batch['coords'][:,:,-1] = float('nan')
    batch['residue_index'][0] = torch.tensor([0,1,5,6,9,10])
    batch['chain_id'][0,-2:] = 1
    other = copy.deepcopy(batch)
    other['residue_mask'][:,4:,-1] = True
    other['coords'][:,4:,-1] = 1e6
    a_rng,b_rng = rng(),rng()
    a = augmented.physical_distance_augmented_path_losses(model,batch,3,args,a_rng)
    ga = gradient(model,a['loss'])
    b = augmented.physical_distance_augmented_path_losses(model,other,3,args,b_rng)
    gb = gradient(model,b['loss'])
    assert all(torch.equal(a[k],b[k]) for k in a)
    equal_gradients(ga,gb)
    assert torch.equal(a_rng.get_state(),b_rng.get_state())


@pytest.mark.parametrize('field,value',[
    ('distance_path_objective','unknown'),('distance_path_weight',.5),
    ('contact_estimator','population_u'),('coord_objective','joint_path'),('n_samples',3),
    ('distance_weight',1.),('grad_steps',1),
])
def test_unregistered_configuration_fails_before_draw(field,value):
    model,batch,args = fixture(3)
    setattr(args,field,value)
    generator = rng()
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        augmented.physical_distance_augmented_path_losses(model,batch,3,args,generator)
    assert torch.equal(generator.get_state(),before)


def test_nonphysical_kernel_step_fails_before_draw():
    model,batch,args = fixture(3)
    model.kernel.cfg.step_ps = 1.
    generator = rng()
    before = generator.get_state().clone()
    with pytest.raises(ValueError,match='1000 ps kernel'):
        augmented.physical_distance_augmented_path_losses(model,batch,3,args,generator)
    assert torch.equal(generator.get_state(),before)


@pytest.mark.parametrize('horizon',[1,3])
@pytest.mark.parametrize('arm',augmented.DISTANCE_OBJECTIVES)
def test_v1_augmented_equivalence_when_physical_pairs_equal(horizon,arm):
    from protein_jepa_frame.training import physical_distance_augmented_objective_v1 as v1
    model,batch,args=fixture(horizon);args.distance_path_objective=arm
    a_rng,b_rng=rng(),rng()
    a=v1.physical_distance_augmented_path_losses(model,batch,horizon,args,a_rng)
    ga=gradient(model,a['loss'])
    b=augmented.physical_distance_augmented_path_losses(model,batch,horizon,args,b_rng)
    gb=gradient(model,b['loss'])
    assert all(torch.equal(a[key],b[key]) for key in a)
    equal_gradients(ga,gb)
    assert torch.equal(a_rng.get_state(),b_rng.get_state())


def test_duplicate_legacy_embedding_ids_supported_without_changing_base_or_inputs():
    model,batch,args=fixture(3)
    batch['residue_index'][0,1]=batch['residue_index'][0,0]
    before=copy.deepcopy(batch);a_rng,b_rng=rng(),rng()
    a=frozen.ordered_path_losses(model,batch,3,args,a_rng);ga=gradient(model,a['loss'])
    b=augmented.physical_distance_augmented_path_losses(model,batch,3,args,b_rng);gb=gradient(model,b['loss_base'])
    assert torch.equal(a['loss'],b['loss_base'])
    equal_gradients(ga,gb)
    assert torch.equal(a_rng.get_state(),b_rng.get_state())
    assert all(torch.equal(value,batch[key]) for key,value in before.items())


@pytest.mark.parametrize('bad',['missing_ids','missing_eligibility','duplicate_geometry','shape','dtype','asymmetric','diagonal','empty'])
def test_physical_support_fails_before_any_sampling_or_rng_change(monkeypatch,bad):
    model,batch,args=fixture(3)
    if bad=='missing_ids':batch.pop('geometry_row_ids')
    elif bad=='missing_eligibility':batch.pop('physical_pair_eligibility')
    elif bad=='duplicate_geometry':batch['geometry_row_ids'][0,1]=0
    elif bad=='shape':batch['physical_pair_eligibility']=batch['physical_pair_eligibility'][0]
    elif bad=='dtype':batch['geometry_row_ids']=batch['geometry_row_ids'].float()
    elif bad=='asymmetric':batch['physical_pair_eligibility'][0,0,1]=True
    elif bad=='diagonal':batch['physical_pair_eligibility'][0,0,0]=True
    else:batch['physical_pair_eligibility'].zero_()
    monkeypatch.setattr(augmented,'sample_paths',lambda *a,**kw:pytest.fail('sampling before physical support validation'))
    generator=rng();before=generator.get_state().clone();global_before=torch.get_rng_state().clone()
    with pytest.raises(ValueError):augmented.physical_distance_augmented_path_losses(model,batch,3,args,generator)
    assert torch.equal(generator.get_state(),before) and torch.equal(torch.get_rng_state(),global_before)
    assert all(parameter.grad is None for parameter in model.parameters())
