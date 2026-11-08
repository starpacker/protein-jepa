"""CPU mathematical and production-path contracts; no real trajectory access."""
import copy
from types import SimpleNamespace

import pytest
import torch

from test_hs_target_pilot import tiny, equal_gradients
from run_scripts.run_hs_target_pilot import EXPECTED
from protein_jepa_frame.training import ordered_path_objective_v1 as path


def setup(horizon=2,batch_size=1):
    model,batch=tiny()
    if horizon+4>batch['coords'].shape[1]:
        extra=horizon+4-batch['coords'].shape[1]
        batch['coords']=torch.cat([batch['coords'],batch['coords'][:,-1:].expand(-1,extra,-1,-1,-1).clone()+.2],1)
        batch['residue_mask']=torch.cat([batch['residue_mask'],batch['residue_mask'][:,-1:].expand(-1,extra,-1).clone()],1)
    batch['coords']=batch['coords'][:,:horizon+4]
    batch['residue_mask']=batch['residue_mask'][:,:horizon+4]
    batch['time_ps']=torch.arange(horizon+4,dtype=torch.float64)[None]*model.kernel.cfg.step_ps
    if batch_size>1:
        batch={k:v.expand(batch_size,*v.shape[1:]).clone() for k,v in batch.items()}
    args=SimpleNamespace(**dict(EXPECTED,target='endpoint',coord_objective='joint_path',path_weight=.5,checkpoint_decoder=False))
    args.n_samples=3;args.grad_steps=horizon
    return model,batch,args


def draw(model,batch,args,horizon,generator=None):
    return path.sample_paths(model,batch['coords'][:,:4],batch['residue_mask'][:,:4].all(1),
        *[batch[k] for k in ('aatype','residue_index','chain_id')],batch['temperature_k'],horizon,args.n_samples,
        generator or torch.Generator().manual_seed(20261101),args.grad_steps,
        observed_time_ps=batch['time_ps'][:,:4],checkpoint_decoder=args.checkpoint_decoder)


def gradient(model,loss):
    model.zero_grad(set_to_none=True);loss.backward()
    return {name:None if p.grad is None else p.grad.detach().clone() for name,p in model.named_parameters()}


@pytest.mark.parametrize('horizon',[1,3])
def test_terminal_original_batch_shape_value_and_rng_are_bitwise_identical(horizon,monkeypatch):
    model,batch,args=setup(horizon,batch_size=2)
    shapes=[];original=model.bundle.decode_ca
    def capture(h,*a,**kw):shapes.append(tuple(h.shape));return original(h,*a,**kw)
    monkeypatch.setattr(model.bundle,'decode_ca',capture)
    ga=torch.Generator().manual_seed(31);gb=torch.Generator().manual_seed(31)
    ca,h=model.sample(batch['coords'][:,:4],batch['residue_mask'][:,:4].all(1),
        *[batch[k] for k in ('aatype','residue_index','chain_id')],batch['temperature_k'],horizon,args.n_samples,ga,args.grad_steps)
    result=draw(model,batch,args,horizon,gb)
    assert torch.equal(ca,result['endpoint_ca']) and torch.equal(h,result['endpoint_h'])
    assert torch.equal(result['ca_paths'][:,:,-1],ca) and torch.equal(result['h_paths'][:,:,-1],h)
    assert torch.equal(ga.get_state(),gb.get_state())
    assert result['ca_paths'].shape==(2,3,horizon+1,6,3)
    assert result['h_paths'].shape==(2,3,horizon+1,6,8)
    assert torch.equal(result['ca_paths'][:,:,0],batch['coords'][:,None,3,:,1].expand(2,3,6,3))
    assert shapes==[(6,6,8)]*(1+horizon)  # old endpoint, then new endpoint and intermediates


@pytest.mark.parametrize('objective',path.OBJECTIVES)
@pytest.mark.parametrize('batch_size',[1,3])
def test_h1_original_endpoint_loss_gradients_and_rng_exact(objective,batch_size):
    model,batch,args=setup(1,batch_size=batch_size);args.coord_objective=objective
    ga=torch.Generator().manual_seed(63);gb=torch.Generator().manual_seed(63)
    old=model.losses(batch,1,args,ga);a=gradient(model,old['loss'])
    new=path.ordered_path_losses(model,batch,1,args,gb);b=gradient(model,new['loss'])
    assert all(torch.equal(old[k],new[k]) for k in old)
    assert new['coordinate_path_es'] is new['coord_es'] and new['coordinate_marginal_es'] is new['coord_es']
    equal_gradients(a,b)
    assert torch.equal(ga.get_state(),gb.get_state())


def test_single_posterior_and_one_unreset_continuous_rollout(monkeypatch):
    model,batch,args=setup(4)
    counts={'posterior':0,'rollout':0,'step':0}
    for name,label in [('initial_bath_state','posterior'),('rollout','rollout'),('sample_step','step')]:
        original=getattr(model.kernel,name)
        def capture(*a,_original=original,_label=label,**kw):
            counts[_label]+=1
            return _original(*a,**kw)
        monkeypatch.setattr(model.kernel,name,capture)
    result=path.ordered_path_losses(model,batch,4,args,torch.Generator().manual_seed(72))
    assert counts=={'posterior':1,'rollout':1,'step':4}
    grads=gradient(model,result['loss'])
    assert any(v is not None and v.abs().sum()>0 for k,v in grads.items() if k.startswith('kernel.'))
    assert all(p.grad is None for p in model.bundle.parameters())
    assert all(v is None or torch.isfinite(v).all() for v in grads.values())


def test_future_coordinates_change_labels_but_never_paths_or_rng(monkeypatch):
    model,batch,args=setup(3);other=copy.deepcopy(batch)
    other['coords'][:,4:]+=torch.arange(6,dtype=torch.float64)[None,None,:,None,None]*.8
    observed=[];original=path.sample_paths
    def capture(*a,**kw):
        result=original(*a,**kw);observed.append((result['ca_paths'].detach().clone(),result['h_paths'].detach().clone()));return result
    monkeypatch.setattr(path,'sample_paths',capture)
    ga=torch.Generator().manual_seed(93);gb=torch.Generator().manual_seed(93)
    a=path.ordered_path_losses(model,batch,3,args,ga)
    b=path.ordered_path_losses(model,other,3,args,gb)
    assert all(torch.equal(x,y) for x,y in zip(observed[0],observed[1]))
    assert torch.equal(ga.get_state(),gb.get_state()) and not torch.equal(a['loss'],b['loss'])


def test_joint_distinguishes_dependence_with_identical_time_marginals():
    # Law A is (--)/(++); law B is (-+)/(+-), identical at either individual time.
    a=torch.zeros(1,4,3,1,3,dtype=torch.float64)
    b=a.clone()
    a[0,:,1:,0,0]=torch.tensor([[-1.,-1.],[-1.,-1.],[1.,1.],[1.,1.]])
    b[0,:,1:,0,0]=torch.tensor([[-1.,1.],[-1.,1.],[1.,-1.],[1.,-1.]])
    masks=torch.ones(1,1,dtype=torch.bool);joint=[];marginal=[]
    for sign in (-1.,1.):
        target=torch.zeros(1,3,1,3,dtype=torch.float64);target[:,1:,:,0]=sign
        sa=path.coordinate_path_scores(a,target,masks);sb=path.coordinate_path_scores(b,target,masks)
        assert torch.equal(sa['marginal'],sb['marginal']) and torch.equal(sa['endpoint'],sb['endpoint'])
        joint.append(sb['joint']-sa['joint']);marginal.append(sb['marginal']-sa['marginal'])
    assert torch.stack(joint).mean()>0 and torch.stack(marginal).mean()==0


@pytest.mark.parametrize('checkpoint_decoder',[False,True])
def test_joint_vs_marginal_share_paths_endpoint_terms_and_rng(checkpoint_decoder):
    model,batch,args=setup(3);args.checkpoint_decoder=checkpoint_decoder
    ga=torch.Generator().manual_seed(8);gb=torch.Generator().manual_seed(8)
    a=path.ordered_path_losses(model,batch,3,args,ga)
    args.coord_objective='marginal_path'
    b=path.ordered_path_losses(model,batch,3,args,gb)
    for name in ('coord_es','latent_es','contact','clash','coordinate_path_es','coordinate_marginal_es'):
        assert torch.equal(a[name],b[name]),name
    assert torch.equal(ga.get_state(),gb.get_state())
    torch.testing.assert_close(a['loss']-b['loss'],.5*(a['coordinate_path_es']-a['coordinate_marginal_es']),rtol=1e-12,atol=1e-12)


def test_nonreentrant_decoder_checkpoint_preserves_outputs_gradients_rng():
    model,batch,args=setup(3)
    ga=torch.Generator().manual_seed(52);gb=torch.Generator().manual_seed(52)
    a=path.ordered_path_losses(model,batch,3,args,ga);gradient_a=gradient(model,a['loss'])
    args.checkpoint_decoder=True
    b=path.ordered_path_losses(model,batch,3,args,gb);gradient_b=gradient(model,b['loss'])
    assert all(torch.equal(a[k],b[k]) for k in a)
    equal_gradients(gradient_a,gradient_b)
    assert torch.equal(ga.get_state(),gb.get_state())


@pytest.mark.parametrize('failure',['time','missing_future','short','nonfinite','grad_steps','weight','encoder','submodule_train','atom'])
def test_invalid_time_support_scope_and_nonfinite_fail_before_rng(failure):
    model,batch,args=setup(2)
    if failure=='time':batch['time_ps'][0,4]+=.001
    elif failure=='missing_future':batch['residue_mask'][0,4,0]=False
    elif failure=='short':batch['coords']=batch['coords'][:,:-1]
    elif failure=='nonfinite':batch['coords'][0,4,0,1,0]=float('nan')
    elif failure=='grad_steps':args.grad_steps=1
    elif failure=='weight':args.rmsf_weight=1.
    elif failure=='encoder':next(model.bundle.encoder.parameters()).requires_grad_(True)
    elif failure=='submodule_train':model.bundle.decoder.train()
    else:
        batch['atom_mask']=torch.ones(*batch['coords'].shape[:-1],dtype=torch.bool)
        batch['atom_mask'][0,4,0,0]=False
    g=torch.Generator().manual_seed(91);before=g.get_state().clone()
    with pytest.raises(ValueError):path.ordered_path_losses(model,batch,2,args,g)
    assert torch.equal(before,g.get_state())


def test_masked_nan_safe_without_shrinking_observed_support():
    model,batch,args=setup(2)
    batch['residue_mask'][:,:,-1]=False
    batch['coords'][:,:,-1]=float('nan')
    result=draw(model,batch,args,2)
    assert torch.isfinite(result['ca_paths']).all() and torch.isfinite(result['h_paths']).all()
    terms=path.ordered_path_losses(model,batch,2,args,torch.Generator().manual_seed(4))
    assert all(torch.isfinite(x) for x in terms.values())


@pytest.mark.parametrize('extra_value',[float('nan'),1e6])
def test_future_extra_valid_residues_do_not_change_fixed_support_or_loss(extra_value,monkeypatch):
    model,batch,args=setup(3)
    batch['residue_mask'][:,:,-1]=False
    batch['coords'][:,:,-1]=float('nan')
    other=copy.deepcopy(batch)
    other['residue_mask'][:,4:,-1]=True
    other['coords'][:,4:,-1]=extra_value
    captured=[];original=path.sample_paths
    def capture(*a,**kw):
        result=original(*a,**kw)
        captured.append((a[2].clone(),result['ca_paths'].detach().clone(),result['h_paths'].detach().clone()))
        return result
    monkeypatch.setattr(path,'sample_paths',capture)
    ga=torch.Generator().manual_seed(94);gb=torch.Generator().manual_seed(94)
    a=path.ordered_path_losses(model,batch,3,args,ga);grad_a=gradient(model,a['loss'])
    b=path.ordered_path_losses(model,other,3,args,gb);grad_b=gradient(model,b['loss'])
    assert all(torch.equal(x,y) for x,y in zip(captured[0],captured[1]))
    assert not captured[0][0][:,-1].any() and captured[0][0].sum()==5
    assert all(torch.equal(a[k],b[k]) for k in a)
    equal_gradients(grad_a,grad_b)
    assert torch.equal(ga.get_state(),gb.get_state())


def test_checkpoint_actual_geometric_decoder_with_bond_projection(monkeypatch):
    from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
    from protein_jepa_frame.models.ca_geometry import project_ca_bonds
    model,batch,args=setup(3)
    model.bundle.decoder=GeometricDeltaCADecoder(latent_dim=8,hidden=8,depth=1).double().requires_grad_(False).eval()
    def decode(h,h0,aa,ri,ci,rotation,anchor,mask):
        raw=model.bundle.decoder(h,h0,aa,ri,ci,rotation,anchor,mask)
        return project_ca_bonds(raw,anchor,mask,ri,ci,iterations=2)
    monkeypatch.setattr(model.bundle,'decode_ca',decode)
    ga=torch.Generator().manual_seed(88);gb=torch.Generator().manual_seed(88)
    a=path.ordered_path_losses(model,batch,3,args,ga);a_grad=gradient(model,a['loss'])
    args.checkpoint_decoder=True
    b=path.ordered_path_losses(model,batch,3,args,gb);b_grad=gradient(model,b['loss'])
    assert all(torch.equal(a[k],b[k]) for k in a)
    equal_gradients(a_grad,b_grad)
    assert all(p.grad is None for p in model.bundle.parameters())
    assert torch.equal(ga.get_state(),gb.get_state())
