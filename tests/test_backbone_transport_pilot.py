"""Synthetic CPU integration checks for the separately frozen transport runner."""
import copy
from pathlib import Path
import runpy

import pytest
import torch

from run_scripts import run_backbone_transport_pilot as runner
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame
from protein_jepa_frame.models.stage_c.structure_decoder import reconstruct_atom37_from_frames
from protein_jepa_frame.utils.contracts import state_sha256


def fixture():
    factory=runpy.run_path(str(Path(__file__).with_name('test_backbone_readout_pilot.py')))['fixture']
    return factory()


def rotation():
    # Exactly proper SO(3), avoiding a model/checkpoint or any dataset dependency.
    return torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])


def ideal_row():
    _,row=fixture()
    row['chemical_aatype']=torch.tensor([0,7,14,20,3])
    length=len(row['observed_mask'])
    r=torch.eye(3).expand(1,length,3,3).clone()
    ca=row['observed_backbone'][...,1,:].clone()
    psi=torch.tensor([[-2.8,-.3,.5,1.3,3.1]])
    torsions=torch.zeros(1,length,7);torsions[...,1]=psi
    atoms,_=reconstruct_atom37_from_frames(r,ca,row['chemical_aatype'][None],torsions)
    bb=atoms[...,[0,1,2,4],:]
    row['observed_backbone']=bb
    row['observed_R']=RigidBodyFrame.from_backbone(bb[...,0,:],ca,bb[...,2,:])[0]
    row['oracle_CA']=ca
    return row,psi


def test_observed_O_psi_sign_and_seed_roundtrip_use_chemical_ids():
    row,psi=ideal_row()
    parsed=runner.observed_psi(row)
    circular=torch.atan2((parsed-psi).sin(),(parsed-psi).cos()).abs()
    assert circular.max()<2e-6
    seed=runner.seed_backbone(row,row['observed_R'],row['oracle_CA'])
    assert torch.equal(seed[...,1,:],row['oracle_CA'])
    assert torch.allclose(seed,row['observed_backbone'],atol=3e-6,rtol=0)
    # This prior legitimately preserves an observed angle beyond the head's midpoint range.
    assert parsed.abs().max()>35*torch.pi/36


def test_transport_and_seed_have_no_future_label_or_target_mask_dependency():
    row,_=ideal_row()
    ca=row['oracle_CA']@rotation().T+torch.tensor([2.,-1.,3.])
    before=runner.transport(row,ca)
    seed=runner.seed_backbone(row,before['frame_seed'],ca)
    altered=copy.deepcopy(row)
    altered['target_backbone'].fill_(float('nan'))
    altered['target_atom_mask'].zero_()
    altered['oracle_H'].fill_(float('nan'))
    altered['forecast_H'].fill_(float('nan'))
    after=runner.transport(altered,ca)
    assert torch.equal(before['frame_seed'],after['frame_seed'])
    assert torch.equal(runner.observed_psi(row),runner.observed_psi(altered))
    assert torch.equal(seed,runner.seed_backbone(altered,after['frame_seed'],ca))


def test_observed_control_decode_and_scores_match_frozen_evaluator_bitwise():
    heads,row=fixture()
    old=runner.ev.score_row(heads,row)
    before={name:state_sha256(head) for name,head in heads.items()}
    rng=torch.random.get_rng_state().clone()
    for name in ('init','coordinate'):
        for variant in ('oracle','forecast'):
            ca,h=row[variant+'_CA'],row[variant+'_H']
            bb=runner.decode(heads[name],row,h,ca,row['observed_R'])
            now=runner.ev.score_backbone(bb,row,role='test same frozen path',fixed_ca=ca)
            reference=old['routes'][name+'/'+variant]
            assert now['n_samples']==(12 if variant=='forecast' else 1)
            for field in ('backbone_sha256','error','legacy_geometry','explicit_geometry','fixed_ca_sha256'):
                assert now[field]==reference[field]
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(state_sha256(head)==before[name] for name,head in heads.items())
    assert all(p.grad is None for head in heads.values() for p in head.parameters())


def test_twelve_member_transport_decode_keeps_each_CA_and_ignores_target_labels():
    heads,row=fixture();head=heads['init']
    moved=runner.transport(row,row['forecast_CA'])
    bb=runner.decode(head,row,row['forecast_H'],row['forecast_CA'],moved['frame_seed'])
    assert bb.shape==(12,len(row['observed_mask']),4,3)
    assert torch.equal(bb[...,1,:],row['forecast_CA'])
    altered=copy.deepcopy(row);altered['target_backbone']+=20;altered['target_atom_mask'].zero_()
    again=runner.decode(head,altered,altered['forecast_H'],altered['forecast_CA'],runner.transport(altered,altered['forecast_CA'])['frame_seed'])
    assert torch.equal(bb,again) and not bb.requires_grad


def test_transported_training_is_exactly_frozen_objective_with_only_frame_replaced():
    heads,row=fixture();head=heads['init'].train();head.dropout.p=.3
    row['oracle_CA']=row['oracle_CA']@rotation().T+torch.tensor([2.,-1.,3.])
    for key in ('oracle_H','oracle_CA','observed_R','target_backbone','observed_backbone'):
        row[key].requires_grad_()
    before={k:v.detach().clone() for k,v in row.items() if torch.is_tensor(v)}
    moved=runner.transport(row,row['oracle_CA'])
    assert not torch.equal(moved['frame_seed'],row['observed_R'])
    torch.manual_seed(271)
    coordinate,geometry,details,actual_moved=runner.transported_losses(head,row)
    torch.manual_seed(271)
    reference=runner.base.actual_losses(head,dict(row,observed_R=moved['frame_seed']))
    assert torch.equal(coordinate,reference[0]) and torch.equal(geometry,reference[1])
    assert torch.equal(actual_moved['frame_seed'],moved['frame_seed'])
    (coordinate+0.*geometry).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters() if p.requires_grad)
    assert any(p.grad.abs().sum()>0 for p in head.parameters() if p.requires_grad)
    assert all(row[k].grad is None for k in ('oracle_H','oracle_CA','observed_R','target_backbone','observed_backbone'))
    assert all(torch.equal(row[k],v) for k,v in before.items())


def test_unchanged_CA_control_preserves_same_dropout_draws_loss_and_gradients():
    heads,row=fixture();a=heads['init'].train();b=copy.deepcopy(a)
    a.dropout.p=b.dropout.p=.3
    torch.manual_seed(177)
    first=runner.base.actual_losses(a,row)
    (first[0]+0.*first[1]).backward()
    first_rng=torch.random.get_rng_state().clone()
    torch.manual_seed(177)
    second=runner.transported_losses(b,row)
    (second[0]+0.*second[1]).backward()
    assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1])
    assert torch.equal(first_rng,torch.random.get_rng_state())
    for p,q in zip(a.parameters(),b.parameters()):
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad,q.grad)


def checkpoint_contracts():
    expected=dict(seed=42,steps=500,optimizer_steps=500,optimizer='AdamW',learning_rate=1e-4,
        final_learning_rate=1e-5,warmup_steps=25,weight_decay=1e-4,gradient_clip=1.,geometry_lambda=0.,
        train_cache=runner.TRAIN,initial=runner.INITIAL,validation_used_for_training=False,
        encoder_CA_dynamics_frozen=True,checkpoint_selection='fixed latest; no validation selection')
    initial=dict(step=0,head_tensor_sha256='initial',head_config={'width':16},initialization={'same':True})
    control=dict(step=500,arm='coordinate',initial_head_tensor_sha256='initial',protocol=expected,
        head_config=initial['head_config'],initialization=initial['initialization'],
        input_hashes={p:runner.PINNED[p] for p in (runner.TRAIN,runner.INITIAL)})
    transport=copy.deepcopy(control);transport.update(arm='coordinate_transport',frame_basis=runner.FRAME_BASIS,
        head_tensor_sha256='trained',parameter_delta_l2=.1)
    transport['protocol'].update(schema_version=runner.VERSION,mode='train',frame_basis=runner.FRAME_BASIS)
    return initial,control,transport


@pytest.mark.parametrize('key,value',[('seed',43),('steps',499),('optimizer_steps',499),('geometry_lambda',.1),
                                      ('learning_rate',2e-4),('validation_used_for_training',True)])
def test_matched_training_contract_rejects_changed_budget_objective_or_selection(key,value):
    initial,control,transport=checkpoint_contracts()
    runner.validate_match(initial,control,transport)
    transport['protocol'][key]=value
    with pytest.raises(ValueError):runner.validate_match(initial,control,transport)


def test_matched_contract_rejects_wrong_initial_or_frame_contract():
    initial,control,transport=checkpoint_contracts()
    transport['initial_head_tensor_sha256']='other'
    with pytest.raises(ValueError):runner.validate_match(initial,control,transport)
    initial,control,transport=checkpoint_contracts();transport['frame_basis']='observed'
    with pytest.raises(ValueError):runner.validate_match(initial,control,transport)
