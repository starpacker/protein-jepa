"""Synthetic CPU tests for the unchanged C0 production path and reuse contract."""
import copy
from dataclasses import asdict
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_hs_target_pilot import tiny,TinyBundle
from run_scripts import run_hs_markov_pilot_v1 as pilot
from protein_jepa_frame.training import train_lasp_ca_dynamics as training
from scripts import eval_ca_endpoint_scores as endpoint
from scripts import eval_lasp_ca_dynamics as ensemble
from scripts import eval_lasp_ca_suite as suite


def zero_c(model):
    model.kernel.cfg.coupling_scale=0.;model.kernel.bath.coupling_scale=0.
    return model


def call_sample(model,batch,generator):
    observed=batch['coords'][:,:4];mask=batch['residue_mask'][:,:4].all(1)
    with torch.no_grad():
        return model.eval().sample(observed,mask,*[batch[k] for k in ('aatype','residue_index','chain_id')],
            batch['temperature_k'],2,3,generator)


def test_complete_exact_spatial_c0_sample_is_history_value_independent():
    model,batch=tiny();zero_c(model);changed=copy.deepcopy(batch)
    changed['coords'][:,:3]+=torch.arange(6,dtype=torch.float64)[None,None,:,None,None]*.7
    ga=torch.Generator().manual_seed(87);gb=torch.Generator().manual_seed(87)
    a=call_sample(model,batch,ga);b=call_sample(model,changed,gb)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert torch.equal(ga.get_state(),gb.get_state())
    assert model.history_conditioning and model.kernel.s0_mode=='exact_spatial'


def test_actual_full_sampler_h_draws_and_rng_match_hs_c0(monkeypatch):
    hs,batch=tiny();markov=zero_c(copy.deepcopy(hs));original=torch.randn;calls=[]
    def capture(*args,**kwargs):
        value=original(*args,**kwargs);calls.append(value.detach().clone());return value
    monkeypatch.setattr(torch,'randn',capture)
    ga=torch.Generator().manual_seed(71);gb=torch.Generator().manual_seed(71)
    call_sample(hs,batch,ga);a=calls[:];calls.clear();call_sample(markov,batch,gb);b=calls[:]
    assert [tuple(v.shape) for v in a]==[(1,3,12),(3,6,8),(3,6,2),(3,6,8),(3,6,2)]
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert torch.equal(ga.get_state(),gb.get_state())


def test_explicit_s0_and_s_noise_cannot_change_c0_h_path():
    model,batch=tiny();zero_c(model);k=model.kernel
    mask=batch['residue_mask'][:,0];h0=model.bundle.encode(batch['coords'][:,3],mask,*[batch[n] for n in ('aatype','residue_index','chain_id')])
    temp=batch['temperature_k'];factor=model.anchor_kernel(batch['coords'][:,3,:,1],mask)
    gen=torch.Generator().manual_seed(25)
    z=torch.randn(1,3,1,6,8,dtype=torch.float64,generator=gen)
    zs=torch.randn(1,3,1,6,2,dtype=torch.float64,generator=gen);s0=torch.randn(1,6,2,dtype=torch.float64,generator=gen)
    with torch.no_grad():
        args=dict(seq_feature=h0,noise_factor=factor,collective_mobility=True,microsteps=1,brownian_H=z,return_bath=True)
        a=k.rollout_microstep_v1(h0,mask,temp,3,S0=s0,brownian_S=zs,**args)
        b=k.rollout_microstep_v1(h0,mask,temp,3,S0=s0+11,brownian_S=zs*-4,**args)
    assert torch.equal(a[0],b[0]) and not torch.equal(a[1],b[1])


@pytest.mark.parametrize('change',['s0','s_noise'])
def test_production_rollout_s0_snoise_independence_at_identical_h_draws(monkeypatch,change):
    model,batch=tiny();zero_c(model);k=model.kernel
    mask=batch['residue_mask'][:,0];h0=model.bundle.encode(batch['coords'][:,3],mask,*[batch[n] for n in ('aatype','residue_index','chain_id')])
    factor=model.anchor_kernel(batch['coords'][:,3,:,1],mask)
    s0=torch.ones(1,6,2,dtype=h0.dtype);ga=torch.Generator().manual_seed(51);gb=torch.Generator().manual_seed(51)
    args=dict(seq_feature=h0,noise_factor=factor,collective_mobility=True,return_bath=True)
    with torch.no_grad():a=k.rollout(h0,mask,batch['temperature_k'],3,S0=s0,generator=ga,**args)
    original=torch.randn;seen=[]
    def changed_noise(*a,**kw):
        value=original(*a,**kw);seen.append(tuple(value.shape))
        return value+13 if change=='s_noise' and value.shape==s0.shape else value
    monkeypatch.setattr(torch,'randn',changed_noise)
    with torch.no_grad():b=k.rollout(h0,mask,batch['temperature_k'],3,
        S0=s0+7 if change=='s0' else s0,generator=gb,**args)
    assert seen==[(1,6,8),(1,6,2)]*3
    assert torch.equal(a[0],b[0]) and not torch.equal(a[1],b[1])
    assert torch.equal(ga.get_state(),gb.get_state())


def test_zero_c_endpoint_loss_has_zero_bath_predictive_gradients_and_active_energy_D():
    model,batch=tiny();zero_c(model);model.train()
    args=SimpleNamespace(**dict(pilot.old.EXPECTED,target='endpoint'));args.n_samples=3;args.grad_steps=2
    result=model.losses(batch,2,args,torch.Generator().manual_seed(7));result['loss'].backward()
    for name,p in model.kernel.named_parameters():
        if name.startswith(('bath.c_net.','bath.gamma_net.','history_posterior.')):
            assert p.grad is None or torch.count_nonzero(p.grad)==0,name
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.kernel.energy.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.kernel.bath.d_net.parameters())
    assert all(p.grad is None for p in model.bundle.parameters())


def saved_zero_checkpoint(tmp_path,monkeypatch):
    model,_=tiny();model=zero_c(model.float())
    component=dict(checkpoint_kind='synthetic_test',decoder_mode='geometric_ca',calibration_args={},
        encoder=model.bundle.encoder.state_dict(),delta_ca_decoder=model.bundle.decoder.state_dict())
    ck=dict(checkpoint_kind='lasp_ca_dynamics',kernel=model.kernel.state_dict(),kernel_config=asdict(model.kernel.cfg),
        component_checkpoint=component,args=dict(s0_mode='exact_spatial',history_conditioning=True,
        noise_mix=.8,noise_length=8.,noise_kernel='gaussian',noise_cutoff=10.,collective_mobility=True,coupling_scale=0.))
    path=tmp_path/'c0.pt';torch.save(ck,path)
    def bundle(path,device):
        source=torch.load(path,map_location='cpu',weights_only=False);value=TinyBundle().float()
        value.encoder.load_state_dict(source['encoder']);value.decoder.load_state_dict(source['delta_ca_decoder']);return value
    monkeypatch.setattr(training,'CAComponentBundle',bundle)
    return path,ck


@pytest.mark.parametrize('producer',['endpoint','ensemble','suite'])
def test_every_real_loader_constructor_restores_zero(tmp_path,monkeypatch,producer):
    path,ck=saved_zero_checkpoint(tmp_path,monkeypatch)
    if producer=='endpoint':model=endpoint.load_model(str(path),ck,'cpu')
    else:
        module=ensemble if producer=='ensemble' else suite;captured=[]
        class Loaded(Exception):pass
        def load(component,device,checkpoint,*args,**kwargs):
            assert device=='cuda'
            captured.append(training.CADynamics(component,'cpu',checkpoint,*args,**kwargs));raise Loaded()
        monkeypatch.setattr(module,'CADynamics',load)
        monkeypatch.setattr(sys,'argv',[module.__name__,'--checkpoint',str(path),'--out',str(tmp_path/'out.json')])
        with pytest.raises(Loaded):module.main()
        model=captured[0]
    assert model.kernel.cfg.coupling_scale==0. and model.kernel.bath.coupling_scale==0.
    assert model.kernel.s0_mode=='exact_spatial' and model.history_conditioning and model.collective_mobility
    _,c,_=model.kernel.bath(torch.zeros(1,6,8),torch.tensor([320.]))
    assert torch.isfinite(c).all() and torch.count_nonzero(c)==0
    assert not torch.cuda.is_initialized()


def test_commands_only_change_scale_output_and_all_three_real_parsers_accept():
    commands=pilot.commands();a=commands['train'][:];b=pilot.old.commands('endpoint')['train'][:]
    for flag in ('--coupling-scale','--output'):a[a.index(flag)+1]=b[b.index(flag)+1]='<changes>'
    assert a==b
    parsed={k:pilot.parse_cli(v) for k,v in commands.items()}
    assert tuple(commands)==('train','ensemble','endpoint')
    assert parsed['train']['coupling_scale']==0. and parsed['train']['history_conditioning']
    assert parsed['train']['s0_mode']=='exact_spatial' and parsed['train']['bath_dim']==8
    assert parsed['train']['steps']==500 and parsed['train']['n_samples']==12
    assert '--max-residues' not in commands['endpoint']
    pilot.assert_endpoint_fixed96()


@pytest.mark.parametrize('field',['python','numpy','torch','cuda'])
def test_strict_global_rng_tree_rejects_single_state_change(field):
    before={'python':(3,(1,2,3),None),'numpy':('MT19937',np.array([1,2],dtype=np.uint32),0,0,0.),
        'torch':torch.tensor([1,2],dtype=torch.uint8),'cuda':[torch.tensor([1,2],dtype=torch.uint8)]}
    after=copy.deepcopy(before);pilot.exact_tree(after,before,'RNG')
    if field=='python':after[field]=(3,(1,2,4),None)
    elif field=='numpy':after[field][1][0]+=1
    elif field=='torch':after[field][0]+=1
    else:after[field][0][0]+=1
    with pytest.raises(ValueError,match='RNG'):pilot.exact_tree(after,before,'RNG')


def test_metadata_stream_does_not_read_coordinates_and_matches_stateless_rule(tmp_path,monkeypatch):
    path=tmp_path/'train.h5';path.write_bytes(b'metadata fixture')
    class Dataset:
        def __init__(self,h):self.h=h
        def __len__(self):return 101
        def __getitem__(self,index):pytest.fail('metadata audit cannot read coordinates')
        def _locate(self,index):return SimpleNamespace(protein_id='p',trajectory_id='t',path=path,temperature_k=320.),np.arange(index,index+4+self.h)
    monkeypatch.setattr(pilot,'make_loader',lambda args,manifest,h,training:SimpleNamespace(dataset=Dataset(h)))
    rows,stats=pilot.selected_training_stream(dict(manifest='train',horizons=[1,4,16,64],seed=42))
    assert len(rows)==500 and [rows[i]['horizon'] for i in range(4)]==[1,4,16,64]
    for row in rows:
        assert row['dataset_index']==pilot.random.Random(42+1000003*row['step']).randrange(101)
        assert row['target_frame']-row['context_frames'][-1]==row['horizon']
    assert stats[str(path)]['size']==16


def test_protocol_lock_snapshot_and_exclusive_execution_files(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'RESULTS',tmp_path)
    protocol=dict(input_hashes={},source_hashes={},source_snapshot=str(tmp_path/'snapshot'))
    path=tmp_path/'protocol.json';pilot.old.write_new(path,protocol);digest=pilot.old.sha256_file(path)
    pilot.verify(protocol,digest)
    with pytest.raises(FileExistsError):pilot.old.write_new(path,protocol)
    path.write_text('{}')
    with pytest.raises(ValueError,match='protocol changed'):pilot.verify(protocol,digest)


def test_failed_real_job_status_preserves_pid_code_and_never_retries(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'RESULTS',tmp_path);monkeypatch.setattr(pilot,'ARTIFACTS',tmp_path/'artifacts')
    protocol=dict(schema_version=pilot.VERSION,input_hashes={},source_hashes={},source_snapshot=str(tmp_path/'snapshot'),
        gpu=1,commands={'train':['fixed-command']})
    pilot.old.write_new(tmp_path/'protocol.json',protocol);calls=[]
    class FailedProcess:
        pid=123456
        def __init__(self,command,**kwargs):calls.append((command,kwargs['env']['CUDA_VISIBLE_DEVICES']))
        def wait(self):return 9
    monkeypatch.setattr(pilot.subprocess,'Popen',FailedProcess)
    with pytest.raises(RuntimeError,match='exited9'):pilot.run()
    saved=json.loads((tmp_path/'execution.json').read_text())
    assert saved['status']=='failed' and saved['jobs'][0]['pid']==123456 and saved['jobs'][0]['returncode']==9
    assert calls==[(['fixed-command'],'1')]
    with pytest.raises(FileExistsError):pilot.run()
