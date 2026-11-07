"""Synthetic TRAIN attribution tests: no actual checkpoint/manifest/trajectory reads."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from run_scripts import diagnose_train_path_readout_v1 as d


def metadata(length=6):
    aa=torch.zeros(length,dtype=torch.long);ri=torch.arange(length);ci=torch.zeros_like(ri)
    mask=torch.ones(length,dtype=torch.bool)
    anchor=torch.stack((ri.float()*3.8,ri.float().sin(),ri.float().cos()),-1)
    rotation=torch.eye(3).expand(length,3,3)
    return aa,ri,ci,rotation,anchor,mask


@pytest.mark.parametrize('horizon',[1,8])
def test_real_geometric_decoder_p32_exact_adapter_batch_parity(horizon):
    from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
    from protein_jepa_frame.models.ca_geometry import project_ca_bonds
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        decoder=GeometricDeltaCADecoder().eval().requires_grad_(False)
    aa,ri,ci,rotation,anchor,mask=metadata(8)
    mask[-1]=False;anchor[-1]=0
    latent=torch.randn(12,horizon+1,8,128,generator=torch.Generator().manual_seed(8))
    bundle=SimpleNamespace(decoder=decoder,projection_steps=32)
    raw,p32=d.decode_latents(bundle,latent,aa,ri,ci,rotation,anchor,mask)
    # Independent reconstruction of the frozen adapter ordering, exercising 64-boundary and h1.
    expected=torch.empty_like(raw);expected[:,0]=anchor
    def production(member,time):
        rep=lambda x:x[None].expand(len(member),*x.shape)
        with torch.no_grad():
            value=decoder(latent[member,time],latent[member,0],rep(aa),rep(ri),rep(ci),rep(rotation),rep(anchor),rep(mask))
            return project_ca_bonds(value,rep(anchor),rep(mask),rep(ri),rep(ci),32)
    expected[:,-1]=production(torch.arange(12),torch.full((12,),horizon))
    for offset in range(0,12*(horizon-1),64):
        index=torch.arange(offset,min(offset+64,12*(horizon-1)))
        expected[index//(horizon-1),index%(horizon-1)+1]=production(index//(horizon-1),index%(horizon-1)+1)
    assert torch.equal(p32,expected)
    assert bundle.projection_steps==32
    assert torch.equal(raw[:,0],anchor.expand(12,-1,-1))


class TinyBundle(nn.Module):
    def __init__(self,events):
        super().__init__();self.register_buffer('identity',torch.tensor(1.));self.projection_steps=32
        self.events=events
    def encode(self,coords,mask,aa,ri,ci):
        self.events.append('encode')
        return coords[:,:,1]*mask[:,:,None]
    def decoder(self,future,initial,aa,ri,ci,rotation,anchor,mask):
        return anchor+(future-initial)*mask[:,:,None]


class TinyModel(nn.Module):
    def __init__(self,events):
        super().__init__();self.bundle=TinyBundle(events);self.kernel=nn.Linear(1,1)
        self.kernel.cfg=SimpleNamespace(step_ps=1000.)
        self.eval()


def batch_and_args(horizon=4):
    from protein_jepa_frame.training.train_lasp_ca_ordered_v1 import arguments
    aa,ri,ci,rotation,anchor,mask=metadata()
    frames=[]
    for t in range(4+horizon):
        ca=anchor.clone();ca[:,1]+=.05*t*ri.float().sin()
        frames.append(torch.stack((ca+torch.tensor([0.,1.,0.]),ca,ca+torch.tensor([1.,0.,0.]),ca+torch.tensor([0.,0.,1.])),1))
    coords=torch.stack(frames)[None]
    batch=dict(coords=coords,residue_mask=mask[None,None].expand(1,4+horizon,-1).clone(),
        atom_mask=torch.ones(1,4+horizon,len(mask),4,dtype=torch.bool),aatype=aa[None],
        residue_index=ri[None],chain_id=ci[None],time_ps=torch.arange(4+horizon)[None].float()*1000,
        temperature_k=torch.tensor([348.]),protein_id=['synthetic'],trajectory_id=['synthetic'])
    return batch,arguments('unused','marginal_path',device='cpu')


def install_fake(monkeypatch,events):
    import evaluation.ca_trajectory_adapter as adapter
    from protein_jepa_frame.training import ordered_path_objective_v1 as objective
    monkeypatch.setattr(objective,'_frozen',lambda model:None)
    def sample(model,observed,mask,aa,ri,ci,temp,horizon,members,generator,**kwargs):
        events.append('sample_start')
        assert observed.shape[1]==4 and members==12
        h0=model.bundle.encode(observed[:,-1],mask,aa,ri,ci)[0]
        increments=torch.randn(members,horizon,len(h0),3,generator=generator)*.01
        latent=torch.cat((h0[None,None].expand(members,1,-1,-1),h0[None,None]+increments.cumsum(1)),1)
        from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
        rotation,anchor=_extract_backbone_frames(observed[:,-1])
        _,ca=d.decode_latents(model.bundle,latent,aa[0],ri[0],ci[0],rotation[0],anchor[0],mask[0])
        events.append('sample_complete')
        return SimpleNamespace(latent=latent[None],ca=ca[None],bath=None,step_ps=1000.)
    monkeypatch.setattr(adapter,'sample_ca_trajectory',sample)


def test_six_roles_same_h_forecast_before_oracle_rng_no_updates_and_replay(monkeypatch,tmp_path):
    events=[];install_fake(monkeypatch,events)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer instantiated'))
    batch,args=batch_and_args();model=TinyModel(events)
    before=copy.deepcopy(model.state_dict());generator=torch.Generator().manual_seed(42)
    saved,report=d.diagnose_batch(model,batch,4,args,generator)
    assert tuple(saved['roles'])==d.ROLES and events.count('sample_start')==1
    assert events[:3]==['sample_start','encode','sample_complete']
    assert events[3:]==['encode']*5
    assert report['forecast_P32_exact_adapter_match']
    assert report['rng_sha256']['rng_before']!=report['rng_sha256']['rng_after_forecast']
    assert report['rng_sha256']['rng_after_forecast']==report['rng_sha256']['rng_after']
    assert saved['true_H'].shape==(1,5,6,3) and saved['forecast_H'].shape==(12,5,6,3)
    for value in saved['roles'].values():
        assert torch.equal(value[:,0],saved['anchor'].to(value.dtype).expand(len(value),-1,-1))
    assert report['roles']['true_MD']['ordered_metrics']['scalars']['per_step_rmsd_mean']==0.
    assert report['roles']['oracle_raw']['ordered_metrics']['scalars']['coordinate_path_energy_score'] is None
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
    assert all(p.grad is None for p in model.parameters())
    p=tmp_path/'replay.pt';torch.save(saved,p);loaded=torch.load(p,weights_only=False)
    assert d.tree_sha(loaded)==d.tree_sha(saved)
    assert d.score_roles(loaded['roles'],loaded['reference_unaligned'],loaded['anchor'].double(),
        loaded['mask'],loaded['residue_index'],loaded['chain_id'],loaded['time_ps'])==report['roles']


def test_successive_rng_not_reset_between_windows(monkeypatch):
    events=[];install_fake(monkeypatch,events)
    model=TinyModel(events);generator=torch.Generator().manual_seed(42);reference=torch.Generator().manual_seed(42)
    for h in (1,4):
        batch,args=batch_and_args(h);saved,_=d.diagnose_batch(model,batch,h,args,generator)
        torch.randn(12,h,6,3,generator=reference)
        assert torch.equal(generator.get_state(),reference.get_state())
        assert torch.equal(saved['rng_after']['sampling'],reference.get_state())


@pytest.mark.parametrize('failure',['missing_residue','missing_atom','nonfinite','time_grid'])
def test_bad_fixed_support_fails_without_sampling_or_replacement(monkeypatch,failure):
    events=[];install_fake(monkeypatch,events);batch,args=batch_and_args()
    if failure=='missing_residue':batch['residue_mask'][0,-1,1]=False
    elif failure=='missing_atom':batch['atom_mask'][0,-1,1,0]=False
    elif failure=='nonfinite':batch['coords'][0,-1,1,0,0]=float('nan')
    else:batch['time_ps'][0,-1]+=1.
    with pytest.raises(ValueError):d.diagnose_batch(TinyModel(events),batch,4,args,torch.Generator().manual_seed(42))
    assert not events


def test_latent_descriptives_exact_support_norm_variance_increments():
    h=torch.tensor([[[[1.,3.],[float('nan'),0.]],[[4.,8.],[float('nan'),0.]]]])
    mask=torch.tensor([True,False]);tensors,report=d.latent_descriptives(h,mask)
    assert tensors['channel_mean'][0,:,0].tolist()==[2.,6.]
    assert tensors['channel_variance'][0,:,0].tolist()==[1.,4.]
    assert tensors['centered_norm'][0,0,0]==pytest.approx(2**.5)
    assert tensors['increment_norm'][0,0,0]==pytest.approx(34**.5)
    assert report['curves']['channel_mean']['mean_over_members_and_valid_residues']==[2.,6.]
    json.dumps(report,allow_nan=False)
    with pytest.raises(ValueError,match='nonfinite'):d.latent_descriptives(h,torch.ones(2,dtype=torch.bool))


def test_fixed_gauge_shape_gauge_geometry_and_undefined_lags():
    aa,ri,ci,rotation,anchor,mask=metadata()
    reference=torch.stack((anchor,anchor)).double();prediction=reference[None].clone();prediction[:,1]+=torch.tensor([1.,0.,0.])
    roles={name:prediction.clone() for name in d.ROLES}
    out=d.score_roles(roles,reference,anchor.double(),mask,ri,ci,torch.tensor([0.,1000.]))
    row=out['oracle_raw']
    assert row['anchor_gauge']['path_length_per_member']==pytest.approx([1.])
    assert row['ordered_metrics']['scalars']['path_length']<1e-10
    assert row['anchor_gauge']['increment_rmsd_per_member'][0]==pytest.approx([1.])
    lag=row['ordered_metrics']['lag_statistics']['prediction']['aggregate'][-1]
    assert lag['n_pairs']==0 and lag['acf'] is None and lag['msd'] is None
    roles['oracle_raw'][:,1,1]=roles['oracle_raw'][:,1,0]
    out=d.score_roles(roles,reference,anchor.double(),mask,ri,ci,torch.tensor([0.,1000.]))
    assert out['oracle_raw']['geometry_per_member_per_time'][0][1]['ca_bond_invalid_fraction']>0
    assert out['oracle_raw']['n_paths']==1


def test_projection_padding_sanitization_and_anchor_identity():
    aa,ri,ci,rotation,anchor,mask=metadata();mask[-1]=False;anchor[-1]=0
    x=anchor[None,None].expand(2,3,-1,-1).clone();x[:,:,~mask]=float('nan')
    out=d.project_path(x,anchor,mask,ri,ci)
    assert torch.isfinite(out).all() and torch.equal(out[:,0],anchor.expand(2,-1,-1))
    x[0,1,0,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):d.project_path(x,anchor,mask,ri,ci)


def test_unregistered_or_mutated_run_fails_before_load_and_preserves_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path);monkeypatch.setattr(d,'OUT',tmp_path/'out')
    monkeypatch.setattr(d,'execute',lambda *a:pytest.fail('model or data before source checks'))
    with pytest.raises(ValueError,match='register before'):d.run()
    d.OUT.mkdir();(tmp_path/'changed.py').write_text('changed')
    d.write_new(d.OUT/'protocol.json',dict(schema_version=d.VERSION,source_hashes={'changed.py':'0'*64},input_hashes={}))
    with pytest.raises(ValueError,match='frozen file changed'):d.run()
    record=json.loads((d.OUT/'execution.json').read_text())
    assert record['status']=='failed' and record['exit_code']==1 and record['pid']>0
    with pytest.raises(ValueError,match='refusing rerun'):d.run()


def test_registration_metadata_only_source_snapshot_checkpoint_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path);monkeypatch.setattr(d,'OUT',tmp_path/'out')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('checkpoint deserialization during register'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA queried during register'))
    monkeypatch.setattr(d,'SOURCES',('new.py',));(tmp_path/'new.py').write_text('new')
    (tmp_path/'old.py').write_text('old');(tmp_path/'old_snapshot').mkdir();(tmp_path/'old_snapshot/old.py').write_text('old')
    inputs={}
    for name in (*d.TRAIN_INPUTS,d.CHECKPOINT):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic only')
        inputs[name]=d.sha(p)
    monkeypatch.setattr(d,'CHECKPOINT_SHA',inputs[d.CHECKPOINT])
    hdf=tmp_path/'fake.h5';hdf.write_text('stat only')
    selected=[dict(step=i,horizon=h,record_path=str(hdf)) for i,h in enumerate((1,4,16,64),1)]
    historical=[dict(row,actual_batch_sha256='a'*64) for row in selected]
    for name,value in ((d.WINDOW_LOG,''.join(json.dumps(r)+'\n' for r in historical)),
        (d.HISTORY,json.dumps(dict(training_windows=selected,input_hashes=inputs,source_hashes={'old.py':d.sha('old.py')},
            source_snapshot='old_snapshot',selected_hdf5_stats={str(hdf):d.file_stat(hdf)},training_arguments={'marginal_path':{}}))),
        (d.EXECUTION,json.dumps(dict(status='completed',jobs=[dict(kind='train',returncode=0)],
            checkpoint_audit=dict(step=500,optimizer_steps=500),latest_sha256=d.CHECKPOINT_SHA)))):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    for constant,name in (('HISTORY_SHA',d.HISTORY),('EXECUTION_SHA',d.EXECUTION),('WINDOW_LOG_SHA',d.WINDOW_LOG)):
        monkeypatch.setattr(d,constant,d.sha(name))
    d.register();protocol=json.loads((d.OUT/'protocol.json').read_text());d.verify_files(protocol)
    assert protocol['device']=='cpu' and protocol['torch_threads']==2 and protocol['optimizer_steps']==0
    assert protocol['historical_actual_batches']==historical and protocol['roles']==list(d.ROLES)
    with pytest.raises(ValueError,match='existing'):d.register()
    hdf.write_text('changed stat')
    with pytest.raises(ValueError,match='HDF identity'):d.verify_files(protocol)


def test_paths_exclusive_writes_and_nested_tensor_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path)
    with pytest.raises(ValueError):d.path('test.jsonl')
    with pytest.raises(ValueError):d.path(tmp_path.parent/'outside')
    p=tmp_path/'result.json';d.write_new(p,{'ok':True})
    with pytest.raises(FileExistsError):d.write_new(p,{})
    bad=tmp_path/'bad.json'
    with pytest.raises(ValueError):d.write_new(bad,{'bad':float('nan')})
    assert not bad.exists()
    state={'x':{'h':torch.ones(2)},'meta':[1,None],'empty':{}}
    before=d.tree_sha(state);state['x']['h'][0]=2
    assert d.tree_sha(state)!=before


def test_cpu_visibility_guard_precedes_model_or_rng_work(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('checkpoint opened before CPU guard'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA queried before CPU guard'))
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES empty'):d.execute({})
