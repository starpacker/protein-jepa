"""Synthetic offline map/readout tests; no actual saved inputs or checkpoint reads."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from run_scripts import diagnose_saved_h_support_readout_v1 as d


def latent_fixture():
    h0=torch.tensor([[1.,2.,4.],[3.,1.,-1.],[9.,9.,9.]],dtype=torch.float32)
    h=h0[None,None].expand(2,3,-1,-1).clone()
    h[:,1:,:2]=h[:,1:,:2]*3+5
    h[:,:,2]=float('nan')
    h0=h[0,0].clone()
    return h,h0,torch.tensor([True,True,False])


def test_map_exact_formula_anchor_masked_bytes_radius_and_float_residuals():
    h,h0,mask=latent_fixture();before=h.clone()
    result,detail,report=d.support_map(h,h0,mask)
    assert d.bitwise_equal(h,before)
    assert d.bitwise_equal(result[:,0],h[:,0]) and d.bitwise_equal(result[...,~mask,:],h[...,~mask,:])
    x=h[:,1:,mask].double();center=x-x.mean(-1,keepdim=True)
    anchor=h0[mask].double();radius=(anchor-anchor.mean(-1,keepdim=True)).norm(dim=-1)
    expected=(center/center.norm(dim=-1,keepdim=True)*radius[None,None,:,None]).float()
    torch.testing.assert_close(result[:,1:,mask],expected,rtol=1e-7,atol=1e-7)
    assert report['future_valid_tokens']==8
    assert report['max_absolute_channel_mean_residual']<1e-6
    assert report['max_absolute_centered_radius_residual']<1e-6
    assert torch.equal(detail['observed_centered_radius'],radius)
    assert report['H_change_max']>1


@pytest.mark.parametrize('bad',['future_constant','anchor_constant','nan','inf','changed_anchor','empty_mask'])
def test_map_rejects_nonfinite_degenerate_or_wrong_support_without_clamp(bad):
    h,h0,mask=latent_fixture()
    if bad=='future_constant':h[0,1,0]=2
    elif bad=='anchor_constant':h[:,:,0]=2;h0=h[0,0].clone()
    elif bad=='nan':h[0,1,0,0]=float('nan')
    elif bad=='inf':h[0,1,0,0]=float('inf')
    elif bad=='changed_anchor':h[1,0,0,0]+=1
    else:mask[:]=False
    with pytest.raises(ValueError):d.support_map(h,h0,mask)


def test_true_h_control_near_identity_uses_only_observed_radius():
    h0=torch.tensor([[1.,-1.,0.,0.],[0.,0.,2.,-2.]])
    h=torch.stack((h0,h0.roll(1,-1),-h0))[None]
    mapped,details,report=d.support_map(h,h0,torch.ones(2,dtype=torch.bool))
    torch.testing.assert_close(mapped,h,rtol=0,atol=1e-7)
    assert report['H_change_max']<1e-7
    # Different future radius is not used to set the target radius.
    h[:,1:]*=4
    mapped,details,report=d.support_map(h,h0,torch.ones(2,dtype=torch.bool))
    torch.testing.assert_close(mapped[:,1:].norm(dim=-1),h0.norm(dim=-1)[None,None].expand(1,2,2))


def synthetic_saved(bundle,horizon):
    length=6;ri=torch.arange(length);ci=torch.zeros_like(ri);aa=torch.zeros_like(ri)
    mask=torch.ones(length,dtype=torch.bool)
    anchor=torch.stack((ri.float()*3.8,ri.float().sin(),ri.float().cos()),-1)
    rotation=torch.eye(3).expand(length,3,3)
    gen=torch.Generator().manual_seed(88)
    h0=torch.nn.functional.layer_norm(torch.randn(length,128,generator=gen),(128,))
    forecast=torch.randn(12,horizon+1,length,128,generator=gen)*2+2
    forecast[:,0]=h0
    true=h0[None,None].expand(1,horizon+1,-1,-1).clone()
    raw,p32=d.parent.decode_latents(bundle,forecast,aa,ri,ci,rotation,anchor,mask)
    reference=anchor.double()[None].expand(horizon+1,-1,-1).clone()
    roles=dict(zip(d.parent.ROLES,(reference[None],reference.float()[None],reference.float()[None],
                                  reference.float()[None],raw,p32)))
    return dict(forecast_H=forecast,true_H=true,roles=roles,mask=mask,aatype=aa,residue_index=ri,chain_id=ci,
        rotation=rotation,anchor=anchor,time_ps=torch.arange(horizon+1).double()*1000,
        absolute_time_ps=torch.arange(horizon+4).float()*1000,reference_unaligned=reference)


def production_bundle():
    from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        decoder=GeometricDeltaCADecoder().eval().requires_grad_(False)
    return SimpleNamespace(decoder=decoder,projection_steps=32)


@pytest.mark.parametrize('horizon',[1,8])
def test_real_decoder_baseline_parity_chunk_boundary_mapping_rng_and_saved_replay(horizon,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('optimizer created'))
    bundle=production_bundle();saved=synthetic_saved(bundle,horizon)
    initial=d.tree_sha(dict(bundle.decoder.state_dict()));rng_before=d.tree_sha(d.rng_snapshot())
    replay,report=d.diagnose_saved(bundle,saved,horizon)
    assert report['baseline_raw_bitwise_match'] and report['baseline_P32_bitwise_match']
    assert tuple(replay['roles'])==d.ROLES and report['rng_before_sha256']==report['rng_after_sha256']==rng_before
    assert initial==d.tree_sha(dict(bundle.decoder.state_dict()))
    assert all(p.grad is None for p in bundle.decoder.parameters())
    assert report['true_H_control']['H_change_max']<1e-6
    assert report['map']['H_change_max']>1.
    for value in replay['roles'].values():
        assert d.bitwise_equal(value[:,0],saved['anchor'][None].expand(12,-1,-1))
    for values in replay['coordinate_displacement_norm'].values():assert torch.equal(values[:,0],torch.zeros_like(values[:,0]))
    p=tmp_path/'synthetic.pt';torch.save(replay,p);loaded=torch.load(p,weights_only=False)
    assert d.tree_sha(loaded)==d.tree_sha(replay)
    assert d.score_new_roles(loaded['roles'],saved)==report['roles']


def test_untreated_mismatch_fails_before_map(monkeypatch):
    bundle=production_bundle();saved=synthetic_saved(bundle,1)
    saved['roles']['forecast_raw'][0,1,0,0]+=1.
    monkeypatch.setattr(d,'support_map',lambda *a:pytest.fail('map applied before baseline parity'))
    with pytest.raises(ValueError,match='baseline mismatch'):d.diagnose_saved(bundle,saved,1)


def test_decoder_only_construction_strict_identity_and_no_rng_change(monkeypatch):
    from protein_jepa_frame.utils.contracts import state_sha256
    original=production_bundle();state=copy.deepcopy(original.decoder.state_dict())
    checkpoint=dict(step=500,args=dict(coord_objective='marginal_path'),component_checkpoint=dict(
        decoder_mode='geometric_ca',calibration_args=dict(bond_projection_steps=32),delta_ca_decoder=state))
    audit=dict(component_identity=dict(decoder=state_sha256(state)))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('unrequested checkpoint read'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA queried'))
    before=d.tree_sha(d.rng_snapshot());loaded=d.load_decoder(checkpoint,audit)
    assert state_sha256(loaded.decoder)==state_sha256(original.decoder)
    assert d.tree_sha(d.rng_snapshot())==before and loaded.projection_steps==32
    checkpoint['component_checkpoint']['delta_ca_decoder']['gate.bias']+=1
    with pytest.raises(ValueError,match='decoder identity'):d.load_decoder(checkpoint,audit)


def test_displacement_matched_gauge_counts_and_invalid_geometry_retained():
    bundle=production_bundle();saved=synthetic_saved(bundle,1)
    roles={name:saved['roles']['forecast_raw'].clone() for name in d.ROLES}
    roles['support_raw'][:,1,1]=roles['support_raw'][:,1,0]
    result=d.score_new_roles(roles,saved)
    assert result['support_raw']['geometry_per_member_per_time'][0][1]['ca_bond_invalid_fraction']>0
    lag=result['support_raw']['ordered_metrics']['lag_statistics']['prediction']['aggregate'][-1]
    assert lag['n_pairs']==0 and lag['acf'] is None
    left=torch.zeros(2,3,6,3);right=left.clone();right[:,1:,:,0]=3
    values,report=d.displacement(left,right,saved['mask'])
    assert report['rms_per_member_time']==[[0.,3.,3.],[0.,3.,3.]]
    assert values.shape==(2,3,6)


def test_cpu_guard_precedes_checkpoint_and_cuda_query(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('checkpoint before CPU guard'))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('CUDA queried'))
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES empty'):d.execute({})


def test_unregistered_or_mutated_run_preserves_failure_before_load(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path);monkeypatch.setattr(d,'OUT',tmp_path/'out')
    monkeypatch.setattr(d,'execute',lambda *a:pytest.fail('model or sidecar read before verification'))
    with pytest.raises(ValueError,match='register before'):d.run()
    d.OUT.mkdir();(tmp_path/'source.py').write_text('changed')
    d.write_new(d.OUT/'protocol.json',dict(schema_version=d.VERSION,source_hashes={'source.py':'0'*64},input_hashes={}))
    with pytest.raises(ValueError,match='frozen file changed'):d.run()
    record=json.loads((d.OUT/'execution.json').read_text())
    assert record['status']=='failed' and record['exit_code']==1 and record['pid']>0
    with pytest.raises(ValueError,match='refusing rerun'):d.run()


def test_registration_only_metadata_hashes_never_tensor_load_hdf_or_parent_verify(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path);monkeypatch.setattr(d,'OUT',tmp_path/'out')
    monkeypatch.setattr(d,'SOURCES',('new.py',));(tmp_path/'new.py').write_text('new')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('tensor deserialized during register'))
    monkeypatch.setattr(d.parent,'verify_files',lambda *a:pytest.fail('parent HDF stat/read'))
    (tmp_path/'old.py').write_text('old');(tmp_path/'old_snapshot').mkdir();(tmp_path/'old_snapshot/old.py').write_text('old')
    ck=tmp_path/d.parent.CHECKPOINT;ck.parent.mkdir(parents=True,exist_ok=True);ck.write_text('synthetic checkpoint bytes')
    monkeypatch.setattr(d.parent,'CHECKPOINT_SHA',d.sha(ck))
    rows=[]
    for step,h in enumerate((1,4,16,64),1):
        p=tmp_path/f'fake{step}.pt';p.write_text('synthetic sidecar bytes')
        rows.append(dict(step=step,horizon=h,batch=dict(step=step),sidecar=p.name,sidecar_sha256=d.sha(p),replay_tensor_sha256='a'*64))
    protocol=dict(checkpoint=d.parent.CHECKPOINT,checkpoint_sha256=d.parent.CHECKPOINT_SHA,
        historical_actual_batches=[r['batch'] for r in rows],metadata_snapshots={},source_hashes={'old.py':d.sha('old.py')},
        source_snapshot='old_snapshot',checkpoint_audit={})
    result=dict(initial_model_sha256='m',final_model_sha256='m',rows=rows)
    hashes={}
    for filename,value in (('protocol.json',protocol),('result.json',result)):
        name=d.PARENT_ROOT+'/'+filename;p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);d.write_new(p,value);hashes[name]=d.sha(p)
    name=d.PARENT_ROOT+'/execution.json';d.write_new(tmp_path/name,dict(status='completed',exit_code=0,
        protocol_sha256=hashes[d.PARENT_ROOT+'/protocol.json'],output_sha256=hashes[d.PARENT_ROOT+'/result.json']))
    hashes[name]=d.sha(name);monkeypatch.setattr(d,'PARENT_FILES',hashes)
    d.register();registered=json.loads((d.OUT/'protocol.json').read_text());d.verify_files(registered)
    assert registered['new_rollouts']==registered['optimizer_steps']==0 and not registered['trajectory_files_opened']
    assert '2.2033' in registered['rationale'] and '2.5667' in registered['rationale']
    assert set(registered['input_hashes'])==set(hashes)|{d.parent.CHECKPOINT}|{r['sidecar'] for r in rows}
    with pytest.raises(ValueError,match='existing'):d.register()
    (tmp_path/rows[0]['sidecar']).write_text('changed')
    with pytest.raises(ValueError,match='frozen file changed'):d.verify_files(registered)


def test_sealed_paths_and_exclusive_nan_reject(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'REPO',tmp_path)
    with pytest.raises(ValueError):d.path('test.jsonl')
    with pytest.raises(ValueError):d.path(tmp_path.parent/'outside')
    p=tmp_path/'r.json';d.write_new(p,{})
    with pytest.raises(FileExistsError):d.write_new(p,{})
    with pytest.raises(ValueError):d.write_new(tmp_path/'bad.json',{'bad':float('nan')})
