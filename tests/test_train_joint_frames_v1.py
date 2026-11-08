"""Synthetic-only driver tests; never read real caches, checkpoints or MD."""
import copy
import json
from pathlib import Path
import sys
import types

import pytest
import torch

from run_scripts import diagnose_train_joint_frames_v1 as driver


def synthetic_row():
    g=torch.Generator().manual_seed(19)
    n=6
    ca=torch.stack((torch.arange(n)*3.8,torch.sin(torch.arange(n).float()),torch.cos(torch.arange(n).float())),dim=-1)[None]
    offsets=torch.tensor([[-.8,.9,.2],[0.,0.,0.],[1.2,.3,-.1],[1.8,1.,.5]])
    bb=ca[:,:,None]+offsets
    return dict(metadata=dict(protein='synthetic',trajectory='t',temperature_k=300.,anchor_frame=7,
        horizon=1,target_frame=8,sampling_seed=None),observed_mask=torch.ones(n,dtype=torch.bool),
        atom_mask=torch.ones(n,4,dtype=torch.bool),target_atom_mask=torch.ones(n,4,dtype=torch.bool),
        embedding_aatype=torch.zeros(n,dtype=torch.long),embedding_residue_index=torch.arange(n),
        embedding_chain_id=torch.zeros(n,dtype=torch.long),chemical_aatype=torch.zeros(n,dtype=torch.long),
        geometry_row_ids=torch.arange(n),peptide_edges=torch.tensor([[i,i+1] for i in range(n-1)]),
        observed_R=torch.eye(3).expand(1,n,3,3).clone(),observed_backbone=bb.clone(),
        target_backbone=bb.clone(),oracle_CA=ca+torch.tensor([0.,.2,0.]),
        oracle_H=torch.randn(1,n,128,generator=g))


@pytest.fixture
def registered_inputs(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'OUTPUT','out')
    for key,value in [('CACHE','cache.pt'),('CHECKPOINT','head.pt'),('SELECTION','cache.selection.json'),
                      ('CACHE_PROTOCOL','cache.protocol.json')]:monkeypatch.setattr(driver,key,value)
    for filename in ['cache.pt','head.pt','synthetic.py']:(tmp_path/filename).write_bytes(b'synthetic bytes only')
    monkeypatch.setattr(driver,'CACHE_SHA',driver.sha(tmp_path/'cache.pt'))
    monkeypatch.setattr(driver,'CHECKPOINT_SHA',driver.sha(tmp_path/'head.pt'))
    monkeypatch.setattr(driver,'SOURCES',('synthetic.py',))
    rows=[dict(protein=f'p{i}',horizon=[1,4,16,64][i%4]) for i in range(500)]
    selection=dict(schema_version='backbone_readout_cache_v1',mode='train',seed=42,n_windows=500,
        selected_prefix=500,full_selection=[dict(metadata=r) for r in rows])
    driver.write_json(tmp_path/driver.SELECTION,selection)
    p=dict(mode='train',dataset_role='training_readout_calibration',n_windows=500,
        selection_file=driver.SELECTION,selection_sha256=driver.sha(tmp_path/driver.SELECTION))
    driver.write_json(tmp_path/driver.CACHE_PROTOCOL,dict(protocol=p))
    return selection,dict(protocol=p)


def test_registration_metadata_only_fixed32_and_exclusive(registered_inputs,monkeypatch):
    monkeypatch.setattr(driver.torch,'load',lambda *a,**k:pytest.fail('deserialized at registration'))
    monkeypatch.setattr(driver,'BackboneReadoutV2',lambda *a,**k:pytest.fail('constructed model at registration'))
    protocol=driver.register()
    assert len(protocol['selected_metadata'])==32
    assert protocol['horizon_counts']=={'1':8,'4':8,'16':8,'64':8}
    assert protocol['roles']==list(driver.ROLES) and len(driver.ROLES)==5
    assert protocol['budget']['steps']==200 and protocol['budget']['selection']=='final only'
    assert protocol['objective']['periodic_phi_psi']==1.
    execution=json.loads((driver.ROOT/'out/registration.execution.json').read_text())
    assert execution['returncode']==0 and execution['pid']>0
    driver.verify(protocol)
    with pytest.raises(FileExistsError):driver.register()


@pytest.mark.parametrize('bad',['duplicate','horizon','prefix','selection_link'])
def test_fixed_prefix_fails_without_replacement(registered_inputs,bad):
    selection,protocol=copy.deepcopy(registered_inputs)
    if bad=='duplicate':selection['full_selection'][1]['metadata']['protein']='p0'
    elif bad=='horizon':selection['full_selection'][0]['metadata']['horizon']=64
    elif bad=='prefix':selection['selected_prefix']=32
    else:protocol['protocol']['selection_sha256']='0'*64
    with pytest.raises(ValueError):driver.selected_metadata(selection,protocol)


@pytest.mark.parametrize('changed',['input','source','snapshot'])
def test_run_hash_failure_before_deserialize_records_real_failure(registered_inputs,monkeypatch,changed):
    driver.register()
    target={'input':'cache.pt','source':'synthetic.py','snapshot':'out/source_snapshot/synthetic.py'}[changed]
    (driver.ROOT/target).write_text('drift')
    monkeypatch.setattr(driver.torch,'load',lambda *a,**k:pytest.fail('loaded before provenance verification'))
    with pytest.raises(ValueError,match='changed'):driver.run()
    execution=json.loads((driver.ROOT/'out/execution.json').read_text())
    assert execution['status']=='failed' and execution['returncode']==1 and execution['pid']>0
    assert not (driver.ROOT/'out/result.json').exists()
    with pytest.raises(ValueError,match='already attempted'):driver.run()


def test_cpu_guard_does_not_query_cuda_or_write(monkeypatch,tmp_path):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('queried CUDA'))
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):driver.register()
    assert not list(tmp_path.iterdir())


def test_row_support_fixed_and_missing_future_fails():
    row=synthetic_row()
    assert torch.equal(driver.validate_row(row),row['atom_mask'])
    row['target_atom_mask'][2,0]=False
    with pytest.raises(ValueError,match='complete fixed'):driver.validate_row(row)


def test_explicit_angles_omega_keep_masked_and_degenerate_denominators():
    row=synthetic_row();mask=driver.validate_row(row)
    x=row['target_backbone'].double()
    base=driver.explicit_edge_geometry(x,row,mask)
    assert base['counts']['omega']['n_edges']==5 and base['counts']['omega']['n_valid']==5
    changed=x.clone();changed[0,1,0]=changed[0,0,2]  # collapse central C-N
    result=driver.explicit_edge_geometry(changed,row,mask)
    assert result['counts']['omega']['n_edges']==5 and result['counts']['omega']['n_degenerate']==1
    mask[0,2]=False
    result=driver.explicit_edge_geometry(x,row,mask)
    assert result['counts']['omega']['n_masked']==1 and result['counts']['c_n']['n_masked']==1


def test_explicit_omega_reports_cis_and_trans_not_only_trans_validity():
    row=synthetic_row();row['peptide_edges']=torch.tensor([[0,1]])
    x=row['target_backbone'].double();mask=driver.validate_row(row)
    x[0,0,1]=torch.tensor([0.,1.,0.]);x[0,0,2]=torch.tensor([0.,0.,0.])
    x[0,1,0]=torch.tensor([1.,0.,0.]);x[0,1,1]=torch.tensor([1.,1.,0.])
    cis=driver.explicit_edge_geometry(x,row,mask)
    assert cis['counts']['omega']['cis_count']==1
    x[0,1,1]=torch.tensor([1.,-1.,0.])
    trans=driver.explicit_edge_geometry(x,row,mask)
    assert trans['counts']['omega']['trans_count']==1


def test_official_scoring_keeps_all_six_classes_coverage_and_exact_shared_mask():
    row=synthetic_row();bb=row['target_backbone'].double()
    value=driver.score(bb,row,'target_md',bb[...,1,:])
    rama=value['ramachandran']
    assert set(rama['per_sample'][0]['per_class'])==set(driver.rama.CLASSES)
    assert rama['totals']['n_scored']+rama['totals']['n_excluded']==6
    assert rama['per_sample'][0]['exclusion_counts']['topology_incomplete']==2
    assert 'explicit_geometry' in value and 'error' in value and value['fixed_ca_bitwise_equal']
    assert value['nco_and_geometry']['coordinate_rmsd_NCO_angstrom']==0.
    assert value['score_mask_sha256']==driver.state_sha256({'mask':row['atom_mask']})


def test_real_random_head_decodes_once_with_exact_ca_and_frozen_global_rng():
    from protein_jepa_frame.models.backbone_readout_v2 import BackboneReadoutV2,BackboneReadoutConfig
    row=synthetic_row()
    head=BackboneReadoutV2(BackboneReadoutConfig(c_hidden=16,n_heads=2,n_ipa_blocks=1,dropout=0.)).eval().requires_grad_(False)
    before=driver.state_sha256(head);rng=torch.get_rng_state().clone();calls=[]
    hook=head.register_forward_hook(lambda *args:calls.append(True))
    output,audit=driver.decode_condition(head,row,row['oracle_CA'])
    hook.remove()
    assert len(calls)==1 and torch.equal(output['backbone'][...,1,:],row['oracle_CA'])
    assert {'R','psi_logits','torsions','backbone'} <= set(output)
    assert torch.equal(torch.get_rng_state(),rng) and driver.state_sha256(head)==before
    assert all(p.grad is None for p in head.parameters()) and audit['frame_seed_sha256']


def install_fake_optimizer(monkeypatch,fail=False):
    calls=[]
    def optimize(initial,target,**kw):
        calls.append((initial,target,kw))
        result=dict(initial_backbone=initial.clone(),final_backbone=initial.clone(),
            psi_logits=kw['psi_logits'].clone(),trace=[{'step':s} for s in range(201)])
        if fail:
            error=RuntimeError('synthetic partial optimization')
            error.partial={**result,'trace':result['trace'][:7]}
            raise error
        return result
    module=types.ModuleType('protein_jepa_frame.models.stage_c.joint_frame_oracle_v1')
    module.optimize_joint_frames=optimize
    monkeypatch.setitem(sys.modules,module.__name__,module)
    return calls


def fake_decode(calls):
    def decode(head,row,ca):
        calls.append(ca.clone())
        bb=row['target_backbone'].clone();bb[...,1,:]=ca
        return dict(backbone=bb,R=row['observed_R'].clone(),psi_logits=torch.zeros(1,len(ca[0]),36),
                    torsions=torch.zeros(1,len(ca[0]),7)),{'synthetic':True}
    return decode


def test_five_roles_two_head_calls_fixed_support_and_full_witnesses(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    calls=[];monkeypatch.setattr(driver,'decode_condition',fake_decode(calls))
    optimizations=install_fake_optimizer(monkeypatch)
    row=synthetic_row();record=driver.diagnose_row(None,row,0,tmp_path)
    assert not record['errors'] and set(record['roles'])==set(driver.ROLES)
    assert len(calls)==len(optimizations)==2
    assert torch.equal(calls[0],row['oracle_CA']) and torch.equal(calls[1],row['target_backbone'][...,1,:])
    assert len({r['score_mask_sha256'] for r in record['roles'].values()})==1
    for condition in driver.CONDITIONS:
        artifact=record['artifacts'][condition]
        assert driver.sha(tmp_path/artifact['path'])==artifact['sha256']
        saved=torch.load(tmp_path/artifact['path'],weights_only=False)
        assert len(saved['optimization']['trace'])==201 and saved['conditioned_on_reference']
    with pytest.raises(FileExistsError):driver.diagnose_row(None,row,0,tmp_path)


def test_failed_conditions_preserve_initial_partial_trace_and_do_not_replace(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    calls=[];monkeypatch.setattr(driver,'decode_condition',fake_decode(calls))
    install_fake_optimizer(monkeypatch,fail=True)
    row=synthetic_row();record=driver.diagnose_row(None,row,3,tmp_path)
    assert len(calls)==2 and len(record['errors'])==2
    assert set(record['roles'])=={'target_md',*(c+'_baseline' for c in driver.CONDITIONS)}
    for condition in driver.CONDITIONS:
        saved=torch.load(tmp_path/record['artifacts'][condition]['path'],weights_only=False)
        assert saved['status']=='failed' and 'head_output' in saved
        assert len(saved['optimization']['trace'])==7


def test_registration_pin_failure_is_preserved(registered_inputs):
    (driver.ROOT/driver.CHECKPOINT).write_text('wrong')
    with pytest.raises(ValueError,match='pinned'):driver.register()
    result=json.loads((driver.ROOT/'out/registration.execution.json').read_text())
    assert result['status']=='failed' and result['returncode']==1
    assert not (driver.ROOT/'out/protocol.json').exists()


def test_all32_attempted_and_failed_row_witness_retained(registered_inputs,monkeypatch):
    driver.register()
    head=torch.nn.Linear(1,1).eval().requires_grad_(False)
    rows=[{'metadata':{'protein':f'p{i}'}} for i in range(32)]
    monkeypatch.setattr(driver.torch,'load',lambda *a,**k:{'synthetic':True})
    monkeypatch.setattr(driver,'validate_cache',lambda *a:rows)
    monkeypatch.setattr(driver,'load_head',lambda:head)
    visited=[]
    def diagnose(head,row,index,out):
        visited.append(index)
        if index==2:raise ValueError('synthetic invalid fixed row')
        return dict(index=index,metadata=row['metadata'],errors=[],roles={r:{} for r in driver.ROLES})
    monkeypatch.setattr(driver,'diagnose_row',diagnose)
    with pytest.raises(ValueError,match='preserved failed'):driver.run()
    assert visited==list(range(32))
    result=json.loads((driver.ROOT/'out/result.json').read_text())
    assert result['status']=='failed' and result['n_attempted_windows']==32 and result['n_successful_windows']==31
    artifact=result['per_window'][2]['artifact']
    assert driver.sha(driver.ROOT/artifact['path'])==artifact['sha256']
    assert result['cpu_rng_before_sha256']==result['cpu_rng_after_sha256']


def test_real_random_head_to_fixed200_optimizer_api_preserves_ca_psi_and_rng():
    from protein_jepa_frame.models.backbone_readout_v2 import BackboneReadoutV2,BackboneReadoutConfig
    from protein_jepa_frame.models.stage_c.joint_frame_oracle_v1 import optimize_joint_frames
    row=synthetic_row()
    head=BackboneReadoutV2(BackboneReadoutConfig(c_hidden=16,n_heads=2,n_ipa_blocks=1,dropout=0.)).eval().requires_grad_(False)
    output,_=driver.decode_condition(head,row,row['oracle_CA'])
    rng=torch.get_rng_state().clone();before=driver.state_sha256(head)
    result=optimize_joint_frames(output['backbone'].double(),row['target_backbone'].double(),
        residue_mask=row['observed_mask'],atom_mask=driver.validate_row(row),
        chemical_aatype=row['chemical_aatype'],geometry_row_ids=row['geometry_row_ids'],
        peptide_edges=row['peptide_edges'],initial_R=output['R'].double(),psi_logits=output['psi_logits'])
    assert result['status']=='completed' and [r['step'] for r in result['trace']]==list(range(201))
    assert torch.equal(result['initial_backbone'],output['backbone'].double())
    assert torch.equal(result['final_backbone'][...,1,:],output['backbone'].double()[...,1,:])
    assert torch.equal(result['psi_logits'],output['psi_logits'])
    assert float(torch.linalg.vector_norm(result['rotation_vectors'],dim=-1).max())<=torch.pi/6+1e-14
    assert torch.equal(torch.get_rng_state(),rng) and driver.state_sha256(head)==before
    assert all(p.grad is None for p in head.parameters())
