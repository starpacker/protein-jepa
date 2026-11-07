"""Synthetic-only support/provenance tests; no real manifest/HDF/model reads."""
import copy
import json
from pathlib import Path

import pytest
import torch

from protein_jepa_frame.training import physical_distance_pair_support_v2 as support
from run_scripts import prepare_distance_pair_support_v2 as prepare


def uids(n):
    return [dict(segment='P', chain='A', number=i, insertion='') for i in range(n)]


def plan_fixture(path='/synthetic/protein.h5', protein='p'):
    n = 5
    full = dict(residue_uids=uids(n), geometry_row_ids=list(range(n)),
        peptide_edges=[[0,1],[1,2],[2,3],[3,4]], embedding_aatype=[1,2,3,4,5],
        embedding_residue_index=[47,48,49,49,50], embedding_chain_id=[0]*n,
        atom_indices=[[4*i+j for j in range(4)] for i in range(n)], atom_mask=[[True]*4]*n)
    full['residue_uids'][3] = dict(segment='P', chain='A', number=49, insertion='A')
    graph = support.build_pair_eligibility(range(n),full['residue_uids'],full['peptide_edges'])
    pairs = torch.nonzero(torch.triu(graph,diagonal=1)).tolist()
    plan = dict(record_key=support.record_key(path,protein), record_path=path, protein=protein,
        domain_id=protein,max_residues=96,hdf_stat=dict(size=1,mtime_ns=2),full_topology=full,
        full_topology_sha256=support.canonical_sha256(full),selected_geometry_row_ids=list(range(n)),
        pair_indices=pairs,pair_count=len(pairs),legacy_mapping={k:full[k] for k in (
            'embedding_aatype','embedding_residue_index','embedding_chain_id','atom_indices','atom_mask')})
    plan['record_plan_sha256']=support.canonical_sha256(plan)
    identity=dict(step=1,record_path=path,protein=protein,trajectory='t',horizon=1)
    document=dict(status='completed',errors=[],training_windows=[identity],records={plan['record_key']:plan})
    batch=dict(aatype=torch.tensor([full['embedding_aatype']]),
        residue_index=torch.tensor([full['embedding_residue_index']]),chain_id=torch.tensor([[0]*n]),
        lengths=torch.tensor([n]),protein_id=[protein],trajectory_id=['t'],
        residue_mask=torch.ones(1,5,n,dtype=torch.bool),atom_mask=torch.ones(1,5,n,4,dtype=torch.bool),
        coords=torch.randn(1,5,n,4,3))
    actual={**identity,'actual_batch_sha256':'a'*64,'file_stat':plan['hdf_stat']}
    return plan,document,batch,actual


def test_full_graph_before_crop_and_mask_and_permutation():
    full=support.build_pair_eligibility(range(6),uids(6),[[0,1],[1,2],[2,3],[3,4]])
    selected=torch.tensor([4,0,2,5])
    actual=support.build_pair_eligibility(selected,uids(6),[[0,1],[1,2],[2,3],[3,4]])
    assert torch.equal(actual,full[selected][:,selected])
    assert not actual[1,2]  # hidden middle node1 still makes0--2 local
    assert actual[0,1] and bool(actual[3,:3].all())  # four hops and disconnected
    assert not bool(actual.diag().any())
    observed=torch.tensor([True,False,True,True])
    assert torch.equal(actual[observed][:,observed],full[selected[observed]][:,selected[observed]])


def test_cross_chain_identity_does_not_override_explicit_connectivity():
    names=uids(4);names[1]['chain']='B'
    matrix=support.build_pair_eligibility(range(4),names,[[0,1],[1,2]])
    assert not matrix[0,1] and not matrix[0,2] and matrix[0,3]


@pytest.mark.parametrize('change',['uid_collision','selection_duplicate','selection_negative','selection_float','edge_self','edge_range','edge_duplicate','edge_float'])
def test_genuine_invalid_identity_or_graph_fails(change):
    names=uids(4);ids=[0,1,2,3];edges=[[0,1],[1,2]]
    if change=='uid_collision':names[1]=names[0]
    elif change=='selection_duplicate':ids=[0,0]
    elif change=='selection_negative':ids=[-1,0]
    elif change=='selection_float':ids=[0.,1.]
    elif change=='edge_self':edges=[[0,0]]
    elif change=='edge_range':edges=[[0,4]]
    elif change=='edge_duplicate':edges=[[0,1],[1,0]]
    else:edges=[[0.,1.]]
    with pytest.raises(ValueError):support.build_pair_eligibility(ids,names,edges)


def test_attachment_accepts_insertion_embedding_duplicates_preserves_every_old_object():
    plan,doc,batch,identity=plan_fixture()
    before=copy.deepcopy(identity)
    result,audit=support.attach_physical_support(batch,identity,doc)
    assert all(result[k] is v for k,v in batch.items()) and identity==before
    assert audit['pair_count']==3 and audit['record_plan_sha256']==plan['record_plan_sha256']
    assert len(audit['physical_metadata_sha256'])==64
    assert result['physical_pair_eligibility'].dtype==torch.bool
    assert result['geometry_row_ids'].tolist()==[[0,1,2,3,4]]


def test_padded_batch_preserves_sentinel_and_false_support():
    _,doc,batch,identity=plan_fixture()
    for k,pad in [('aatype',20),('residue_index',-1),('chain_id',-1)]:
        batch[k]=torch.cat((batch[k],torch.full((1,2),pad,dtype=torch.long)),1)
    batch['coords']=torch.cat((batch['coords'],torch.zeros(1,5,2,4,3)),2)
    batch['atom_mask']=torch.cat((batch['atom_mask'],torch.zeros(1,5,2,4,dtype=torch.bool)),2)
    batch['residue_mask']=torch.cat((batch['residue_mask'],torch.zeros(1,5,2,dtype=torch.bool)),2)
    result,_=support.attach_physical_support(batch,identity,doc)
    assert result['geometry_row_ids'][0,-2:].tolist()==[-1,-1]
    assert not bool(result['physical_pair_eligibility'][0,5:].any())
    assert not bool(result['physical_pair_eligibility'][0,:,5:].any())


@pytest.mark.parametrize('mutation',['aa','ri','ci','protein','trajectory','step','record','length','stat','plan','failed','duplicate_attach'])
def test_attachment_fails_closed(mutation):
    plan,doc,batch,identity=plan_fixture()
    if mutation in ('aa','ri','ci'):batch[{'aa':'aatype','ri':'residue_index','ci':'chain_id'}[mutation]][0,0]+=1
    elif mutation=='protein':batch['protein_id']=['other']
    elif mutation=='trajectory':batch['trajectory_id']=['other']
    elif mutation=='step':identity['step']=2
    elif mutation=='record':identity['record_path']='/other'
    elif mutation=='length':batch['lengths'][0]-=1
    elif mutation=='stat':identity['file_stat']={'size':9,'mtime_ns':2}
    elif mutation=='plan':plan['pair_indices'].pop()
    elif mutation=='failed':doc['status']='failed'
    else:batch['geometry_row_ids']=torch.arange(5)[None]
    with pytest.raises(ValueError):support.attach_physical_support(batch,identity,doc)


def test_plan_recomputation_rejects_rehashed_wrong_pair_rule():
    plan,_,_,_=plan_fixture();plan['pair_indices'].pop();plan['pair_count']-=1
    plan['record_plan_sha256']=support.canonical_sha256({k:v for k,v in plan.items() if k!='record_plan_sha256'})
    with pytest.raises(ValueError,match='pair support'):support.validate_record_plan(plan)


def test_every_record_error_is_retained_without_replacement(monkeypatch):
    plans=[dict(record_key=str(i),record_path=f'/p{i}',protein=str(i),domain_id=str(i),metadata={}) for i in range(4)]
    visited=[]
    def fake(record,max_residues):
        visited.append(record.protein_id)
        if record.protein_id in ['0','2']:raise ValueError('bad metadata')
        return {'ok':record.protein_id}
    monkeypatch.setattr(prepare,'support_record',fake)
    records,errors=prepare.execute_records(dict(record_plans=plans,max_residues=96))
    assert visited==['0','1','2','3'] and set(records)=={'1','3'}
    assert [e['record_key'] for e in errors]==['0','2']


@pytest.mark.parametrize('which',['input','source','snapshot','stat'])
def test_provenance_mutations_fail(tmp_path,which):
    inp=tmp_path/'input';src=tmp_path/'source';snap=tmp_path/'snapshot';hdf=tmp_path/'not_hdf'
    for p in (inp,src,hdf):p.write_text('frozen')
    # Absolute source path would escape the snapshot root; test realistic relative path.
    relative='synthetic.py';root=tmp_path/'root';root.mkdir();(root/relative).write_text('source')
    snap.mkdir();(snap/relative).write_text('source')
    import os
    old=Path.cwd();os.chdir(root)
    try:
        protocol=dict(input_hashes={str(inp):support.sha256_file(inp)},source_hashes={relative:support.sha256_file(relative)},source_snapshot=str(snap),selected_hdf5_stats={str(hdf):support.file_stat(hdf)})
        support.verify_provenance(protocol)
        {'input':inp,'source':root/relative,'snapshot':snap/relative,'stat':hdf}[which].write_text('changed contents')
        with pytest.raises(ValueError):support.verify_provenance(protocol)
    finally:os.chdir(old)


def test_failed_run_preserves_all_errors_and_forbids_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(prepare,'OUTPUT',tmp_path)
    prepare.write_new(tmp_path/'protocol.json',dict(schema_version=support.VERSION,record_plans=[{},{}],training_windows=[]))
    monkeypatch.setattr(prepare,'verify_provenance',lambda p:None)
    errors=[{'record_key':'a','error':'failure'},{'record_key':'b','error':'failure'}]
    monkeypatch.setattr(prepare,'execute_records',lambda p:({},errors))
    with pytest.raises(ValueError,match='preflight failed'):prepare.run()
    result=json.loads((tmp_path/'support.json').read_text());execution=json.loads((tmp_path/'execution.json').read_text())
    assert result['errors']==errors and result['status']=='failed'
    assert execution['returncode']==1 and execution['pid']>0 and execution['command']
    with pytest.raises(ValueError,match='already attempted'):prepare.run()


def test_exclusive_json_write(tmp_path):
    p=tmp_path/'file.json';prepare.write_new(p,{'original':True})
    with pytest.raises(FileExistsError):prepare.write_new(p,{'replace':True})
    assert json.loads(p.read_text())=={'original':True}


def test_postscan_provenance_failure_still_retains_record_errors(tmp_path,monkeypatch):
    monkeypatch.setattr(prepare,'OUTPUT',tmp_path)
    prepare.write_new(tmp_path/'protocol.json',dict(schema_version=support.VERSION,record_plans=[{}],training_windows=[]))
    calls=[]
    def verify(protocol):
        calls.append(True)
        if len(calls)==2:raise ValueError('input drift')
    monkeypatch.setattr(prepare,'verify_provenance',verify)
    errors=[{'record_key':'a','error':'bad topology'}]
    monkeypatch.setattr(prepare,'execute_records',lambda p:({},errors))
    with pytest.raises(ValueError,match='input drift'):prepare.run()
    execution=json.loads((tmp_path/'execution.json').read_text())
    assert execution['errors']==errors and execution['returncode']==1
    assert not (tmp_path/'support.json').exists()


def test_record_plans_deduplicate_shared_topology_without_losing_windows():
    from protein_jepa_frame.data.manifest import TrajectoryRecord
    records=[TrajectoryRecord(Path('/p.h5'),'p',t,format='mdcath-hdf5',temperature_k=300.,metadata={'atom_count':8}) for t in ('t1','t2')]
    windows=[dict(record_path='/p.h5',protein='p',trajectory=t,temperature_k=300.) for t in ('t1','t2')]
    plans=prepare.plan_records(windows,records)
    assert len(plans)==1 and len(windows)==2 and plans[0]['metadata']=={'atom_count':8}
    records[1]=TrajectoryRecord(Path('/p.h5'),'p','t2',format='mdcath-hdf5',temperature_k=300.,metadata={'atom_count':9})
    with pytest.raises(ValueError,match='conflicting'):prepare.plan_records(windows,records)


@pytest.mark.parametrize('mutation',[None,'hash','status','coverage','counts','window','execution','plan',
    'execution_protocol','execution_pid','execution_command','execution_start','execution_completion'])
def test_load_support_requires_complete_success_and_pinned_graph(tmp_path,monkeypatch,mutation):
    plan,doc,_,_=plan_fixture()
    windows=[{**doc['training_windows'][0],'step':step} for step in range(1,501)]
    protocol=dict(training_windows=windows,training_windows_sha256=support.canonical_sha256(windows),
        max_residues=96,selected_hdf5_stats={plan['record_path']:plan['hdf_stat']},
        record_plans=[{k:plan[k] for k in ('record_key','record_path','protein','domain_id')}])
    protocol_path=tmp_path/'protocol.json';prepare.write_new(protocol_path,protocol)
    doc.update(schema_version=support.VERSION,protocol_path=str(protocol_path),
        protocol_sha256=support.sha256_file(protocol_path),training_windows=windows,
        registered_record_count=1,completed_record_count=1)
    if mutation=='status':doc['status']='failed'
    elif mutation=='coverage':doc['records']={}
    elif mutation=='counts':doc['completed_record_count']=0
    elif mutation=='window':doc['training_windows']=windows[:-1]
    elif mutation=='plan':plan['pair_indices'].pop()
    path=tmp_path/'support.json';prepare.write_new(path,doc);digest=support.sha256_file(path)
    started=dict(pid=123,command=['prepare','--mode','run'],started_utc='2026-11-02T00:00:00+00:00',protocol_sha256=doc['protocol_sha256'])
    prepare.write_new(tmp_path/'started.json',started)
    execution=dict(started,status='failed' if mutation=='execution' else 'completed',
        returncode=0,support_sha256=digest,completed_utc='2026-11-02T00:00:01+00:00',seconds=1.)
    if mutation=='execution_protocol':execution['protocol_sha256']='0'*64
    elif mutation=='execution_pid':execution['pid']=124
    elif mutation=='execution_command':execution['command']=['other']
    elif mutation=='execution_start':execution['started_utc']='other'
    elif mutation=='execution_completion':execution.pop('completed_utc')
    prepare.write_new(tmp_path/'execution.json',execution)
    monkeypatch.setattr(support,'verify_provenance',lambda protocol:None)
    if mutation:
        with pytest.raises(ValueError):support.load_support(path,'0'*64 if mutation=='hash' else digest)
    else:assert support.load_support(path,digest)['completed_record_count']==1


def raw_line(serial, number, *, atom='CA', insertion='', altloc='', chain='A', segment='P0'):
    line=[' ']*80
    for start,stop,value in [(0,6,'ATOM  '),(6,11,f'{serial:5d}'),(12,16,f'{atom:>4}'),
        (16,17,altloc or ' '),(17,20,'ALA'),(21,22,chain),(22,26,f'{number:4d}'),
        (26,27,insertion or ' '),(72,76,f'{segment:<4}')]:line[start:stop]=value
    return ''.join(line)


def test_raw_identity_guard_keeps_insertions_and_legitimate_altlocs():
    text='\n'.join([raw_line(1,49,altloc='A'),raw_line(2,49,altloc='B'),
                    raw_line(3,49,insertion='A'),raw_line(4,50)])
    audit=support.validate_raw_pdb_identity(text)
    assert audit['full_uid_count']==3 and audit['raw_atom_count']==4
    assert audit['selected_backbone_atom_count']==3


@pytest.mark.parametrize('fault',['duplicate_atom','blank_and_A','repeated_block','duplicate_residue'])
def test_raw_identity_guard_rejects_silent_parser_merge(fault):
    if fault=='duplicate_atom':lines=[raw_line(1,1),raw_line(2,1)]
    elif fault=='blank_and_A':lines=[raw_line(1,1),raw_line(2,1,altloc='A')]
    elif fault=='repeated_block':lines=[raw_line(1,1),raw_line(2,2),raw_line(3,1,atom='N')]
    else:lines=[raw_line(i+1,number,atom=atom) for i,(number,atom) in enumerate(
        [(n,a) for n in [1,1,2,3,4] for a in ['N','CA','C','O']])]
    with pytest.raises(ValueError):support.validate_raw_pdb_identity('\n'.join(lines))


def test_support_record_guard_precedes_frozen_parser_and_never_reads_coordinates(tmp_path,monkeypatch):
    import h5py
    from protein_jepa_frame.data.manifest import TrajectoryRecord
    from protein_jepa_frame.data import backbone_topology_v2 as frozen
    path=tmp_path/'synthetic.h5'
    with h5py.File(path,'w') as f:
        g=f.create_group('p');g.create_dataset('pdbProteinAtoms',data=(raw_line(1,1)+'\n'+raw_line(2,1)).encode())
        g.create_dataset('psf',data=b'not reached');g.create_dataset('resid',data=[1,1])
    def forbidden(*args,**kwargs):raise AssertionError('frozen parser called before raw collision guard')
    monkeypatch.setattr(frozen,'parse_mdcath_backbone_topology',forbidden)
    record=TrajectoryRecord(path,'p','t',format='mdcath-hdf5')
    with pytest.raises(ValueError,match='duplicate raw atom'):support.support_record(record)


@pytest.mark.parametrize('duplicate',[False,True])
def test_metadata_only_record_preserves_parity_and_blocks_actual_frozen_merge(tmp_path,duplicate):
    import h5py
    from protein_jepa_frame.data.manifest import TrajectoryRecord
    from protein_jepa_frame.data.backbone_topology_v2 import parse_mdcath_backbone_topology
    identities=[(1,''),(1,'' if duplicate else 'A'),(2,''),(3,''),(4,'')]
    pdb=[];atoms=[]
    for row,(number,insertion) in enumerate(identities):
        for col,atom in enumerate(('N','CA','C','O')):
            serial=4*row+col+1
            pdb.append(raw_line(serial,number,atom=atom,insertion=insertion))
            atoms.append(f'{serial} P0 {number}{insertion} ALA {atom} TYPE 0.0 12.0 0')
    pdb='\n'.join(pdb)+'\n'
    psf='PSF\n\n20 !NATOM\n'+'\n'.join(atoms)+'\n\n4 !NBOND\n'+\
        '\n'.join(f'{4*i+3} {4*(i+1)+1}' for i in range(4))+'\n\n0 !NTHETA\n'
    frozen=parse_mdcath_backbone_topology(pdb,psf,source_path='synthetic',domain_id='p',expected_atom_count=20)
    assert frozen.n_residues==(4 if duplicate else 5)  # confirmed old-parser merge
    path=tmp_path/'metadata_only.h5'
    with h5py.File(path,'w') as f:
        g=f.create_group('p');g.create_dataset('pdbProteinAtoms',data=pdb.encode())
        g.create_dataset('psf',data=psf.encode());g.create_dataset('resid',data=[1]*20)
    record=TrajectoryRecord(path,'p','t',format='mdcath-hdf5')
    if duplicate:
        with pytest.raises(ValueError,match='duplicate raw atom'):support.support_record(record)
    else:
        plan=support.support_record(record)
        assert plan['pair_count']==3 and plan['selected_geometry_row_ids']==list(range(5))
        assert plan['legacy_mapping']['embedding_residue_index']==[1,1,2,3,4]
        assert plan['full_topology']['source_identity']['raw_identity_audit']['full_uid_count']==5
        assert torch.equal(support.validate_record_plan(plan),torch.tensor([
            [False,False,False,True,True],[False,False,False,False,True],
            [False,False,False,False,False],[True,False,False,False,False],
            [True,True,False,False,False]]))
