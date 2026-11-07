"""Synthetic metadata/coordinate fixtures only; no actual cache/model/HDF IO."""
import copy
import json
from pathlib import Path

import pytest
import torch

from protein_jepa_frame.data import frozen_h_distance_support_v2 as support
from protein_jepa_frame.training import physical_distance_pair_support_v2 as old
from protein_jepa_frame.data.backbone_topology_v2 import CHEMICAL_ALIAS_POLICY
from run_scripts import prepare_frozen_h_distance_support_v2 as prepare


def fixture(index=0,path='/synthetic/p.h5'):
    protein=f'p{index}';n=5;h=[1,4,16,64][index%4]
    identity=dict(path=path,protein_id=protein,domain_id=protein,
                  chemical_alias_policy=copy.deepcopy(CHEMICAL_ALIAS_POLICY))
    full=dict(schema_version='synthetic',source_identity={**identity,'raw_identity_audit':{'ok':True}},
        metadata_hashes={'pdb':'a'*64,'psf':'b'*64},residue_uids=[dict(number=i,insertion='') for i in range(n)],
        residue_names=['ALA']*n,embedding_aatype=[1,2,3,4,5],embedding_residue_index=[47,48,49,49,50],
        embedding_chain_id=[0]*n,chemical_aatype=[0]*n,geometry_row_ids=list(range(n)),
        atom_indices=[[4*i+j for j in range(4)] for i in range(n)],atom_mask=[[True]*4 for i in range(n)],
        peptide_edges=[[0,1],[1,2],[2,3],[3,4]],peptide_atom_indices=[[4*i+2,4*(i+1)] for i in range(4)])
    full['residue_uids'][3]['insertion']='A'
    crop=copy.deepcopy(full);crop['source_identity']=identity
    metadata=dict(protein=protein,trajectory=protein+'_T320_R1',temperature_k=320.,anchor_frame=7,
        horizon=h,target_frame=7+h,horizon_ps=h*1000.,context_frames=list(range(8)),sampling_seed=None)
    historical=dict(metadata=metadata,source_identity=support.canonical_metadata(identity),metadata_hashes=full['metadata_hashes'],
        topology_sha256=support.canonical_sha256(crop))
    record=dict(path=path,protein_id=protein,trajectory_id=metadata['trajectory'],format='mdcath-hdf5',
        num_frames=100,frame_interval_ps=1000.,first_frame_time_ps=0.,temperature_k=320.,
        replica_id='1',dataset='mdCATH',domain_id=protein,metadata={})
    window=dict(cache_row_index=index,split='fit' if index<400 else 'readout_heldout',metadata=metadata,
        record=record,record_key=old.record_key(path,protein),historical_topology=historical)
    graph=old.build_pair_eligibility(range(n),full['residue_uids'],full['peptide_edges'])
    pairs=torch.nonzero(torch.triu(graph,diagonal=1)).tolist()
    rs=dict(record_key=window['record_key'],record_path=path,protein=protein,domain_id=protein,max_residues=96,
        hdf_stat={'size':1,'mtime_ns':2},full_topology=full,full_topology_sha256=support.canonical_sha256(full),
        selected_geometry_row_ids=list(range(n)),pair_indices=pairs,pair_count=len(pairs),
        legacy_mapping={k:full[k] for k in ('embedding_aatype','embedding_residue_index','embedding_chain_id','atom_indices','atom_mask')})
    rs['record_plan_sha256']=support.canonical_sha256(rs)
    plan=support.make_plan(window,rs)
    row=dict(metadata=copy.deepcopy(metadata),topology=dict(sha256=support.canonical_sha256(crop),**copy.deepcopy(crop)),
        observed_mask=torch.ones(n,dtype=torch.bool),target_atom_mask=torch.ones(n,4,dtype=torch.bool),
        observed_backbone=torch.arange(n*12,dtype=torch.float32).reshape(1,n,4,3),
        target_backbone=torch.arange(n*12,dtype=torch.float32).reshape(1,n,4,3)+.25)
    for k in ('embedding_aatype','embedding_residue_index','embedding_chain_id','chemical_aatype','geometry_row_ids','peptide_edges','atom_mask'):
        row[k]=torch.tensor(crop[k],dtype=torch.bool if k=='atom_mask' else torch.long)
    return window,plan,row


def selection_fixture():
    rows=[];topologies=[]
    for i in range(500):
        window,_,_=fixture(i)
        rows.append(dict(metadata=window['metadata'],record=window['record'],record_sha256=support.canonical_sha256(window['record'])))
        topologies.append(window['historical_topology'])
    selection=dict(schema_version='backbone_readout_cache_v1',mode='train',seed=42,n_windows=500,selected_prefix=500,full_selection=rows)
    cp=dict(protocol=dict(mode='train',n_windows=500,dataset_role='training_readout_calibration',max_residues=96,n_forecast_samples=0),provenance=dict(topology_metadata=topologies))
    return selection,cp


def test_registered500_whole_proteins_and_balanced_readout_holdout():
    selection,cp=selection_fixture();windows=support.registered_windows(selection,cp)
    assert len(windows)==500 and windows[399]['split']=='fit' and windows[400]['split']=='readout_heldout'
    assert [w['metadata']['horizon'] for w in windows[400:]]==[1,4,16,64]*25


@pytest.mark.parametrize('mutation',['duplicate_protein','short_panel','horizon','record_hash','wrong_target','topology_identity'])
def test_bad_registered_panel_fails(mutation):
    selection,cp=selection_fixture()
    if mutation=='duplicate_protein':selection['full_selection'][1]['metadata']['protein']='p0'
    elif mutation=='short_panel':selection['full_selection'].pop()
    elif mutation=='horizon':selection['full_selection'][1]['metadata']['horizon']=1
    elif mutation=='record_hash':selection['full_selection'][1]['record_sha256']='x'*64
    elif mutation=='wrong_target':selection['full_selection'][1]['metadata']['target_frame']+=1
    else:cp['provenance']['topology_metadata'][1]['source_identity']['path']='/wrong'
    with pytest.raises(ValueError):support.registered_windows(selection,cp)


def test_runtime_preserves_inputs_and_legacy_insertion_ids_uses_all_pairs():
    _,plan,row=fixture();before={k:v.clone() for k,v in row.items() if isinstance(v,torch.Tensor)}
    result=support.validate_cached_row(row,plan)
    assert result['pair_indices'].tolist()==[[0,3],[0,4],[1,4]]
    assert result['audit']['pair_count']==3 and result['audit']['observed_residue_count']==5
    assert all(torch.equal(row[k],v) for k,v in before.items())
    assert result['observed_mask'] is not row['observed_mask']


def test_masked_intermediate_stays_local_and_no_distance_filtering_or_h_reads():
    _,plan,row=fixture();row['observed_mask'][1]=False
    row['observed_backbone'][:,1]=float('nan');row['target_backbone'][:,1]=float('nan')
    row['oracle_H']=object();row['anchor_H']=object()
    result=support.validate_cached_row(row,plan)
    assert result['pair_indices'].tolist()==[[0,3],[0,4]] and [0,2] not in result['pair_indices'].tolist()
    row['observed_backbone'][:,[0,2,3,4]]=0;row['target_backbone'][:,[0,2,3,4]]=0
    assert torch.equal(result['pair_indices'],support.validate_cached_row(row,plan)['pair_indices'])


def test_full_graph_hidden_beyond_prefix96_preserves_two_hop_exclusion():
    names=[{'id':i} for i in range(97)]
    graph=old.build_pair_eligibility(range(96),names,[[0,96],[96,2]])
    assert not graph[0,2] and graph[0,3]


@pytest.mark.parametrize('mutation',['window','aa','ri','ci','chemical','geometry_ids','edges','atom_map','topology_hash',
                                   'target_mask','nonfinite_observed','nonfinite_target','dtype','plan_hash','rehashed_pairs'])
def test_runtime_mutations_fail_closed(mutation):
    _,plan,row=fixture()
    if mutation=='window':row['metadata']['horizon']=64
    elif mutation in ('aa','ri','ci','chemical','geometry_ids'):
        field={'aa':'embedding_aatype','ri':'embedding_residue_index','ci':'embedding_chain_id','chemical':'chemical_aatype','geometry_ids':'geometry_row_ids'}[mutation];row[field][0]+=1
    elif mutation=='edges':row['peptide_edges'][0,0]=4
    elif mutation=='atom_map':row['topology']['atom_indices'][0][0]+=1
    elif mutation=='topology_hash':row['topology']['sha256']='x'*64
    elif mutation=='target_mask':row['target_atom_mask'][0,0]=False
    elif mutation=='nonfinite_observed':row['observed_backbone'][0,0,0,0]=float('nan')
    elif mutation=='nonfinite_target':row['target_backbone'][0,0,0,0]=float('inf')
    elif mutation=='dtype':row['target_backbone']=row['target_backbone'].double()
    elif mutation=='plan_hash':plan['plan_sha256']='x'*64
    else:
        rs=plan['record_support'];rs['pair_indices'].pop();rs['pair_count']-=1
        rs['record_plan_sha256']=support.canonical_sha256({k:v for k,v in rs.items() if k!='record_plan_sha256'})
        plan['plan_sha256']=support.canonical_sha256({k:v for k,v in plan.items() if k!='plan_sha256'})
    with pytest.raises(ValueError):support.validate_cached_row(row,plan)


def test_rebuilt_cache_topology_must_match_old_parser_hash():
    window,plan,_=fixture();window['historical_topology']['topology_sha256']='c'*64
    with pytest.raises(ValueError,match='prefix topology'):support.make_plan(window,plan['record_support'])


def test_preflight_visits_all_records_and_retains_every_error(monkeypatch):
    windows=[fixture(i)[0] for i in range(4)];visited=[]
    def fake(record,_):
        index=int(record.protein_id[1:]);visited.append(index)
        if index in (0,2):raise ValueError('synthetic invalid topology')
        return fixture(index)[1]['record_support']
    monkeypatch.setattr(prepare,'support_record',fake)
    plans,errors=prepare.execute_records(dict(windows=windows,max_residues=96))
    assert visited==[0,1,2,3] and [p['cache_row_index'] for p in plans]==[1,3]
    assert [e['cache_row_index'] for e in errors]==[0,2]


@pytest.mark.parametrize('drift',[False,True])
def test_failed_run_preserves_errors_and_disallows_retry(tmp_path,monkeypatch,drift):
    monkeypatch.setattr(prepare,'OUTPUT',tmp_path)
    prepare.write_new(tmp_path/'protocol.json',dict(schema_version=support.VERSION,windows=[{}]))
    calls=[]
    def verify(_):
        calls.append(True)
        if drift and len(calls)==2:raise ValueError('postscan drift')
    monkeypatch.setattr(prepare,'verify_provenance',verify)
    errors=[dict(cache_row_index=0,error='synthetic failure')]
    monkeypatch.setattr(prepare,'execute_records',lambda _: ([],errors))
    with pytest.raises(ValueError):prepare.run()
    execution=json.loads((tmp_path/'execution.json').read_text())
    assert execution['status']=='failed' and execution['returncode']==1 and execution['errors']==errors
    if not drift:assert json.loads((tmp_path/'support.json').read_text())['errors']==errors
    with pytest.raises(ValueError,match='already attempted'):prepare.run()


def successful_support(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    hdf=tmp_path/'metadata.bin';hdf.write_bytes(b'x')
    cache=tmp_path/'opaque_cache.bin';cache.write_bytes(b'not a torch file')
    Path('source.py').write_text('source');Path('snapshot').mkdir();Path('snapshot/source.py').write_text('source')
    stat=support.full_file_stat(hdf);windows=[];plans=[]
    for i in range(500):
        w,p,_=fixture(i,str(hdf));rs=p['record_support'];rs['hdf_stat']={k:stat[k] for k in ('size','mtime_ns')}
        rs['record_plan_sha256']=support.canonical_sha256({k:v for k,v in rs.items() if k!='record_plan_sha256'})
        windows.append(w);plans.append(support.make_plan(w,rs))
    protocol=dict(schema_version=support.VERSION,windows=windows,windows_sha256=support.canonical_sha256(windows),
        max_residues=96,n_records=500,n_distinct_proteins=500,
        fit_indices=list(range(400)),readout_heldout_indices=list(range(400,500)),
        training_order=list(range(400))+list(range(100)),input_hashes={str(cache):support.sha256_file(cache)},
        source_hashes={'source.py':support.sha256_file('source.py')},source_snapshot='snapshot',
        selected_hdf5_stats={str(hdf):stat})
    prepare.write_new('protocol.json',protocol);digest=support.sha256_file('protocol.json')
    doc=dict(schema_version=support.VERSION,status='completed',errors=[],protocol_path='protocol.json',protocol_sha256=digest,
        plans=plans,registered_record_count=500,completed_record_count=500,
        trajectory_coordinates_loaded=False,cache_deserialized=False,model_loaded=False,validation_loaded=False,test_opened=False)
    prepare.write_new('support.json',doc);sha=support.sha256_file('support.json')
    started=dict(pid=123,command=['synthetic','--mode','run'],started_utc='recorded',protocol_sha256=digest)
    prepare.write_new('started.json',started)
    execution=dict(**started,status='completed',returncode=0,support_sha256=sha,completed_utc='recorded_later',seconds=1.,errors=[])
    prepare.write_new('execution.json',execution)
    return doc,sha


def test_completed500_load_and_attachment(tmp_path,monkeypatch):
    _,sha=successful_support(tmp_path,monkeypatch);doc=support.load_support('support.json',sha)
    _,_,row=fixture(400,str(tmp_path/'metadata.bin'))
    result=support.attach_cached_physical_support(row,400,doc)
    assert result['audit']['split']=='readout_heldout' and result['pair_indices'].shape==(3,2)


@pytest.mark.parametrize('mutation',['pid','protocol','command','completion','status','support_sha','short_panel','source','snapshot','input','hdf_stat'])
def test_support_load_execution_provenance_and_coverage_fail(tmp_path,monkeypatch,mutation):
    doc,sha=successful_support(tmp_path,monkeypatch)
    if mutation in ('source','snapshot','input','hdf_stat'):
        path={'source':'source.py','snapshot':'snapshot/source.py','input':'opaque_cache.bin','hdf_stat':'metadata.bin'}[mutation]
        Path(path).write_text('changed')
    elif mutation=='short_panel':
        doc['plans'].pop();Path('support.json').write_text(json.dumps(doc));sha=support.sha256_file('support.json')
        ex=json.loads(Path('execution.json').read_text());ex['support_sha256']=sha;Path('execution.json').write_text(json.dumps(ex))
    else:
        ex=json.loads(Path('execution.json').read_text())
        key,value={'pid':('pid',124),'protocol':('protocol_sha256','wrong'),'command':('command',['other']),
            'completion':('completed_utc',None),'status':('status','failed'),'support_sha':('support_sha256','wrong')}[mutation]
        ex[key]=value;Path('execution.json').write_text(json.dumps(ex))
    with pytest.raises(ValueError):support.load_support('support.json',sha)


def test_exclusive_output_and_pinned_digest_shapes(tmp_path):
    p=tmp_path/'evidence.json';prepare.write_new(p,{'original':True})
    with pytest.raises(FileExistsError):prepare.write_new(p,{'replace':True})
    assert json.loads(p.read_text())=={'original':True}
    for digest in (prepare.CACHE_SHA,prepare.SELECTION_SHA,prepare.CACHE_PROTOCOL_SHA):
        assert len(digest)==64 and int(digest,16)>0


@pytest.mark.parametrize('input_drift',[False,True])
def test_synthetic_registration_pins500_before_metadata_and_is_exclusive(tmp_path,monkeypatch,input_drift):
    monkeypatch.chdir(tmp_path)
    selection,cp=selection_fixture()
    selection_path=Path('cache.selection.json');cache_path=Path('cache.pt');protocol_path=Path('cache.protocol.json')
    prepare.write_new(selection_path,selection);cache_path.write_bytes(b'opaque no tensors')
    stat=dict(size=1,mtime_ns=2,ctime_ns=3,inode=4,device=5)
    cp['provenance']['data_file_stats']={'/synthetic/p.h5':stat}
    cp['protocol'].update(selection_file=str(selection_path),selection_sha256=support.sha256_file(selection_path))
    prepare.write_new(protocol_path,cp);Path('source.py').write_text('synthetic source')
    monkeypatch.setattr(prepare,'OUTPUT',Path('output'));monkeypatch.setattr(prepare,'SOURCES',('source.py',))
    prior=Path('prior_failed.json');prior.write_text('{"status":"failed"}')
    monkeypatch.setattr(prepare,'FAILED_V1',{str(prior):support.sha256_file(prior)})
    for name,path in [('CACHE',cache_path),('SELECTION',selection_path),('CACHE_PROTOCOL',protocol_path)]:
        monkeypatch.setattr(prepare,name,path);monkeypatch.setattr(prepare,name+'_SHA',support.sha256_file(path))
    monkeypatch.setattr(prepare,'full_file_stat',lambda _:stat)
    monkeypatch.setattr(support,'full_file_stat',lambda _:stat)
    def forbidden(*args,**kwargs):raise AssertionError('registration must not read topology or deserialize')
    monkeypatch.setattr(prepare,'support_record',forbidden);monkeypatch.setattr(torch,'load',forbidden)
    if input_drift:
        cache_path.write_bytes(b'changed')
        with pytest.raises(ValueError,match='pinned input'):prepare.register()
        assert not Path('output').exists()
    else:
        p=prepare.register()
        assert len(p['windows'])==500 and p['training_order']==list(range(400))+list(range(100))
        assert p['cache_deserialized'] is False and Path(p['source_snapshot'],'source.py').read_text()=='synthetic source'
        with pytest.raises(FileExistsError):prepare.register()


def test_real_policy_live_tuple_to_json_plan_and_torch_cache_like_row_parity():
    window,plan,row=fixture()
    policy=row['topology']['source_identity']['chemical_alias_policy']
    assert type(CHEMICAL_ALIAS_POLICY['histidine_aliases']) is tuple
    assert type(policy['histidine_aliases']) is tuple
    live=copy.deepcopy(plan['record_support'])
    live['full_topology']['source_identity']['chemical_alias_policy']=copy.deepcopy(CHEMICAL_ALIAS_POLICY)
    assert support.canonical_sha256(live)==support.canonical_sha256(plan['record_support'])
    produced=support.make_plan(window,live)
    serialized=json.loads(json.dumps(produced))
    assert produced==serialized==plan
    before=copy.deepcopy(row['topology'])
    from_live=support.validate_cached_row(row,plan)
    from_json=support.validate_cached_row(row,serialized)
    assert torch.equal(from_live['pair_indices'],from_json['pair_indices'])
    assert row['topology']==before and type(policy['histidine_aliases']) is tuple


@pytest.mark.parametrize('mutation',['alias_changed','alias_reordered','alias_missing','identity','metadata_hash','edge'])
def test_normalization_does_not_weaken_identity_or_historical_sha(mutation):
    window,plan,row=fixture();rs=copy.deepcopy(plan['record_support']);full=rs['full_topology']
    if mutation.startswith('alias'):
        aliases=full['source_identity']['chemical_alias_policy']['histidine_aliases']
        if mutation=='alias_changed':aliases[0]='BAD'
        elif mutation=='alias_reordered':aliases.reverse()
        else:aliases.pop()
    elif mutation=='identity':full['source_identity']['domain_id']='other'
    elif mutation=='metadata_hash':full['metadata_hashes']['pdb']='changed'
    else:
        full['peptide_edges'].pop();full['peptide_atom_indices'].pop()
        graph=old.build_pair_eligibility(rs['selected_geometry_row_ids'],full['residue_uids'],full['peptide_edges'])
        rs['pair_indices']=torch.nonzero(torch.triu(graph,diagonal=1)).tolist();rs['pair_count']=len(rs['pair_indices'])
    rs['full_topology_sha256']=support.canonical_sha256(full)
    rs['record_plan_sha256']=support.canonical_sha256({k:v for k,v in rs.items() if k!='record_plan_sha256'})
    with pytest.raises(ValueError):support.make_plan(window,rs)
    if mutation.startswith('alias'):
        row['topology']['source_identity']['chemical_alias_policy']=copy.deepcopy(full['source_identity']['chemical_alias_policy'])
        with pytest.raises(ValueError,match='cached topology'):support.validate_cached_row(row,plan)


@pytest.mark.parametrize('bad',[{1:'bad'}, {'a':float('nan')}, {'a':torch.tensor(1)}, {'a':{1,2}}])
def test_metadata_normalization_rejects_non_json_values(bad):
    with pytest.raises(ValueError):support.canonical_metadata(bad)
