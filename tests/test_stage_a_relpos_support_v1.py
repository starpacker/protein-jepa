"""Synthetic metadata/topology/clips only; no HDF, checkpoint or model IO."""
import copy
from dataclasses import replace

import pytest
import torch

from protein_jepa_frame.data import stage_a_relpos_support_v1 as adapter
from protein_jepa_frame.data.backbone_topology_v2 import BackboneTopologyV2, ResidueUID
from protein_jepa_frame.training import physical_distance_pair_support_v2 as physical
from protein_jepa_frame.training.train_stage_a_v2 import build_clip_batch_from_pair


def example(index=0, n=7, partial=False, origin=123.5):
    protein=f'p{index}'; path='/synthetic/'+protein+'.h5'
    ri=torch.arange(n)+47
    if n>3:ri[3]=ri[2]  # Real UID remains distinct through insertion code.
    mask=torch.ones(n,4,dtype=torch.bool)
    if partial:mask[1,3]=False
    topology=BackboneTopologyV2(
        residue_uids=tuple(ResidueUID(path,protein,'seg','A',int(ri[i]),'A' if i==3 else '') for i in range(n)),
        residue_names=('ALA',)*n,embedding_aatype=torch.arange(n)%20,
        embedding_residue_index=ri,embedding_chain_id=torch.zeros(n,dtype=torch.long),
        chemical_aatype=torch.zeros(n,dtype=torch.long),atom_indices=torch.arange(n*4).reshape(n,4),
        atom_mask=mask,geometry_row_ids=torch.arange(n),
        peptide_edges=torch.tensor([[i,i+1] for i in range(n-1)]),
        peptide_atom_indices=torch.tensor([[4*i+2,4*(i+1)] for i in range(n-1)]),
        metadata_hashes={'pdb':'a'*64,'psf':'b'*64},source_identity=dict(path=path,protein_id=protein,domain_id=protein))
    full=topology.as_metadata();graph=physical.build_pair_eligibility(range(n),full['residue_uids'],full['peptide_edges'])
    pairs=torch.nonzero(torch.triu(graph,diagonal=1)).tolist()
    rs=dict(record_key=physical.record_key(path,protein),record_path=path,protein=protein,domain_id=protein,
        max_residues=96,hdf_stat={'size':1,'mtime_ns':2},full_topology=full,
        full_topology_sha256=physical.canonical_sha256(full),selected_geometry_row_ids=list(range(n)),
        pair_indices=pairs,pair_count=len(pairs),legacy_mapping={k:full[k] for k in
        ('embedding_aatype','embedding_residue_index','embedding_chain_id','atom_indices','atom_mask')})
    rs['record_plan_sha256']=physical.canonical_sha256(rs)
    h=(1,4,16,64)[index%4]
    record=dict(path=path,protein_id=protein,trajectory_id=protein+'_T320_R1',format='mdcath-hdf5',
        num_frames=200,frame_interval_ps=1000.,first_frame_time_ps=origin,temperature_k=320.,
        replica_id='1',dataset='mdcath',domain_id=protein,metadata={})
    metadata=dict(protein=protein,trajectory=record['trajectory_id'],temperature_k=320.,anchor_frame=9,
        horizon=h,target_frame=9+h,horizon_ps=h*1000.,context_frames=list(range(2,10)),sampling_seed=None)
    historical=dict(metadata=metadata,source_identity=full['source_identity'],metadata_hashes=full['metadata_hashes'],
                    topology_sha256=topology.topology_sha256)
    window=dict(cache_row_index=index,split='fit' if index<400 else 'readout_heldout',metadata=metadata,
                record=record,record_key=rs['record_key'],historical_topology=historical)
    plan=adapter.support.make_plan(window,rs)
    def clip(frames):
        t=len(frames)
        coords=torch.arange(t*n*12,dtype=torch.float32).reshape(t,n,4,3)/10
        coords += torch.tensor(frames,dtype=torch.float32)[:,None,None,None]
        if partial:coords[:,1,3]=float('nan')
        return dict(coords=coords,atom_mask=mask[None].expand(t,-1,-1).clone(),
            residue_mask=mask.all(-1)[None].expand(t,-1).clone(),aatype=topology.embedding_aatype.clone(),
            residue_index=ri.clone(),chain_id=topology.embedding_chain_id.clone(),
            time_ps=origin+torch.tensor(frames,dtype=torch.float32)*1000,
            temperature_k=torch.tensor(320.),dataset='mdcath',protein_id=protein,trajectory_id=record['trajectory_id'])
    bound=dict(schema_version=adapter.VERSION,cache_plan=plan,window=window,
               support_protocol_sha256='s'*64,registered_windows_sha256='w'*64)
    rehash(bound)
    return bound,clip(metadata['context_frames']),clip([metadata['target_frame']]),topology


def rehash(bound):
    bound['adaptation_plan_sha256']=physical.canonical_sha256({k:v for k,v in bound.items() if k!='adaptation_plan_sha256'})


@pytest.fixture(scope='module')
def panel():
    examples=[example(i) for i in range(500)]
    windows=[x[0]['window'] for x in examples];plans=[x[0]['cache_plan'] for x in examples]
    doc=dict(schema_version=adapter.support.VERSION,status='completed',errors=[],plans=plans,
             registered_record_count=500,completed_record_count=500,protocol_sha256='s'*64)
    protocol=dict(schema_version=adapter.support.VERSION,windows=windows,windows_sha256=physical.canonical_sha256(windows),
                  n_records=500,n_distinct_proteins=500,max_residues=96,fit_indices=list(range(400)),
                  readout_heldout_indices=list(range(400,500)))
    return doc,protocol


def test_exact400_order_balanced_and_detached_whole500_metadata(panel):
    doc,protocol=panel;before=adapter.fingerprint(panel)
    selected=adapter.select_adaptation_plans(doc,protocol)
    assert len(selected)==400 and [p['cache_plan']['cache_row_index'] for p in selected]==list(range(400))
    assert [p['cache_plan']['metadata']['horizon'] for p in selected]==[1,4,16,64]*100
    assert {p['cache_plan']['metadata']['protein'] for p in selected}.isdisjoint(
        p['metadata']['protein'] for p in doc['plans'][400:])
    assert selected[0]['window']['record']['first_frame_time_ps']==123.5
    selected[0]['window']['record']['first_frame_time_ps']=0.
    assert adapter.fingerprint(panel)==before


@pytest.mark.parametrize('fault',['short','reorder','replacement','split','horizon','duplicate_protein','window_hash','count'])
def test_full500_contract_rejects_selection_changes(panel,fault):
    d,p=copy.deepcopy(panel)
    if fault=='short':d['plans']=d['plans'][:400]
    elif fault=='reorder':d['plans'][0],d['plans'][1]=d['plans'][1],d['plans'][0]
    elif fault=='replacement':d['plans'][0]=copy.deepcopy(d['plans'][400])
    elif fault=='split':d['plans'][400]['split']='fit'
    elif fault=='horizon':p['windows'][400]['metadata']['horizon']=4
    elif fault=='duplicate_protein':p['windows'][400]['metadata']['protein']='p0'
    elif fault=='window_hash':p['windows_sha256']='x'*64
    else:d['completed_record_count']=499
    if fault!='window_hash':p['windows_sha256']=physical.canonical_sha256(p['windows'])
    with pytest.raises(ValueError):adapter.select_adaptation_plans(d,p)


@pytest.mark.parametrize('index',[0,1,2,3])
def test_native_single_endpoint_all_horizons_with_nonzero_origin(index):
    plan,context,future,topology=example(index)
    before=adapter.fingerprint((plan,context,future,topology.as_metadata()));rng=torch.get_rng_state().clone()
    row=adapter.prepare_adaptation_row(plan,context,future,topology)
    assert row['context']['coords'].shape[0]==8 and row['future']['coords'].shape[0]==1
    assert torch.equal(row['context']['coords'],context['coords'])
    assert torch.equal(row['future']['coords'],future['coords'])  # No alignment/centering.
    assert row['horizon_time_ps']==future['time_ps'][0]-context['time_ps'][-1]
    assert row['horizon_time_ps']==(1,4,16,64)[index]*1000
    assert row['context']['time_ps'][0]==2123.5
    assert adapter.fingerprint((plan,context,future,topology.as_metadata()))==before
    assert torch.equal(torch.get_rng_state(),rng)
    assert row['adaptation_audit']['input_sha256_before']==row['adaptation_audit']['input_sha256_after']
    row['context']['coords'].zero_()
    assert bool(context['coords'].any())


@pytest.mark.parametrize('index',[400,499])
def test_true_heldout_plan_rejected_even_with_valid_clips(index):
    args=example(index)
    with pytest.raises(ValueError,match='excludes last100'):adapter.prepare_adaptation_row(*args)


def test_hidden_residue_zeroing_preserves_partial_atommask_uid_and_duplicate_ids():
    plan,context,future,topology=example(partial=True)
    before=adapter.fingerprint((context,future))
    row=adapter.prepare_adaptation_row(plan,context,future,topology)
    for name,original in [('context',context),('future',future)]:
        assert bool(torch.isfinite(row[name]['coords']).all()) and not bool(row[name]['coords'][:,1].any())
        assert torch.equal(row[name]['coords'][:,[0,2,3,4,5,6]],original['coords'][:,[0,2,3,4,5,6]])
        assert torch.equal(row[name]['atom_mask'],original['atom_mask'])
        assert row[name]['residue_index'][2]==row[name]['residue_index'][3]==49
    assert row['adaptation_metadata']['residue_uids'][2]!=row['adaptation_metadata']['residue_uids'][3]
    assert adapter.fingerprint((context,future))==before


@pytest.mark.parametrize('fault',['context_shape','future_shape','context_time','target_time','time_dtype',
    'valid_nan','endpoint_inf','context_support','endpoint_support','ri','aa','chain','protein','trajectory',
    'temperature','record_origin','plan_sha','topology_identity','mask_dtype','coord_dtype','no_grad'])
def test_support_identity_time_and_nonfinite_faults_fail(fault):
    plan,context,future,topology=example()
    if fault=='context_shape':context['coords']=context['coords'][:7]
    elif fault=='future_shape':future['coords']=future['coords'].repeat(8,1,1,1)
    elif fault=='context_time':context['time_ps'][0]+=1000
    elif fault=='target_time':future['time_ps'][0]=context['time_ps'][-1]
    elif fault=='time_dtype':future['time_ps']=future['time_ps'].double()
    elif fault=='valid_nan':context['coords'][0,0,0,0]=float('nan')
    elif fault=='endpoint_inf':future['coords'][0,0,0,0]=float('inf')
    elif fault=='context_support':context['residue_mask'][0,0]=False
    elif fault=='endpoint_support':future['atom_mask'][0,0,0]=False;future['residue_mask'][0,0]=False
    elif fault in ('ri','aa','chain'):
        future[{'ri':'residue_index','aa':'aatype','chain':'chain_id'}[fault]][0]+=1
    elif fault=='protein':future['protein_id']='wrong'
    elif fault=='trajectory':context['trajectory_id']='wrong'
    elif fault=='temperature':context['temperature_k']+=1
    elif fault=='record_origin':plan['window']['record']['first_frame_time_ps']=0.;rehash(plan)
    elif fault=='plan_sha':plan['adaptation_plan_sha256']='x'*64
    elif fault=='topology_identity':topology=replace(topology,residue_names=('GLY',)*7)
    elif fault=='mask_dtype':context['residue_mask']=context['residue_mask'].long()
    elif fault=='coord_dtype':future['coords']=future['coords'].double()
    else:future['coords'].requires_grad_(True)
    with pytest.raises(ValueError):adapter.prepare_adaptation_row(plan,context,future,topology)


def test_collation_mixed_lengths_preserves_legacy_padding_and_spatial_context():
    rows=[adapter.prepare_adaptation_row(*example(0,n=5,partial=True)),
          adapter.prepare_adaptation_row(*example(1,n=7))]
    before=adapter.fingerprint(rows);batch=adapter.collate_adaptation_rows(rows)
    assert batch['context_coords'].shape==(2,8,7,4,3) and batch['future_coords'].shape==(2,1,7,4,3)
    assert batch['horizon_time_ps'].tolist()==[1000.,4000.]
    for prefix in ('context','future'):
        assert batch[prefix+'_aatype'][0,5:].tolist()==[20,20]
        assert batch[prefix+'_residue_index'][0,5:].tolist()==[-1,-1]
        assert batch[prefix+'_chain_id'][0,5:].tolist()==[-1,-1]
        assert not bool(batch[prefix+'_residue_mask'][0,:,5:].any())
        assert not bool(batch[prefix+'_coords'][0,:,5:].any())
        assert batch[prefix+'_residue_index'][0,:5].tolist()==[47,48,49,49,51]
    assert batch['geometry_row_ids'][0].tolist()==[0,1,2,3,4,-1,-1]
    assert batch['context_dt_ps'][0].tolist()==[0.]+[1000.]*7
    assert batch['future_dt_ps'].tolist()==[[0.],[0.]]  # Lag is the explicit endpoint-anchor field.
    spatial=build_clip_batch_from_pair(batch,torch.device('cpu'))
    assert spatial['coords'] is batch['context_coords'] and spatial['coords'].shape[1]==8
    assert adapter.fingerprint(rows)==before


def test_prepare_detects_unexpected_reader_helper_mutation(monkeypatch):
    args=example();original=adapter.verify_clip
    def mutator(clip,*rest,**kwargs):
        original(clip,*rest,**kwargs);clip['coords'][0,0,0,0]+=1
    monkeypatch.setattr(adapter,'verify_clip',mutator)
    with pytest.raises(ValueError,match='inputs mutated'):adapter.prepare_adaptation_row(*args)


@pytest.mark.parametrize('field',['coords','metadata','holdout'])
def test_collate_rejects_changed_prepared_row(field):
    row=adapter.prepare_adaptation_row(*example())
    if field=='coords':row['context']['coords'][0,0,0,0]+=1
    elif field=='metadata':row['adaptation_metadata']['residue_uids'][0]['insertion_code']='B'
    else:row['adaptation_audit']['cache_row_index']=400
    with pytest.raises(ValueError):adapter.collate_adaptation_rows([row])


def test_collate_detects_dependency_input_mutation(monkeypatch):
    row=adapter.prepare_adaptation_row(*example());original=adapter.collate_paired_windows
    def mutator(rows):
        out=original(rows);rows[0]['context']['coords'][0,0,0,0]+=1;return out
    monkeypatch.setattr(adapter,'collate_paired_windows',mutator)
    with pytest.raises(ValueError,match='collation mutated'):adapter.collate_adaptation_rows([row])


def test_empty_collation_rejected():
    with pytest.raises(ValueError,match='empty'):adapter.collate_adaptation_rows([])


def test_insufficient_static_support_fails_instead_of_replacement():
    with pytest.raises(ValueError,match='fixed all8 observed support'):
        adapter.prepare_adaptation_row(*example(n=4,partial=True))


def test_no_optional_cached_h_or_coordinates_are_consumed():
    plan,context,future,topology=example()
    # Reader extras are outside the explicit consumed clip view.
    context['anchor_H']=object();future['oracle_H']=object()
    row=adapter.prepare_adaptation_row(plan,context,future,topology)
    assert 'anchor_H' not in row['context'] and 'oracle_H' not in row['future']


def test_large_time_origin_rounding_rejects_distorted_history_even_with_exact_endpoint_gap():
    args=example(index=3,origin=float(2**30))
    _,context,future,_=args
    assert future['time_ps'][0]-context['time_ps'][-1]==64000.
    assert torch.diff(context['time_ps']).tolist()==[896.,1024.,1024.,1024.,1024.,896.,1024.]
    before=adapter.fingerprint((context,future))
    with pytest.raises(ValueError,match='seven exact1000ps intervals'):
        adapter.prepare_adaptation_row(*args)
    assert adapter.fingerprint((context,future))==before
