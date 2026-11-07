"""Preparation boundary tests use fake models and metadata; never load a GPU."""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from protein_jepa_frame.data.backbone_topology_v2 import BackboneTopologyV2, ResidueUID
from protein_jepa_frame.data.manifest import TrajectoryRecord
from scripts.prepare_backbone_readout_cache import (
    WindowSpec, approved_manifest, budget, prepare_row, select_train, select_val,
    verify_clip, _write_json_exclusive, _snapshot_sources,
)
from evaluation.loaders import sha256_file


def record(protein='p', replica=0, n_frames=100):
    return TrajectoryRecord(path=Path('/fake.h5'), protein_id=protein,
        trajectory_id=f'{protein}_T320_R{replica}', format='mdcath-hdf5',
        num_frames=n_frames, frame_interval_ps=1000., temperature_k=320., domain_id=protein)


def fixture():
    ca = torch.tensor([[0.,0.,0.],[3.8,0.,0.],[6.,3.,0.],[5.,6.,2.],[2.,8.,1.],[-1.,6.,2.]])
    bb = ca[:,None] + torch.tensor([[-.5,1.3,0.],[0.,0.,0.],[1.5,0.,0.],[2.,1.,0.]])
    topology = BackboneTopologyV2(
        residue_uids=tuple(ResidueUID('/fake.h5','p','P','A',i,'') for i in range(6)),
        residue_names=('ALA',)*6, embedding_aatype=torch.zeros(6,dtype=torch.long),
        embedding_residue_index=torch.arange(6), embedding_chain_id=torch.zeros(6,dtype=torch.long),
        chemical_aatype=torch.zeros(6,dtype=torch.long), atom_indices=torch.arange(24).reshape(6,4),
        atom_mask=torch.ones(6,4,dtype=torch.bool), geometry_row_ids=torch.arange(6),
        peptide_edges=torch.tensor([[0,1],[1,2],[2,3],[3,4],[4,5]]),
        peptide_atom_indices=torch.tensor([[2,4],[6,8],[10,12],[14,16],[18,20]]),
        metadata_hashes={'pdb':'fake','psf':'fake'}, source_identity={'path':'/fake.h5'})
    spec = WindowSpec(record(),7,4)
    observed = dict(coords=bb[None].repeat(8,1,1,1), atom_mask=topology.atom_mask[None].repeat(8,1,1),
        residue_mask=torch.ones(8,6,dtype=torch.bool), aatype=topology.embedding_aatype.clone(),
        residue_index=topology.embedding_residue_index.clone(), chain_id=topology.embedding_chain_id.clone(),
        time_ps=torch.arange(8,dtype=torch.float32)*1000.)
    target = {key:(value[-1:].clone() if key in ('coords','atom_mask','residue_mask','time_ps') else value.clone())
              for key,value in observed.items()}
    target['time_ps'] = torch.tensor([11000.])
    target['coords'][:,3,2,2] += 1.5
    return spec,topology,observed,target


def fake_model_and_reader(spec, observed, target, *, mode='val'):
    events = []
    class Bundle:
        def encode(self, coords, mask, aa, ri, ci):
            events.append('encode')
            assert coords.shape[0] == 1
            out = torch.zeros(1,coords.shape[1],128)
            out[...,:3] = coords.mean(-2)
            return out
        def decode_ca(self, h, anchor_h, aa, ri, ci, rotation, anchor, mask):
            events.append('decode')
            return anchor + .2*(h[...,:3]-anchor_h[...,:3])
    class Model:
        training=False
        prediction_kind='endpoint'
        kernel=SimpleNamespace(cfg=SimpleNamespace(step_ps=1000.))
        bundle=Bundle()
        def sample(self, history, mask, aa, ri, ci, temperature, horizon, count, generator):
            assert events == ['read_observed']
            assert torch.equal(history[0],observed['coords'])
            assert torch.equal(aa[0],observed['aatype'])
            assert horizon == spec.horizon and count == 12
            events.append('sample')
            self.last_observation_steps=[horizon]*count
            ca=history[:,-1,:,1][:,None].repeat(1,count,1,1)
            ca=ca+.1*torch.randn(ca.shape,generator=generator)
            h=torch.zeros(1,count,ca.shape[2],128)
            h[...,:3]=ca
            return ca,h
    def reader(frames):
        if len(frames)==8:
            assert np.array_equal(frames,spec.context_frames)
            events.append('read_observed')
            return copy.deepcopy(observed)
        assert frames.tolist()==[spec.target_frame]
        expected=['read_observed','sample','encode'] if mode=='val' else ['read_observed','encode']
        assert events == expected, 'label was read before forecast/anchor encoding'
        events.append('read_target')
        return copy.deepcopy(target)
    return Model(),reader,events


def test_val_forecast_before_labels_and_future_perturbation_cannot_change_forecast():
    spec,topology,obs,target=fixture()
    model,reader,events=fake_model_and_reader(spec,obs,target)
    first=prepare_row(model,spec,topology,reader,mode='val')
    changed=copy.deepcopy(target)
    changed['coords'][:,2,:,2]+=4
    model,reader,_=fake_model_and_reader(spec,obs,changed)
    second=prepare_row(model,spec,topology,reader,mode='val')
    assert events==['read_observed','sample','encode','read_target','encode','decode']
    for key in ('forecast_H','forecast_CA','anchor_H','observed_R','observed_backbone'):
        assert torch.equal(first[key],second[key])
    assert not torch.equal(first['oracle_H'],second['oracle_H'])
    assert not torch.equal(first['target_backbone'],second['target_backbone'])
    assert first['fingerprints']['observation']==second['fingerprints']['observation']
    assert first['fingerprints']['raw_label']!=second['fingerprints']['raw_label']
    assert first['forecast_H'].shape==(12,6,128)
    assert all(not value.requires_grad and value.device.type=='cpu' for value in first.values() if torch.is_tensor(value))


def test_train_encodes_only_anchor_and_endpoint_no_sampler_and_preserves_legacy_ids():
    spec,topology,obs,target=fixture()
    # Insertion-code example: legacy numbers may repeat while geometry IDs do not.
    topology=replace(topology,embedding_residue_index=torch.tensor([1,1,1,2,3,4]),
        chemical_aatype=torch.tensor([0,0,8,0,0,0]),embedding_aatype=torch.tensor([0,0,20,0,0,0]))
    for clip in (obs,target):
        clip['residue_index']=topology.embedding_residue_index.clone()
        clip['aatype']=topology.embedding_aatype.clone()
    model,reader,events=fake_model_and_reader(spec,obs,target,mode='train')
    result=prepare_row(model,spec,topology,reader,mode='train')
    assert events==['read_observed','encode','read_target','encode','decode']
    assert 'forecast_H' not in result and 'forecast_CA' not in result
    assert result['embedding_residue_index'].tolist()==[1,1,1,2,3,4]
    assert result['embedding_aatype'][2]==20 and result['chemical_aatype'][2]==8
    assert torch.equal(result['peptide_edges'],topology.peptide_edges)
    assert result['metadata']['sampling_seed'] is None


def test_target_rigid_gauge_uses_CA_fit_for_all_atoms():
    spec,topology,obs,target=fixture()
    target['coords']=obs['coords'][-1:].clone()
    R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    target['coords']=target['coords']@R+torch.tensor([5.,-3.,7.])
    model,reader,_=fake_model_and_reader(spec,obs,target,mode='train')
    row=prepare_row(model,spec,topology,reader,mode='train')
    assert torch.allclose(row['target_backbone'],obs['coords'][-1:],atol=1e-6)


@pytest.mark.parametrize('bad', ['time','aa','ri','chain','target_atom','nonfinite'])
def test_bad_boundary_fails_without_mask_shrinking_or_replacement(bad):
    spec,topology,obs,target=fixture()
    if bad=='time': target['time_ps']+=1
    elif bad=='aa': target['aatype'][0]=1
    elif bad=='ri': target['residue_index'][0]=100
    elif bad=='chain': target['chain_id'][0]=1
    elif bad=='target_atom':
        target['atom_mask'][0,0,0]=False
        target['residue_mask'][0,0]=False
    else: target['coords'][0,0,0,0]=float('nan')
    model,reader,_=fake_model_and_reader(spec,obs,target,mode='train')
    with pytest.raises(ValueError): prepare_row(model,spec,topology,reader,mode='train')


def test_selection_fixed_random_population_no_replacement_and_complete_prefix():
    records=[record(f'p{i}',j,90+i) for i in range(12) for j in range(3)]
    selected=select_train(records,population=12,count=8)
    again=select_train(records,population=12,count=8)
    assert selected==again and len({w.protein_id for w in selected})==8
    assert [w.horizon for w in selected]==[1,4,16,64]*2
    assert all(7<=w.anchor_frame<=w.record.num_frames-w.horizon-1 for w in selected)
    assert all(w.target_frame not in w.context_frames for w in selected)
    proteins=list(dict.fromkeys(r.protein_id for r in records))
    expected=[proteins[int(i)] for i in np.random.default_rng(42).permutation(12)[:8]]
    assert [w.protein_id for w in selected]==expected
    with pytest.raises(ValueError,match='population'):select_train(records,population=13,count=8)


def test_val_schedule_matches_all_registered_keys_and_rejects_any_changed_key():
    records=[];reference=[]
    temperatures=(320.,348.,379.,413.,450.)
    for i in range(50):
        family=[]
        for t in temperatures:
            for replica in range(5):
                r=replace(record(f'p{i}',replica),temperature_k=t,trajectory_id=f'p{i}_T{int(t)}_R{replica}')
                records.append(r);family.append(r)
        r=family[i%25]
        reference.extend(WindowSpec(r,7,h).metadata('val') for h in (1,4,16,64))
    result=select_val(records,reference)
    assert len(result)==200 and len({w.protein_id for w in result})==50
    reference[-1]['anchor_frame']+=1
    with pytest.raises(ValueError,match='keys'):select_val(records,reference)


def test_manifest_and_budget_fail_closed_before_loading_files(tmp_path):
    assert approved_manifest('train').name=='train.jsonl'
    assert approved_manifest('val').name=='val.jsonl'
    for name in ('test.jsonl','other.jsonl','train.jsonl'):
        with pytest.raises(ValueError):approved_manifest('val',tmp_path/name)
    assert budget('train',None)==500 and budget('val',None)==200
    for count in (0,1,5,201,True):
        with pytest.raises(ValueError):budget('val',count)


def test_evidence_writes_exclusive_and_snapshot_verifies_source_bytes(tmp_path,monkeypatch):
    target=tmp_path/'protocol.json'
    _write_json_exclusive(target,{'done':True})
    with pytest.raises(FileExistsError):_write_json_exclusive(target,{'done':False})
    monkeypatch.chdir(tmp_path)
    source=Path('source.py');source.write_text('x=1\n')
    snapshot=Path('snapshot')
    _snapshot_sources(snapshot,{str(source):sha256_file(source)})
    assert (snapshot/source).read_bytes()==source.read_bytes()
    with pytest.raises(FileExistsError):_snapshot_sources(snapshot,{str(source):sha256_file(source)})
    with pytest.raises(RuntimeError,match='differs'):_snapshot_sources(Path('bad'),{str(source):'0'*64})
