import json
from pathlib import Path
import numpy as np
import pytest
import torch
from evaluation.loaders import build_bundle,_validate_bundle_checkpoint,ModelBundle
from evaluation.run_eval import run,write_reports,_effective_config,audit_split
from evaluation.run_gate import persist_nll
from evaluation import variants as V


def manifest(tmp_path):
    paths=[]
    rng=np.random.default_rng(7)
    for protein in ['a','b']:
        coords=rng.normal(size=(14,8,4,3)).astype(np.float32)
        path=tmp_path/f'{protein}.npz';np.savez(path,coords=coords,time_ps=np.arange(14,dtype=np.float32)*1000.)
        paths.append(dict(npz_path=str(path),format='npz',protein_id=protein,
            trajectory_id=protein+'_r0',temperature_k=320.,num_frames=14,frame_interval_ps=1000.))
    path=tmp_path/'manifest.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in paths))
    return str(path)


def test_mock_end_to_end_is_diagnostic_and_byte_reproducible(tmp_path,monkeypatch):
    monkeypatch.setattr('evaluation.run_eval.source_provenance',lambda:{'source_sha256':'fixed-test-source'})
    cfg=dict(manifest=manifest(tmp_path),horizons_frames=[1,2],context_frames=3,
        max_proteins=2,max_anchors_per_trajectory=2,max_trajectories_per_protein=1,
        device='cpu',variants=['h_only','h_ft','h_plus_s','persist'],mock_encoder=True,
        n_rollout_reps=3,seed=5)
    first=run(cfg);second=run(cfg)
    a,b=tmp_path/'first',tmp_path/'second'
    write_reports(first,a);write_reports(second,b)
    for name in ['per_window.jsonl','per_protein.jsonl','samples.jsonl']:
        assert (a/name).read_bytes()==(b/name).read_bytes()
    report=json.loads((a/'results.json').read_text())
    assert 'aggregate' not in report and 'diagnostic_aggregate' in report
    assert len(first['rows'])==32
    assert all(r.rmsd==r.persist_rmsd for r in first['rows'])
    for byh in first['aggregate'].values():
        for byv in byh.values():
            for value in byv.values():
                assert value['delta_ci']==[0.,0.,0.]


def test_cli_none_does_not_override_config():
    assert _effective_config({'max_proteins':2},{'max_proteins':None})['max_proteins']==2


@pytest.mark.parametrize('reused_identity',['domain','source'])
def test_protein_holdout_rejects_relabelled_physical_identity(tmp_path,reused_identity):
    val_path=manifest(tmp_path)
    records=[json.loads(line) for line in Path(val_path).read_text().splitlines()]
    train=dict(records[0],protein_id='renamed_train',domain_id='train_domain')
    records[0]['domain_id']='train_domain' if reused_identity=='domain' else 'val_domain'
    if reused_identity=='domain':
        train['npz_path']=str(tmp_path/'separate_source.npz')
    train_path=tmp_path/'train.jsonl'
    train_path.write_text(json.dumps(train)+'\n')
    Path(val_path).write_text(''.join(json.dumps(r)+'\n' for r in records))
    expected='domain overlap' if reused_identity=='domain' else 'source-file overlap'
    with pytest.raises(ValueError,match=expected):
        audit_split(dict(manifest=val_path,train_manifests=[str(train_path)],split='protein_holdout'))


def test_temperature_holdout_allows_same_protein(tmp_path):
    val_path=manifest(tmp_path)
    records=[json.loads(line) for line in Path(val_path).read_text().splitlines()]
    train_path=tmp_path/'train.jsonl'
    train_path.write_text(json.dumps(dict(records[0],temperature_k=300.))+'\n')
    result=audit_split(dict(manifest=val_path,train_manifests=[str(train_path)],split='temperature_holdout'))
    assert result['protein_overlap']==1
    assert result['protein_split_audits']==[]
    assert result['training_exclusion_verified']


def test_changed_source_is_rejected(tmp_path,monkeypatch):
    hashes=iter([{'source_sha256':'before'},{'source_sha256':'after'}])
    monkeypatch.setattr('evaluation.run_eval.source_provenance',lambda:next(hashes))
    with pytest.raises(RuntimeError,match='source code changed'):
        run(dict(manifest=manifest(tmp_path),horizons_frames=[1],context_frames=3,
                 max_proteins=1,max_anchors_per_trajectory=1,device='cpu',variants=['persist']))


def test_legacy_rejected_before_model_loading():
    with pytest.raises(ValueError,match='representation_contract'):
        _validate_bundle_checkpoint({},allow_legacy=False)
    assert not _validate_bundle_checkpoint({},allow_legacy=True)


def test_direct_history_shape_and_delta_contract():
    class Predictor(torch.nn.Module):
        def forward(self,h,*args,**kwargs):
            assert h.ndim==4 and h.shape[1]==3
            return torch.ones(h.shape[0],h.shape[2],2,2)
    class OPF(torch.nn.Module):
        def decompose(self,h): return h.reshape(*h.shape[:-1],2,2)
        def compose(self,h): return h.flatten(-2)
    bundle=ModelBundle(torch.nn.Identity(),torch.nn.Identity(),None,'identity',predictor=Predictor(),opf=OPF())
    h=torch.full((1,3,5,4),2.)
    out=bundle.predict_future(h,torch.tensor([1000.]),torch.tensor([320.]),torch.ones(1,5,dtype=torch.bool))
    assert torch.equal(out,torch.full((1,5,4),3.))
    with pytest.raises(ValueError,match='B,K,L'):
        bundle.predict_future(h[:,-1],torch.tensor([1000.]),torch.tensor([320.]),torch.ones(1,5,dtype=torch.bool))


def test_cached_direct_normalizes_and_inverts_exactly_once():
    class Normalizer(torch.nn.Module):
        def forward(self,h): return (h-3.)/2.
        def inverse(self,h): return h*2.+3.
    class Predictor(torch.nn.Module):
        def forward(self,h,*args,**kwargs):
            assert torch.equal(h,torch.ones_like(h))
            return torch.ones_like(h[:,-1])
    bundle=ModelBundle(torch.nn.Identity(),Normalizer(),None,'identity',predictor=Predictor(),
                       predictor_output='normalized_H_delta')
    h=torch.full((1,3,5,4),5.)
    out=bundle.predict_future(h,torch.tensor([1000.]),torch.tensor([320.]),torch.ones(1,5,dtype=torch.bool))
    assert torch.equal(out,torch.full_like(h[:,-1],7.))


def test_persist_nll_normalizes_actual_transition_count():
    h=torch.zeros(1,5,2,3)
    mask=torch.ones(1,2,dtype=torch.bool)
    D=torch.ones(1,2,3)*.5
    expected=.5*np.log(2*np.pi)
    assert persist_nll(h,mask,D,1.)==pytest.approx(expected)
    assert persist_nll(h[:,:2],mask,D,1.)==pytest.approx(expected)
