from dataclasses import asdict
import copy
import pytest
import torch
from protein_jepa_frame.models.stage_a_v2 import StageAV2,StageAV2Config
from protein_jepa_frame.models.stage_b1a import StageBOracleConfig,StageBOracleDecoder
from protein_jepa_frame.models.stage_b1b import StageB1b,StageB1bConfig
from protein_jepa_frame.utils.contracts import build_representation_contract,build_prediction_contract
from evaluation.loaders import build_bundle,_training_provenance,sha256_file


def checkpoints(tmp_path):
    torch.manual_seed(31)
    ac=StageAV2Config(latent_dim=16,num_factors=2,encoder_depth=1,num_heads=2,
        predictor_depth=1,predictor_heads=2,predictor_spatial_depth=1,dropout=0.,max_frames=16)
    stage=StageAV2(ac)
    oc=StageBOracleConfig(latent_dim=16,decoder_c_hidden=16,decoder_n_heads=2,decoder_n_ipa_blocks=1)
    oracle=StageBOracleDecoder(oc)
    oracle.h_normalizer.fit(torch.randn(16,16)*2+3)
    rep=build_representation_contract(stage.context_encoder.state_dict(),asdict(ac),
        oracle.h_normalizer.state_dict(),oracle.decoder.state_dict(),asdict(oc))
    bc=StageB1bConfig(latent_dim=16,energy_hidden=16,energy_heads=2,bath_dim=2)
    dynamics=StageB1b(bc)
    b1a=dict(encoder=stage.context_encoder.state_dict(),stage_a_v2_config=asdict(ac),
        oracle=oracle.state_dict(),stage_b1a_config=asdict(oc),representation_contract=rep)
    b1b=dict(**b1a,model=dynamics.state_dict(),stage_b1b_config=asdict(bc),
             run_scope='contract_smoke_only',training_contract='autonomous_history_only_v2')
    a=dict(model=stage.state_dict(),stage_a_v2_config=asdict(ac),
        prediction_contract=build_prediction_contract(stage.context_encoder.state_dict(),asdict(ac)))
    paths=[]
    for name,ck in [('a',a),('b1a',b1a),('b1b',b1b)]:
        p=tmp_path/(name+'.pt');torch.save(ck,p);paths.append(str(p))
    return paths


def test_actual_self_contained_bundle_and_normalization_roundtrip(tmp_path):
    a,b,c=checkpoints(tmp_path)
    oracle=build_bundle(variant='oracle',b1a_ckpt=b)
    direct=build_bundle(variant='h_ft',b1a_ckpt=b,stage_a_predictor_ckpt=a)
    dynamics=build_bundle(variant='h_plus_s',b1b_ckpt=c,b1a_ckpt=b)
    assert oracle.representation_contract==direct.representation_contract==dynamics.representation_contract
    raw=torch.randn(1,3,4,16)
    assert torch.allclose(dynamics.inverse(dynamics.normalize(raw)),raw,atol=1e-6)
    assert not torch.allclose(dynamics.normalize(raw),raw)
    assert dynamics.decoder_kind=='oracle' and dynamics.diagnostic_only


def test_mutated_predictor_encoder_and_bundle_state_rejected(tmp_path):
    a,b,c=checkpoints(tmp_path)
    ck=torch.load(a,weights_only=False)
    key=next(k for k in ck['model'] if k.startswith('context_encoder.') and ck['model'][k].is_floating_point())
    ck['model'][key]=ck['model'][key]+.1
    torch.save(ck,a)
    with pytest.raises(ValueError,match='weights differ'):
        build_bundle(variant='h_ft',b1a_ckpt=b,stage_a_predictor_ckpt=a)
    ck=torch.load(c,weights_only=False)
    key=next(k for k in ck['encoder'] if ck['encoder'][k].is_floating_point())
    ck['encoder'][key]=ck['encoder'][key]+.1
    torch.save(ck,c)
    with pytest.raises(ValueError,match='contract mismatch'):
        build_bundle(variant='h_plus_s',b1b_ckpt=c)


def test_upstream_checkpoint_without_actual_manifest_is_rejected(tmp_path):
    source=tmp_path/'source.pt'
    torch.save({'train_args':{}},source)
    contract={'source':{'stage_a_checkpoint':str(source),
                        'stage_a_checkpoint_sha256':sha256_file(source)}}
    with pytest.raises(ValueError,match='no actual training manifest provenance'):
        _training_provenance({},contract)


def test_autonomous_lineage_includes_frozen_b1a_training_manifest(tmp_path):
    _,b,c=checkpoints(tmp_path)
    paths=[]
    for name in ('stage_a','b1a','b1b'):
        path=tmp_path/(name+'_train.jsonl')
        path.write_text('{"protein_id":"'+name+'"}\n')
        paths.append(str(path))
    ac=tmp_path/'stage_a_source.pt'
    torch.save({'train_args':{'manifest':paths[0]}},ac)
    b1a=torch.load(b,weights_only=False)
    contract=b1a['representation_contract']
    contract['source']={'stage_a_checkpoint':str(ac),'stage_a_checkpoint_sha256':sha256_file(ac),
        'stage_a_train_manifest_sha256':sha256_file(paths[0]),
        'b1a_train_manifest_sha256':sha256_file(paths[1])}
    b1a['train_args']={'manifest':paths[1]}
    torch.save(b1a,b)
    b1b=torch.load(c,weights_only=False)
    b1b['representation_contract']=contract
    b1b['train_args']={'manifest':paths[2],'b1a_ckpt':b}
    b1b['manifest_split']={'train_manifest_sha256':sha256_file(paths[2])}
    result=_training_provenance(b1b,contract)
    assert result['training_lineage_complete']
    assert not result['synthetic_lineage']
    assert {r['manifest'] for r in result['training_manifests']}==set(paths)
    assert all(r['saved_hash_verified'] for r in result['training_manifests'])
    del b1b['train_args']['b1a_ckpt']
    assert not _training_provenance(b1b,contract)['training_lineage_complete']


def test_formal_autonomous_bundle_requires_actual_passed_gate_metadata(tmp_path):
    a,b,c=checkpoints(tmp_path)
    ck=torch.load(c,weights_only=False)
    ck['run_scope']='oracle_gate_passed'
    torch.save(ck,c)
    with pytest.raises(ValueError,match='passed held-out oracle gate'):
        build_bundle(variant='h_plus_s',b1b_ckpt=c)
    ck['oracle_gate']={'schema_version':2,'passed':True,
        'representation_id':ck['representation_contract']['representation_id'],
        'scope':'heldout_protein'}
    torch.save(ck,c)
    assert not build_bundle(variant='h_plus_s',b1b_ckpt=c).diagnostic_only


def test_failed_gate_comparison_requires_explicit_scope_and_real_no_bath(tmp_path):
    _,_,c=checkpoints(tmp_path)
    ck=torch.load(c,weights_only=False)
    ck['run_scope']='diagnostic_comparison'
    ck['oracle_gate']={'passed':False}
    torch.save(ck,c)
    with pytest.raises(ValueError,match='explicit diagnostic_comparison'):
        build_bundle(variant='h_plus_s',b1b_ckpt=c)
    bundle=build_bundle(variant='h_plus_s',b1b_ckpt=c,diagnostic_comparison=True)
    assert bundle.diagnostic_only
    with pytest.raises(ValueError,match='bath_dim=0'):
        build_bundle(variant='h_markov',b1b_ckpt=c,diagnostic_comparison=True)
    cfg=StageB1bConfig(latent_dim=16,energy_hidden=16,energy_heads=2,bath_dim=0)
    ck['model']=StageB1b(cfg).state_dict()
    ck['stage_b1b_config']=asdict(cfg)
    torch.save(ck,c)
    assert build_bundle(variant='h_markov',b1b_ckpt=c,diagnostic_comparison=True).b1b.cfg.bath_dim==0
    with pytest.raises(ValueError,match='bath_dim>0'):
        build_bundle(variant='h_plus_s',b1b_ckpt=c,diagnostic_comparison=True)


def test_actual_models_run_complete_diagnostic_forecast(tmp_path,monkeypatch):
    import json
    import numpy as np
    from evaluation.run_eval import run
    monkeypatch.setattr('evaluation.run_eval.source_provenance',lambda:{'source_sha256':'fixed-test-source'})
    a,b,c=checkpoints(tmp_path)
    coords=np.random.default_rng(29).normal(size=(7,8,4,3)).astype('float32')
    trajectory=tmp_path/'trajectory.npz'
    np.savez(trajectory,coords=coords,time_ps=np.arange(7,dtype='float32')*1000.)
    manifest=tmp_path/'val.jsonl'
    manifest.write_text(json.dumps(dict(npz_path=str(trajectory),protein_id='p',trajectory_id='p_r0',
        num_frames=7,temperature_k=320.,frame_interval_ps=1000.))+'\n')
    result=run(dict(manifest=str(manifest),horizons_frames=[1],context_frames=3,
        max_anchors_per_trajectory=1,device='cpu',split='diagnostic',n_rollout_reps=2,
        variants=['h_ft','h_plus_s','persist'],models=dict(b1a_ckpt=b,b1b_ckpt=c,stage_a_predictor_ckpt=a)))
    assert result['meta']['diagnostic_only']
    assert len(result['rows'])==3
    for row in result['rows']:
        assert np.isfinite(row.rmsd)
        assert len(row.sample_rmsds)==(2 if row.variant=='h_plus_s' else 1)
        if row.variant!='persist':
            assert np.isfinite(row.oracle_rmsd)


def test_kalman_diagnostic_uses_real_contiguous_path(tmp_path):
    import argparse,json,numpy as np
    from evaluation.run_gate import run
    _,_,ckpt=checkpoints(tmp_path)
    trajectory=tmp_path/'sequence.npz'
    coords=np.random.default_rng(41).normal(size=(8,8,4,3)).astype('float32')
    np.savez(trajectory,coords=coords,time_ps=np.arange(8,dtype='float32')*1000.)
    manifest=tmp_path/'diagnostic.jsonl'
    manifest.write_text(json.dumps(dict(npz_path=str(trajectory),protein_id='p',trajectory_id='r',
        num_frames=8,temperature_k=320.,frame_interval_ps=1000.))+'\n')
    args=argparse.Namespace(seed=0,device='cpu',ckpt=ckpt,b1a_ckpt=None,manifest=str(manifest),
        horizon=2,context_frames=3,n_prot=1,anchor_stride=80,max_anchors_per_trajectory=1,
        split='diagnostic',train_manifest=[],out=str(tmp_path/'kalman.json'))
    result=run(args)
    assert result['diagnostic_only'] and not result['headline_eligible']
    assert result['rows'][0]['steps']==2
    assert result['rows'][0]['n_scored_dimensions']==2*8*16
    assert np.isfinite(result['rows'][0]['full'])
