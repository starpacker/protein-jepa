"""Synthetic readout/statistical tests; no real cache, checkpoint or trajectories."""
import copy
import json
import math

import numpy as np
import pytest
import torch

from protein_jepa_frame.training import stage_a_relpos_readout_v1 as r
from protein_jepa_frame.models.stage_a_relpos_v1 import StageARelposV1
from protein_jepa_frame.models.stage_a_v2 import StageAV2Config


def row(index, length=4):
    x = torch.arange(length*12, dtype=torch.float32).reshape(1, length, 4, 3)/7
    mask = torch.ones(length, dtype=torch.bool)
    metadata = dict(protein='p'+str(index), horizon=(1,4,16,64)[index%4])
    raw = dict(metadata=metadata, observed_backbone=x, target_backbone=x*1.1,
        observed_mask=mask, atom_mask=torch.ones(length,4,dtype=torch.bool),
        target_atom_mask=torch.ones(length,4,dtype=torch.bool),
        embedding_aatype=torch.zeros(length,dtype=torch.long),
        embedding_residue_index=torch.arange(length), embedding_chain_id=torch.zeros(length,dtype=torch.long),
        oracle_H=object(), anchor_H=object())
    pairs = torch.tensor([[0,3]],dtype=torch.long)
    labels = r.probe_api.pair_log_targets(raw,pairs)
    prepared = dict(cache_row_index=index,row=raw,mask=mask,pairs=pairs,labels=labels,
        physical_support_audit=dict(split='fit' if index<400 else 'readout_heldout'),
        labels_sha256=r.state_sha256(dict(pair_indices=pairs,**labels)))
    H = torch.arange(length*128,dtype=torch.float32).reshape(1,length,128)/1000
    return prepared,dict(observed_H=H,target_H=H*.9)


def test_prepare_all500_geometry_only_and_split(monkeypatch):
    rows = [row(i)[0]['row'] for i in range(500)]
    seen = []
    def attach(raw,index,support):
        seen.append(index)
        return dict(pair_indices=torch.tensor([[0,3]]),observed_mask=raw['observed_mask'].clone(),
            audit=dict(split='fit' if index<400 else 'readout_heldout'))
    monkeypatch.setattr(r.support_api,'attach_cached_physical_support',attach)
    prepared = r.prepare_readout_rows(rows,dict(plans=[None]*500))
    assert seen == list(range(500)) and len(prepared)==500
    assert prepared[0]['row']['oracle_H'] is rows[0]['oracle_H']
    # Opaque cached H would fail if any numerical/hash operation consumed it.
    assert len(r._input_hash(prepared))==64
    rows[-1]['metadata']['protein']='p0'
    with pytest.raises(ValueError):r.prepare_readout_rows(rows,dict(plans=[None]*500))


@pytest.mark.parametrize('mode',['pca','zero','local'])
def test_versioned_encoding_exact_frozen_api_and_all_state_rng_identity(mode):
    cfg=StageAV2Config(latent_dim=128,encoder_depth=1,num_heads=4,num_rbf=3,max_residues=8,
        predictor_depth=1,predictor_spatial_depth=1,predictor_heads=4,struct_recon_dim=8)
    model=StageARelposV1(cfg,feature_mode=mode).eval().requires_grad_(False)
    prepared,_=row(0);rng=torch.get_rng_state().clone()
    state=r.state_sha256(model);buffers=r.state_sha256(dict(model.named_buffers()))
    features,audit=r.encode_readout_rows(model,[prepared])
    observed,target=r.probe_api.encode_pair(model.context_encoder,prepared['row'])
    assert torch.equal(features[0]['observed_H'].view(torch.int32),observed.view(torch.int32))
    assert torch.equal(features[0]['target_H'].view(torch.int32),target.view(torch.int32))
    assert audit['feature_mode']==mode and audit['readout_encoder']=='online'
    assert state==r.state_sha256(model) and buffers==r.state_sha256(dict(model.named_buffers()))
    assert torch.equal(rng,torch.get_rng_state())
    model.context_encoder.train()
    with pytest.raises(ValueError):r.encode_readout_rows(model,[prepared])


def fit_inputs():
    pairs=[row(i) for i in range(400)]
    return [p for p,f in pairs],[f for p,f in pairs]


def test_real_head_two_fresh500_fits_exact_order_initial_and_final_parity():
    prepared,features=fit_inputs();logs=[]
    h,a,p=r.fit_readout(prepared,features,logs.append)
    h2,a2,p2=r.fit_readout(prepared,features,lambda record:json.dumps(record,allow_nan=False))
    assert len(logs)==500 and [x['cache_row_index'] for x in logs]==list(range(400))+list(range(100))
    assert [x['step'] for x in logs]==list(range(1,501)) and all(x['lr']==.001 for x in logs)
    assert a['optimizer_steps']==500 and a==a2
    assert not h.training and not h2.training
    assert all(torch.equal(v,p2['head'][k]) for k,v in p['head'].items())
    assert all(float(s['step'])==500 for s in p['optimizer']['state'].values())
    assert all(isinstance(x['log_mse'],float) for x in logs)


@pytest.mark.parametrize('fault',['heldout','length','gradient_feature','nonfinite_feature'])
def test_fit_rejects_invalid_panel_before_head_creation(monkeypatch,fault):
    prepared,features=fit_inputs()
    if fault=='heldout':prepared[0]['cache_row_index']=400
    elif fault=='length':prepared.pop()
    elif fault=='gradient_feature':features[0]['target_H'].requires_grad_(True)
    else:features[0]['target_H'][0,0,0]=float('nan')
    monkeypatch.setattr(r.probe_api,'new_probe',lambda **kw:pytest.fail('must fail before head construction'))
    with pytest.raises(ValueError):r.fit_readout(prepared,features,lambda _:None)


def test_fit_failure_retains_actual_completed_update_and_detached_state():
    prepared,features=fit_inputs()
    def fail(record):
        if record['step']==3:raise RuntimeError('synthetic emit failure')
    with pytest.raises(RuntimeError,match='emit failure') as caught:r.fit_readout(prepared,features,fail)
    payload=caught.value.readout_failure_payload
    assert payload['completed_updates']==3 and all(float(s['step'])==3 for s in payload['optimizer']['state'].values())
    assert all(not x.requires_grad for x in payload['head'].values())


def test_score_negative_predictions_raw_targets_and_same_head_controls():
    prepared,features=row(400);head=r.probe_api.new_probe().eval()
    with torch.no_grad():
        for p in head.parameters():p.zero_()
        head.net[-1].bias.fill_(-.5)
    result=r.score_readout(head,features,prepared,predicted_H=features['observed_H']*2)
    assert set(result['scores'])==set(r.ROLES)|{'CA_persistence'}
    target=prepared['labels']['target_distance'].numpy().astype(float)
    pred=np.expm1(np.float64(-.5));e=pred-target
    for role in r.ROLES:
        score=result['scores'][role]
        assert score['negative_count']==1
        assert score['mae_angstrom']==pytest.approx(np.abs(e).mean(),abs=1e-12)
        assert score['mse_angstrom2']==pytest.approx(np.mean(e**2),abs=1e-12)
        assert result['arrays'][role+'__predicted_log'].dtype==np.float32
    assert result['scores']['CA_persistence']['negative_count']==0
    with torch.no_grad():head.net[-1].bias.fill_(1000.)
    with pytest.raises(ValueError,match='nonfinite'):r.score_readout(head,features,prepared)


def records():
    result=[]
    for i in range(100):
        n=1 if i%2==0 else 1000
        scores={}
        for arm,factor in zip(r.ARMS,(1.,2.,.5)):
            for role in r.ROLES:
                mse=factor*(i%7)**2
                scores[arm+'/'+role]=dict(log_mse=factor*(i%7),mae_angstrom=factor*(i%5),
                    mse_angstrom2=mse,rmse_angstrom=math.sqrt(mse),n_pairs=n,negative_count=0)
        scores['CA_persistence']=copy.deepcopy(scores['pca/future_H'])
        result.append(dict(cache_row_index=400+i,metadata=dict(protein=f'p{400+i}',horizon=(1,4,16,64)[i%4]),
            n_pairs=n,scores=scores))
    return result


def test_equal_protein_aggregation_not_pair_pooling_or_mean_roots():
    rows=records();a=r.aggregate_readouts(rows)
    values=np.array([x['scores']['pca/future_H']['mse_angstrom2'] for x in rows])
    actual=a['overall']['pca/future_H']
    assert actual['rmse_angstrom']==pytest.approx(np.sqrt(values.mean()))
    assert actual['rmse_angstrom']!=pytest.approx(np.sqrt(values).mean())
    assert actual['mse_angstrom2']!=pytest.approx(np.average(values,weights=[x['n_pairs'] for x in rows]))
    assert set(a)=={'overall','h1','h4','h16','h64'}
    assert all(a['h'+str(h)]['pca/future_H']['n_proteins']==25 for h in r.HORIZONS)


def test_stratified_paired_bootstrap_matches_independent_numpy_and_rmse_transform():
    rows=records();before=np.random.get_state();result=r.paired_bootstrap(rows)
    rng=np.random.default_rng(20261102)
    selected=np.concatenate([np.arange(offset,100,4)[rng.integers(25,size=(10000,25))] for offset in range(4)],axis=1)
    base=np.array([i%7 for i in range(100)],float)
    dist=base[selected].mean(axis=1)  # zero-pca log-MSE; zero=2*pca.
    q=np.percentile(dist,[2.5,97.5],method='linear')
    actual=result['contrasts']['zero-pca']['future_H']['log_mse']
    np.testing.assert_allclose(actual['interval95'],q,rtol=0,atol=1e-14)
    assert actual['difference']==pytest.approx(base.mean())
    mses=(base**2)[selected].mean(axis=1)
    expected=(np.sqrt(2*mses)-np.sqrt(mses))
    np.testing.assert_allclose(result['contrasts']['zero-pca']['future_H']['rmse_angstrom']['interval95'],
        np.quantile(expected,[.025,.975]),rtol=0,atol=1e-14)
    assert result['primary']=='future_H/log_mse' and result['draws']==10000
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_zero_scores_and_identical_arms_have_literal_zero_intervals():
    rows=records()
    for row_ in rows:
        for score in row_['scores'].values():
            for key in r.METRICS:score[key]=0.
    result=r.paired_bootstrap(rows)
    for contrast in result['contrasts'].values():
        for role in contrast.values():
            for metric in role.values():assert metric==dict(difference=0.,interval95=[0.,0.])


@pytest.mark.parametrize('fault',['order','duplicate','horizon','missing_role','pair_count','wrong_root','nan'])
def test_aggregate_and_bootstrap_fail_closed(fault):
    rows=records()
    if fault=='order':rows[0],rows[1]=rows[1],rows[0]
    elif fault=='duplicate':rows[1]['metadata']['protein']=rows[0]['metadata']['protein']
    elif fault=='horizon':rows[0]['metadata']['horizon']=64
    elif fault=='missing_role':rows[0]['scores'].pop('local/future_H')
    elif fault=='pair_count':rows[0]['scores']['local/future_H']['n_pairs']=2
    elif fault=='wrong_root':rows[0]['scores']['local/future_H']['rmse_angstrom']=2.
    else:rows[0]['scores']['local/future_H']['log_mse']=float('nan')
    with pytest.raises(ValueError):r.aggregate_readouts(rows)
    with pytest.raises(ValueError):r.paired_bootstrap(rows)


def test_readout_cpu_default_and_autocast_guards_before_head_construction(monkeypatch):
    monkeypatch.setattr(r.probe_api,'new_probe',lambda **kw:pytest.fail('must reject before construction'))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match='no autocast'):r.fit_readout([],[],lambda _:None)
    monkeypatch.setattr(torch,'get_default_dtype',lambda:torch.float64)
    with pytest.raises(ValueError,match='float32'):r.fit_readout([],[],lambda _:None)
