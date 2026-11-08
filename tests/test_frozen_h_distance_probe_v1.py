"""Synthetic checkpoints, coordinates and features only; no actual inputs."""
import copy
import math

import numpy as np
import pytest
import torch

from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.stage_c import frozen_h_distance_probe_v1 as probe


def encoder_fixture(*,orientation=True,relpos=True):
    config=dict(latent_dim=128,encoder_depth=1,num_heads=4,num_rbf=8,dropout=.1,
        max_residues=32,orientation_features=orientation,relative_position_features=relpos,
        identity_embed_scale=.1,struct_recon_dim=384)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        encoder=ProteinGeometryEncoder(latent_dim=128,depth=1,num_heads=4,num_rbf=8,
            dropout=.1,max_residues=32,orientation_features=orientation,
            relative_position_features=relpos,identity_embed_scale=.1)
    return encoder.eval().requires_grad_(False),config


def row_fixture():
    g=torch.Generator().manual_seed(82)
    observed=torch.randn(1,6,4,3,generator=g)
    observed+=torch.arange(6)[None,:,None,None]*torch.tensor([3.,.2,.1])
    target=observed+torch.randn(1,6,4,3,generator=g)*.2
    return dict(observed_backbone=observed,target_backbone=target,
        observed_mask=torch.tensor([True,True,True,True,True,False]),
        atom_mask=torch.ones(6,4,dtype=torch.bool),target_atom_mask=torch.ones(6,4,dtype=torch.bool),
        embedding_aatype=torch.tensor([0,1,2,3,4,20]),embedding_residue_index=torch.tensor([1,2,2,4,5,-1]),
        embedding_chain_id=torch.tensor([0,0,0,0,0,-1]),
        oracle_H=torch.full((1,6,128),float('nan')),anchor_H='FORBIDDEN cached raw H')


def checkpoint_fixture(kind):
    encoder,config=encoder_fixture()
    if kind=='original':
        checkpoint=dict(model={**{'context_encoder.'+k:v for k,v in encoder.state_dict().items()},
            'struct_recon_head.0.weight':torch.full((384,384),float('nan')),
            'teacher_encoder.unused':torch.ones(2)},stage_a_v2_config=config)
    else:
        checkpoint=dict(component_checkpoint=dict(encoder=encoder.state_dict(),stage_a_v2_config=config,
            oracle='must never construct',h_normalizer='must never apply'),kernel='must never construct')
    return checkpoint,encoder


@pytest.mark.parametrize('kind',['original','current'])
def test_direct_strict_frozen_encoder_load_ignores_other_models_and_preserves_rng(tmp_path,kind):
    checkpoint,reference=checkpoint_fixture(kind)
    path=tmp_path/'synthetic.pt';torch.save(checkpoint,path)
    rng=torch.get_rng_state().clone()
    encoder,audit=probe.load_frozen_encoder(path,kind=kind,expected_sha256=probe.sha256_file(path))
    assert type(encoder) is ProteinGeometryEncoder
    assert torch.equal(torch.get_rng_state(),rng)
    assert not any(m.training for m in encoder.modules()) and not any(p.requires_grad for p in encoder.parameters())
    assert probe.state_sha256(encoder)==probe.state_sha256(reference)==audit['encoder_tensor_sha256']
    assert audit['historical_struct_recon_dim']==384 and audit['fresh_probe_hidden_dim']==256
    assert audit['historical_head_loaded'] is False


def test_checkpoint_hash_rejection_precedes_deserialization(tmp_path,monkeypatch):
    path=tmp_path/'synthetic';path.write_bytes(b'not a checkpoint')
    monkeypatch.setattr(probe.torch,'load',lambda *a,**k:pytest.fail('deserialized before pin check'))
    with pytest.raises(ValueError,match='before deserialization'):
        probe.load_frozen_encoder(path,kind='original',expected_sha256='0'*64)


@pytest.mark.parametrize('orientation,relpos',[(False,False),(True,False),(False,True),(True,True)])
def test_actual_flags_from_state_preserve_saved_mismatch_audit(orientation,relpos):
    encoder,config=encoder_fixture(orientation=orientation,relpos=relpos)
    config.update(orientation_features=not orientation,relative_position_features=not relpos)
    checkpoint=dict(model={'context_encoder.'+k:v for k,v in encoder.state_dict().items()},stage_a_v2_config=config)
    state,kwargs,audit=probe.encoder_payload(checkpoint,'original')
    assert kwargs['orientation_features']==orientation and kwargs['relative_position_features']==relpos
    assert audit['saved_flags']['orientation_features'] != audit['actual_flags']['orientation_features']
    restored=ProteinGeometryEncoder(**kwargs);restored.load_state_dict(state,strict=True)
    assert probe.state_sha256(restored)==probe.state_sha256(encoder)


@pytest.mark.parametrize('mutation',['anchor_weight','anchor_config','dimension','missing_config','nonfinite','unknown_lineage','no_prefix'])
def test_undeclared_or_invalid_encoder_payload_fails(mutation):
    checkpoint,_=checkpoint_fixture('original');kind='original'
    if mutation=='anchor_weight':checkpoint['model']['context_encoder.anchor_pos_proj.weight']=torch.ones(128,3)
    elif mutation=='anchor_config':checkpoint['stage_a_v2_config']['anchor_conditioned']=True
    elif mutation=='dimension':checkpoint['stage_a_v2_config']['latent_dim']=64
    elif mutation=='missing_config':checkpoint['stage_a_v2_config'].pop('num_rbf')
    elif mutation=='nonfinite':checkpoint['model']['context_encoder.geom_gain'][0]=torch.nan
    elif mutation=='unknown_lineage':kind='teacher'
    else:checkpoint['model']={'teacher_encoder.x':torch.ones(3)}
    with pytest.raises(ValueError):probe.encoder_payload(checkpoint,kind)


def test_strict_state_rejects_unknown_encoder_key(tmp_path):
    checkpoint,_=checkpoint_fixture('original')
    checkpoint['model']['context_encoder.not_in_encoder']=torch.ones(3)
    path=tmp_path/'synthetic.pt';torch.save(checkpoint,path)
    with pytest.raises(RuntimeError,match='Unexpected key'):
        probe.load_frozen_encoder(path,kind='original',expected_sha256=probe.sha256_file(path))


def test_encode_pair_reencodes_aligned_coordinates_and_never_cached_H():
    encoder,_=encoder_fixture();row=row_fixture();calls=[]
    hook=encoder.register_forward_pre_hook(lambda module,args:calls.append(tuple(v.clone() for v in args)))
    before=probe.state_sha256(encoder);rng=torch.get_rng_state().clone()
    observed,target=probe.encode_pair(encoder,row);hook.remove()
    assert len(calls)==2 and observed.shape==target.shape==(1,6,128)
    assert torch.equal(calls[0][0][0,0,:5],row['observed_backbone'][0,:5])
    assert torch.equal(calls[1][0][0,0,:5],row['target_backbone'][0,:5])
    assert torch.equal(calls[0][3][0,:5],row['embedding_residue_index'][:5])  # duplicated legacy embedding is retained
    assert not observed.requires_grad and not target.requires_grad and not torch.equal(observed,target)
    assert torch.equal(torch.get_rng_state(),rng) and probe.state_sha256(encoder)==before
    assert all(p.grad is None for p in encoder.parameters())


def test_masked_NaNs_and_IDs_are_irrelevant_identically_for_both_calls():
    encoder,_=encoder_fixture();row=row_fixture()
    expected=probe.encode_pair(encoder,row)
    changed=copy.deepcopy(row)
    for field in ('observed_backbone','target_backbone'):changed[field][:,5]=torch.nan
    for field in ('embedding_aatype','embedding_residue_index','embedding_chain_id'):changed[field][5]=99999
    changed['atom_mask'][5]=False;changed['target_atom_mask'][5]=False
    actual=probe.encode_pair(encoder,changed)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    assert all(torch.equal(x[:,5],torch.zeros_like(x[:,5])) for x in actual)


@pytest.mark.parametrize('fault',['target_missing','observed_missing','target_nonfinite','observed_nonfinite','training','trainable'])
def test_fixed_support_or_encoder_mode_failures(fault):
    encoder,_=encoder_fixture();row=row_fixture()
    if fault=='target_missing':row['target_atom_mask'][1,3]=False
    elif fault=='observed_missing':row['atom_mask'][2,0]=False
    elif fault=='target_nonfinite':row['target_backbone'][0,0,1,0]=torch.nan
    elif fault=='observed_nonfinite':row['observed_backbone'][0,0,2,0]=torch.inf
    elif fault=='training':encoder.train()
    else:next(encoder.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):probe.encode_pair(encoder,row)


def test_fresh256_same_init_isolated_rng_and_no_shared_parameters():
    rng=torch.get_rng_state().clone();a=probe.new_probe();b=probe.new_probe()
    assert torch.equal(torch.get_rng_state(),rng)
    assert probe.state_sha256(a)==probe.state_sha256(b)
    assert [(m.in_features,m.out_features) for m in a.net if isinstance(m,torch.nn.Linear)]==[(384,256),(256,256),(256,1)]
    assert all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters()))
    with pytest.raises(ValueError,match='seed42'):probe.new_probe(43)


def test_symmetry_masked_mean_all_pairs_and_exact_manual_full_loss_gradient():
    head=probe.new_probe();other=copy.deepcopy(head)
    H=torch.randn(1,5,128,generator=torch.Generator().manual_seed(4));H[:,4]=torch.nan
    mask=torch.tensor([True,True,True,True,False]);pairs=torch.tensor([[0,1],[0,3],[1,2],[2,3]])
    target=torch.tensor([1.,2.,3.,4.])
    loss,pred=probe.probe_loss(head,H,mask,pairs,target)
    mean=H[0,:4].mean(0);manual=[]
    for i,j in pairs:
        left=other.net(torch.cat((H[0,i],H[0,j],mean)))
        right=other.net(torch.cat((H[0,j],H[0,i],mean)))
        manual.append((left+right).squeeze()*.5)
    manual=torch.stack(manual);reference=(manual-target).square().mean()
    torch.testing.assert_close(pred,manual,atol=3e-8,rtol=3e-6)
    loss.backward();reference.backward()
    for a,b in zip(head.parameters(),other.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=3e-7,rtol=3e-5)
    swapped=H.clone();swapped[:,0]=H[:,1];swapped[:,1]=H[:,0]
    pair=torch.tensor([[0,1]])
    torch.testing.assert_close(head(H,mask,pair),head(swapped,mask,pair),atol=2e-8,rtol=2e-6)
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())


@pytest.mark.parametrize('bad',['empty','diagonal','reverse','duplicate','unsorted','masked','outside','float'])
def test_pair_support_errors_fail(bad):
    head=probe.new_probe();H=torch.zeros(1,4,128);mask=torch.tensor([True,True,True,False])
    pairs={'empty':torch.empty(0,2,dtype=torch.long),'diagonal':torch.tensor([[1,1]]),
        'reverse':torch.tensor([[1,0]]),'duplicate':torch.tensor([[0,1],[0,1]]),
        'unsorted':torch.tensor([[1,2],[0,1]]),'masked':torch.tensor([[0,3]]),
        'outside':torch.tensor([[0,4]]),'float':torch.tensor([[0.,1.]])}[bad]
    with pytest.raises(ValueError):head(H,mask,pairs)


def test_targets_use_all_explicit_pairs_and_true_aligned_CA_manual_numpy():
    row=row_fixture();pairs=torch.tensor([[0,2],[0,4],[1,3]])
    labels=probe.pair_log_targets(row,pairs)
    i,j=pairs.numpy().T
    for field,key in [('observed_backbone','observed_distance'),('target_backbone','target_distance')]:
        ca=row[field][0,:,1].numpy()
        expected=np.sqrt(np.square(ca[i]-ca[j]).sum(-1))
        np.testing.assert_allclose(labels[key].numpy(),expected,rtol=1e-7,atol=2e-7)
    assert torch.equal(labels['target_log'],torch.log1p(labels['target_distance']))


def test_float64_scoring_manual_numpy_negative_predictions_no_clamp():
    raw=torch.tensor([-2.,0.,.3,3.],dtype=torch.float32)
    target=torch.tensor([.4,1.,2.,7.],dtype=torch.float32);log=torch.log1p(target)
    score=probe.score_predictions(raw,log,target_distance=target)
    prediction=np.expm1(raw.numpy().astype(np.float64));error=prediction-target.numpy().astype(np.float64)
    assert score['n_pairs']==4 and score['negative_count']==1
    assert score['mae_angstrom']==pytest.approx(np.abs(error).mean(),abs=1e-14)
    assert score['mse_angstrom2']==pytest.approx(np.square(error).mean(),abs=1e-14)
    assert score['rmse_angstrom']==pytest.approx(math.sqrt(np.square(error).mean()),abs=1e-14)
    assert score['log_mse']==pytest.approx(np.square(raw.numpy().astype(np.float64)-log.numpy().astype(np.float64)).mean())
    persistence=probe.distance_metrics(target,target)
    assert persistence['mse_angstrom2']==0 and persistence['negative_count']==0


@pytest.mark.parametrize('fault',['log_nan','expm1_overflow','squared_error_overflow','target_mismatch','target_negative'])
def test_nonfinite_scores_and_wrong_labels_fail(fault):
    raw=torch.tensor([0.,1.]);target=torch.tensor([1.,2.]);log=torch.log1p(target)
    if fault=='log_nan':raw[0]=torch.nan
    elif fault=='expm1_overflow':raw[0]=1000.
    elif fault=='squared_error_overflow':raw[0]=400.
    elif fault=='target_mismatch':log[0]+=.1
    else:target[0]=-.5;log=torch.log1p(target)
    with pytest.raises(ValueError):probe.score_predictions(raw,log,target_distance=target)
