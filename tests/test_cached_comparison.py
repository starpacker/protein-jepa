import copy

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from evaluation.geometry import physical_distance_energy_score
from protein_jepa_frame.models.stage_b1a import StageBOracleConfig, StageBOracleDecoder
from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
from protein_jepa_frame.training.train_cached_comparison import (
    DirectJEPAPredictor, endpoint_loss, predict, window,
)


def record():
    torch.manual_seed(81)
    return dict(H=torch.randn(20,8,16),coords=torch.randn(20,8,4,3),
        residue_mask=torch.ones(20,8,dtype=torch.bool),time_ps=torch.arange(20)*1000.,
        aatype=torch.zeros(8,dtype=torch.long),residue_index=torch.arange(8),
        chain_id=torch.zeros(8,dtype=torch.long),protein_id='p')


def oracle():
    net=StageBOracleDecoder(StageBOracleConfig(latent_dim=16,decoder_c_hidden=16,
        decoder_n_heads=2,decoder_n_ipa_blocks=1,decoder_dropout=0.))
    net.h_normalizer.fit(torch.randn(64,16)*2+3)
    return net.eval().requires_grad_(False)


def test_cached_windows_preserve_time_boundary_and_reject_out_of_partition():
    r=record()
    w=window(r,[7,10],N=2)
    assert torch.equal(w['history'][:,-1],r['H'][[7,10]])
    assert torch.equal(w['target'][:,0],r['H'][[8,11]])
    with pytest.raises(ValueError,match='within its partition'):
        window(r,[6],N=2)
    r['residue_mask'][8,0]=False
    with pytest.raises(ValueError,match='future topology'):
        window(r,[7],N=2)


@pytest.mark.parametrize('bath',[0,2])
def test_cached_autonomous_prediction_invariant_to_all_future_labels(bath):
    w=window(record(),[7],N=2)
    other=copy.deepcopy(w)
    other['target'].add_(1000)
    other['batch']['coords'][:,8:].add_(1000)
    net=StageB1b(StageB1bConfig(latent_dim=16,bath_dim=bath,energy_hidden=16,
        energy_depth=1,energy_heads=2,history_hidden=16,max_gamma_step=1.9))
    with torch.no_grad(),sdpa_kernel([SDPBackend.MATH]):
        a=predict(net,'h_markov' if bath==0 else 'h_plus_s',w,
            generator=torch.Generator().manual_seed(91),training=False)
        b=predict(net,'h_markov' if bath==0 else 'h_plus_s',other,
            generator=torch.Generator().manual_seed(91),training=False)
    assert torch.equal(a,b)


def test_frozen_decoder_sees_predicted_H_and_gradient_reaches_energy_bath_posterior(monkeypatch):
    w=window(record(),[7],N=1)
    net=StageB1b(StageB1bConfig(latent_dim=16,bath_dim=2,energy_hidden=16,
        energy_depth=1,energy_heads=2,history_hidden=16,max_gamma_step=1.9))
    dec=oracle()
    seen=[]
    original=dec.decode_structure
    def spy(H,*args,**kwargs):
        seen.append(H.detach().clone())
        return original(H,*args,**kwargs)
    monkeypatch.setattr(dec,'decode_structure',spy)
    with sdpa_kernel([SDPBackend.MATH]):
        smp=predict(net,'h_plus_s',w,generator=torch.Generator().manual_seed(2),training=True)
        loss,logs=endpoint_loss(smp,w,dec)
        loss.backward()
    assert torch.allclose(seen[0],dec.h_normalizer.inverse(smp).reshape(2,8,16))
    assert torch.allclose(seen[1],dec.h_normalizer.inverse(w['history'][:,-1]))
    for prefix in ('energy.','bath.','history_posterior.'):
        grads=[p.grad for n,p in net.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads)>0
    assert all(p.grad is None for p in dec.parameters())


def test_direct_endpoint_uses_history_and_supports_same_detached_label_loss():
    w=window(record(),[7],N=2)
    net=DirectJEPAPredictor(latent_dim=16,depth=1,spatial_depth=1,num_heads=2)
    with sdpa_kernel([SDPBackend.MATH]):
        smp=predict(net,'direct_jepa',w,generator=None,training=True)
        assert smp.shape==(2,1,1,8,16)
        assert torch.equal(smp[0],smp[1])
        loss,_=endpoint_loss(smp,w,oracle())
        loss.backward()
    assert net.network.out_head.weight.grad.abs().sum()>0


def test_physical_score_tensor_matches_report_and_has_finite_coordinate_gradient():
    torch.manual_seed(8)
    x=torch.randn(3,8,3,requires_grad=True)
    y=torch.randn(8,3)
    mask=torch.ones(8,dtype=torch.bool)
    score=physical_distance_energy_score(x,y,mask,return_tensor=True)
    assert float(score.detach())==pytest.approx(physical_distance_energy_score(x.detach(),y,mask))
    score.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
