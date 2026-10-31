import torch
from protein_jepa_frame.training.train_lasp_ca_dynamics import energy_score


def test_energy_score_spread_gradient_and_identical_sample_expectation():
    sample=torch.tensor([[-1.,0.],[1.,0.]],requires_grad=True)
    target=torch.tensor([[-2.,0.],[2.,0.]])
    loss=energy_score(sample,target)
    loss.backward()
    assert torch.isfinite(sample.grad).all() and sample.grad.abs().sum()>0
    # Unbiased pair term excludes self-distance and does not shrink all samples to mean.
    assert energy_score(target,target)<energy_score(torch.zeros_like(target),target)


def test_gle_energy_force_double_backward_math_attention():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
    torch.manual_seed(31)
    kernel=StageB1b(StageB1bConfig(energy_depth=1,bath_dim=0))
    h=torch.randn(2,6,128)
    mask=torch.ones(2,6,dtype=torch.bool)
    with sdpa_kernel(SDPBackend.MATH):
        path=kernel.rollout(h,mask,torch.tensor([320.,348.]),2,detach=False)
    path[:,-1].square().mean().backward()
    grads=[p.grad for p in kernel.energy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads)>0


def test_history_sampler_uses_only_last_four_observed_frames():
    from torch import nn
    from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
    from protein_jepa_frame.models.stage_b1b import StageB1b, StageB1bConfig
    class Bundle(nn.Module):
        def encode(self,coords,*args):
            return coords[:,:,1].repeat(1,1,43)[:,:,:128]
        def decode_ca(self,generated,anchor_h,aa,ri,ci,rotation,anchor,mask):
            return anchor+generated[:,:,:3]-anchor_h[:,:,:3]
    model=CADynamics.__new__(CADynamics);nn.Module.__init__(model)
    model.bundle=Bundle();model.history_conditioning=True
    model.kernel=StageB1b(StageB1bConfig(energy_depth=1,bath_dim=2))
    observed=torch.randn(1,8,6,4,3)
    mask=torch.ones(1,6,dtype=torch.bool);aa=torch.zeros(1,6,dtype=torch.long)
    ri=torch.arange(6)[None];temp=torch.tensor([320.])
    pred,_=model.sample(observed,mask,aa,ri,aa,temp,2,2,
        torch.Generator().manual_seed(17))
    pred.square().mean().backward()
    grads=[p.grad for p in model.kernel.history_posterior.parameters() if p.grad is not None]
    assert grads and sum(g.abs().sum() for g in grads)>0
    model.eval()
    changed=observed.clone();changed[:,:4]+=100
    with torch.no_grad():
        a,_=model.sample(observed,mask,aa,ri,aa,temp,2,2,torch.Generator().manual_seed(17))
        b,_=model.sample(changed,mask,aa,ri,aa,temp,2,2,torch.Generator().manual_seed(17))
    torch.testing.assert_close(a,b)


def test_energy_score_matches_empirical_distance_expectations():
    samples=torch.tensor([[0.,1.],[2.,1.],[4.,3.]],requires_grad=True)
    target=torch.tensor([[1.,0.],[3.,4.]])
    cross=(samples[:,None]-target[None]).norm(dim=-1).mean()
    pair=torch.stack([(samples[i]-samples[j]).norm() for i in range(3) for j in range(i+1,3)]).mean()
    torch.testing.assert_close(energy_score(samples,target),(cross-.5*pair)/(2**.5))


def test_brier_contact_objective_recovers_rare_event_probability():
    from protein_jepa_frame.training.train_lasp_ca_dynamics import contact_probability_loss
    probability=torch.tensor(.2,requires_grad=True)
    outcomes=torch.tensor([0.,0.,0.,0.,1.])
    score=contact_probability_loss(probability.expand_as(outcomes),outcomes,'brier')
    score.backward()
    assert probability.grad.abs()<1e-7
    assert score<contact_probability_loss(torch.zeros_like(outcomes),outcomes,'brier')


def test_slow_modes_identical_spread_and_generated_gradient():
    from protein_jepa_frame.training.train_lasp_ca_dynamics import slow_mode_scale_loss
    torch.manual_seed(91)
    target=torch.randn(8,6,3)
    assert slow_mode_scale_loss(target,target)<1e-6
    sample=(target*.3).requires_grad_()
    loss=slow_mode_scale_loss(sample,target)
    loss.backward()
    assert loss>0 and torch.isfinite(sample.grad).all() and sample.grad.abs().sum()>0


def test_anchor_noise_covariance_padding_and_rigid_motion():
    from protein_jepa_frame.training.train_lasp_ca_dynamics import anchor_noise_factor
    torch.manual_seed(5)
    anchor=torch.randn(2,8,3);mask=torch.ones(2,8,dtype=torch.bool);mask[:,-2:]=False
    factor=anchor_noise_factor(anchor,mask,.7,8.)
    cov=factor@factor.transpose(-1,-2)
    torch.testing.assert_close(cov.diagonal(dim1=-2,dim2=-1),torch.ones(2,8))
    assert cov[:,0,1].min()>.5
    torch.testing.assert_close(cov[:,-2:,:-2],torch.zeros(2,2,6))
    q,_=torch.linalg.qr(torch.randn(3,3))
    other=anchor_noise_factor(anchor@q+10,mask,.7,8.)
    torch.testing.assert_close(factor,other,atol=1e-6,rtol=1e-5)


def test_identity_spatial_factor_preserves_seeded_kernel_rollout():
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from protein_jepa_frame.models.stage_b1b import StageB1b,StageB1bConfig
    torch.manual_seed(47)
    kernel=StageB1b(StageB1bConfig(energy_depth=1,bath_dim=0)).eval()
    h=torch.randn(1,6,128);mask=torch.ones(1,6,dtype=torch.bool)
    with sdpa_kernel(SDPBackend.MATH):
        a=kernel.rollout(h,mask,torch.tensor([320.]),3,generator=torch.Generator().manual_seed(9))
        b=kernel.rollout(h,mask,torch.tensor([320.]),3,generator=torch.Generator().manual_seed(9),
            noise_factor=torch.eye(6)[None])
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_mode_distribution_detects_shape_at_equal_spread_and_has_gradient():
    from protein_jepa_frame.training.train_lasp_ca_dynamics import slow_mode_distribution_loss,slow_mode_scale_loss
    # Equal variance and mean, but different empirical one-dimensional distributions.
    target=torch.tensor([-1.,-1.,1.,1.])[:,None,None].expand(-1,2,3).clone()
    other=torch.tensor([-2.**.5,0.,0.,2.**.5])[:,None,None].expand(-1,2,3).clone().requires_grad_()
    assert slow_mode_scale_loss(other,target,k=1)<1e-6
    loss=slow_mode_distribution_loss(other,target,k=1)
    assert loss>slow_mode_distribution_loss(target,target,k=1)
    loss.backward()
    assert torch.isfinite(other.grad).all() and other.grad.abs().sum()>0
    shifted=slow_mode_distribution_loss(other.detach()+12,target+7,k=1)
    torch.testing.assert_close(shifted,loss.detach())


def test_collective_joint_step_matches_mobility_drift_and_shared_noise():
    from torch import nn
    from protein_jepa_frame.models.stage_b1b import StageB1b,StageB1bConfig
    class Bath(nn.Module):
        def forward(self,h,temp):
            return torch.full_like(h,.2),torch.full((*h.shape,1),.3),torch.full((*h.shape[:2],1),.4)
    kernel=StageB1b(StageB1bConfig(latent_dim=4,energy_heads=1,bath_dim=1,step_ps=.01))
    kernel.bath=Bath()
    kernel.compute_forces=lambda h,*args,**kwargs:-h
    h=torch.arange(8,dtype=torch.float32).reshape(1,2,4)/10
    s=torch.tensor([[[.2],[-.1]]]);mask=torch.ones(1,2,dtype=torch.bool)
    factor=torch.linalg.cholesky(torch.tensor([[[1.,.6],[.6,1.]]]))
    rng=torch.Generator().manual_seed(37)
    actual_h,actual_s=kernel.sample_step(h,s,mask,None,generator=rng,noise_factor=factor,collective_mobility=True)
    # Independent dense construction over residue x channel verifies all blocks.
    K=factor[0]@factor[0].T;M=.2*torch.kron(K.contiguous(),torch.eye(4))
    C=torch.zeros(8,2);C[:4,0]=.3;C[4:,1]=.3
    mobility=torch.cat((torch.cat((M,M@C),1),torch.cat((C.T@M,.4*torch.eye(2)+C.T@M@C),1)),0)
    state=torch.cat((h.flatten(),s.flatten()))
    rng=torch.Generator().manual_seed(37)
    noise_h=(.004**.5*torch.einsum('ij,jd->id',factor[0],torch.randn(h.shape,generator=rng)[0])).flatten()
    noise_s=C.T@noise_h+.008**.5*torch.randn(s.shape,generator=rng).flatten()
    expected=state-.01*mobility@state+torch.cat((noise_h,noise_s))
    torch.testing.assert_close(torch.cat((actual_h.flatten(),actual_s.flatten())),expected)
    assert torch.linalg.eigvalsh(mobility).min()>0
