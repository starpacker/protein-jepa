import torch
from protein_jepa_frame.training.train_lasp_calibration import calibration_terms,teacher_terms


def fixture():
    torch.manual_seed(12)
    x=torch.randn(1,5,8,3)
    mask=torch.ones(1,8,dtype=torch.bool)
    return x,mask,torch.arange(8)[None],torch.zeros(1,8,dtype=torch.long)


def test_reconstruction_identity_and_dynamic_gradient():
    x,m,ri,ci=fixture()
    exact=calibration_terms(x,x,m,ri,ci)
    assert all(v.abs()<1e-6 for k,v in exact.items() if "clash" not in k)
    assert exact["ca_clash_fraction"] == exact["true_ca_clash_fraction"]
    pred=x[:,:1].expand_as(x).clone().requires_grad_()
    terms=calibration_terms(pred,x,m,ri,ci)
    assert terms['delta_distance']>0
    terms['delta_distance'].backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0


def test_padding_and_nonlocal_pair_mask():
    x,m,ri,ci=fixture();m[:,-1]=False
    pred=x.clone();pred[:,:,-1]+=100
    terms=calibration_terms(pred,x,m,ri,ci)
    assert all(v.abs()<1e-6 for k,v in terms.items() if "clash" not in k)
    assert terms["ca_clash_fraction"] == terms["true_ca_clash_fraction"]


def test_teacher_penalizes_erasing_conformation_delta():
    torch.manual_seed(3)
    teacher=torch.randn(1,5,6,4)
    student=teacher[:,:1].expand_as(teacher).clone().requires_grad_()
    absolute,delta=teacher_terms(student,teacher,torch.ones(1,6,dtype=torch.bool))
    assert absolute>0 and torch.allclose(delta,torch.ones_like(delta))
    (absolute+delta).backward()
    assert torch.isfinite(student.grad).all() and student.grad.abs().sum()>0


def test_checkpoint_selection_rejects_geometry_regression():
    from protein_jepa_frame.training.train_lasp_calibration import selection_score
    assert selection_score({'mean': {'ca_rmsd': 1., 'ca_bond': .2}}, .1) == float('inf')
    assert selection_score({'mean': {'ca_rmsd': 2., 'ca_bond': .05}}, .1) == 2.
