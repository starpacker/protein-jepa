import pytest
import torch
from protein_jepa_frame.models.ca_geometry import retract_ca_bond_lengths


def test_retraction_preserves_directions_centroids_gaps_and_gradients():
    torch.manual_seed(27)
    anchor = torch.randn(2, 8, 3, dtype=torch.float64).cumsum(1)
    pred = (anchor + torch.randn_like(anchor)*.2).requires_grad_()
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[:, -1] = False
    ri = torch.arange(8)[None].expand(2, -1).clone()
    ri[:, 4:] += 3
    ci = torch.zeros_like(ri)
    out = retract_ca_bond_lengths(pred, anchor, mask, ri, ci)
    valid = mask[:, 1:] & mask[:, :-1] & (ri.diff(dim=1) == 1)
    edges = out.diff(dim=1)
    torch.testing.assert_close(edges.norm(dim=-1)[valid], anchor.diff(dim=1).norm(dim=-1)[valid])
    torch.testing.assert_close(torch.nn.functional.normalize(edges, dim=-1)[valid],
                               torch.nn.functional.normalize(pred.diff(dim=1), dim=-1)[valid])
    for first, last in ((0, 4), (4, 7), (7, 8)):
        torch.testing.assert_close(out[:, first:last].mean(1), pred[:, first:last].mean(1))
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=pred.dtype))
    shift = torch.randn(3, dtype=pred.dtype)
    moved = retract_ca_bond_lengths(pred@q.T+shift, anchor@q.T+shift, mask, ri, ci)
    torch.testing.assert_close(moved, out@q.T+shift)
    assert torch.autograd.gradcheck(lambda x: retract_ca_bond_lengths(x, anchor, mask, ri, ci),
                                   (pred,), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(retract_ca_bond_lengths(out, anchor, mask, ri, ci), out)


def test_retraction_rejects_singular_valid_edges():
    x = torch.zeros(1, 3, 3)
    anchor = torch.tensor([[[0., 0., 0.], [3.8, 0., 0.], [7.6, 0., 0.]]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    ri = torch.arange(3)[None]
    with pytest.raises(ValueError, match='singular'):
        retract_ca_bond_lengths(x, anchor, mask, ri, torch.zeros_like(ri))

from protein_jepa_frame.models.ca_geometry import project_ca_bonds


def test_projection_reduces_bond_error_preserves_identity_and_gaps():
    torch.manual_seed(11)
    anchor=torch.randn(1,12,3).cumsum(1)
    mask=torch.ones(1,12,dtype=torch.bool)
    ri=torch.arange(12)[None];ci=torch.zeros_like(ri)
    ri[:,6:]+=5
    assert torch.equal(project_ca_bonds(anchor,anchor,mask,ri,ci),anchor)
    pred=(anchor+torch.randn_like(anchor)*.2).requires_grad_()
    result=project_ca_bonds(pred,anchor,mask,ri,ci,32)
    valid=ri.diff(dim=1)==1
    def error(x):return ((x.diff(dim=1).norm(dim=-1)-anchor.diff(dim=1).norm(dim=-1)).abs()*valid).sum()
    assert error(result)<error(pred)*.1
    result.square().sum().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0
    q,_=torch.linalg.qr(torch.randn(3,3));offset=torch.randn(3)
    rotated=project_ca_bonds(pred.detach()@q.T+offset,anchor@q.T+offset,mask,ri,ci,32)
    torch.testing.assert_close(rotated,result.detach()@q.T+offset,atol=1e-6,rtol=1e-5)


def test_refined_projection_gradient_matches_directional_difference():
    generator = torch.Generator().manual_seed(26)
    anchor = torch.zeros(1, 12, 3, dtype=torch.float64)
    anchor[0, :, 0] = torch.arange(12, dtype=torch.float64)*3.8
    mask = torch.ones(1, 12, dtype=torch.bool)
    ri = torch.arange(12)[None]
    ci = torch.zeros_like(ri)
    raw = (anchor + torch.randn(anchor.shape, generator=generator, dtype=anchor.dtype)*2).requires_grad_()
    direction = torch.randn(raw.shape, generator=generator, dtype=raw.dtype)
    direction = direction/direction.norm()
    probe = torch.randn(raw.shape, generator=generator, dtype=raw.dtype)
    for iterations in (32, 128, 256, 512):
        def objective(value):
            return (project_ca_bonds(value, anchor, mask, ri, ci, iterations)*probe).sum()
        gradient, = torch.autograd.grad(objective(raw), raw)
        epsilon = 1e-5
        difference = (objective(raw.detach()+epsilon*direction)
                      - objective(raw.detach()-epsilon*direction))/(2*epsilon)
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close((gradient*direction).sum(), difference, atol=1e-6, rtol=1e-4)
