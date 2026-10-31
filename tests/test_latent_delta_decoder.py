import torch
import pytest
from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder, BoundedEdgeCADecoder
from protein_jepa_frame.models.latent_delta_decoder import LatentDeltaCADecoder


@pytest.mark.parametrize("decoder_class", [LatentDeltaCADecoder, GeometricDeltaCADecoder])
def test_zero_increment_equivariance_and_dynamic_gradient(decoder_class):
    torch.manual_seed(7)
    model = decoder_class(depth=1).eval()
    h = torch.randn(1, 6, 128)
    future = (h + torch.randn_like(h)*0.1).requires_grad_()
    aa = torch.zeros(1, 6, dtype=torch.long)
    ri = torch.arange(6)[None]
    mask = torch.ones(1, 6, dtype=torch.bool)
    anchor = torch.randn(1, 6, 3)
    rotation = torch.eye(3).expand(1, 6, 3, 3)
    def decode(f, r, x):
        return model(f, h, aa, ri, aa, r, x, mask)
    assert torch.equal(decode(h, rotation, anchor), anchor)
    result = decode(future, rotation, anchor)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    translation = torch.randn(3)
    transformed = decode(future, q @ rotation, anchor @ q.T + translation)
    torch.testing.assert_close(transformed, result @ q.T + translation, atol=2e-6, rtol=2e-6)
    (result-anchor).square().sum().backward()
    assert torch.isfinite(future.grad).all() and future.grad.abs().sum() > 0


def test_bounded_edge_identity_equivariance_bonds_and_gradient():
    torch.manual_seed(19)
    model = BoundedEdgeCADecoder(depth=1).double().eval()
    h = torch.randn(1, 8, 128, dtype=torch.float64)
    future = (h+torch.randn_like(h)).requires_grad_()
    aa = torch.zeros(1, 8, dtype=torch.long)
    ri = torch.arange(8)[None]
    mask = torch.ones_like(ri, dtype=torch.bool)
    anchor = torch.zeros(1, 8, 3, dtype=torch.float64)
    anchor[0, :, 0] = torch.arange(8)*3.8
    rotation = torch.eye(3, dtype=h.dtype).expand(1, 8, 3, 3)
    assert torch.equal(model(h,h,aa,ri,aa,rotation,anchor,mask),anchor)
    result = model(future,h,aa,ri,aa,rotation,anchor,mask)
    lengths = result.diff(dim=1).norm(dim=-1)
    assert bool(((lengths >= 2.5) & (lengths <= 5.)).all())
    q, _ = torch.linalg.qr(torch.randn(3,3,dtype=h.dtype))
    moved = model(future,h,aa,ri,aa,q@rotation,anchor@q.T+3,mask)
    torch.testing.assert_close(moved,result@q.T+3)
    result.square().sum().backward()
    assert torch.isfinite(future.grad).all() and future.grad.norm()>0


def test_geometric_eval_is_finite_with_padding_and_matches_grad_path():
    torch.manual_seed(18)
    model=GeometricDeltaCADecoder(depth=1).eval()
    h=torch.randn(1,7,128);f=h+torch.randn_like(h)*.1
    aa=torch.zeros(1,7,dtype=torch.long);ri=torch.arange(7)[None]
    rotation=torch.eye(3).expand(1,7,3,3);anchor=torch.randn(1,7,3)
    mask=torch.tensor([[True,True,True,True,True,False,False]])
    expected=model(f,h,aa,ri,aa,rotation,anchor,mask)
    with torch.no_grad():actual=model(f,h,aa,ri,aa,rotation,anchor,mask)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected)
