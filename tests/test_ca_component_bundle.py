import torch
from torch import nn
from protein_jepa_frame.models.ca_component_bundle import CAComponentBundle
from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder


def test_bundle_generated_latent_has_gradient_with_frozen_components():
    bundle=CAComponentBundle.__new__(CAComponentBundle)
    nn.Module.__init__(bundle)
    bundle.decoder=GeometricDeltaCADecoder(depth=1).eval()
    bundle.projection_steps=4
    for parameter in bundle.parameters():parameter.requires_grad_(False)
    torch.manual_seed(21)
    h=torch.randn(1,8,128)
    generated=(h+.1*torch.randn_like(h)).requires_grad_()
    aa=torch.zeros(1,8,dtype=torch.long);ri=torch.arange(8)[None]
    rotation=torch.eye(3).expand(1,8,3,3);anchor=torch.randn(1,8,3)
    mask=torch.ones(1,8,dtype=torch.bool)
    pred=bundle.decode_ca(generated,h,aa,ri,aa,rotation,anchor,mask)
    pred.square().sum().backward()
    assert torch.isfinite(generated.grad).all() and generated.grad.abs().sum()>0
    assert all(p.grad is None for p in bundle.parameters())
