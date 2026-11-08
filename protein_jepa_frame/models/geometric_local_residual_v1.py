"""Nested local-vector residual for the frozen geometric CA decoder contract.

This independent architecture keeps D0's parameter names and forward arithmetic.
It adds exactly 384 parameters: a zero-initialized, bias-free 128-to-3 map of
the same changed/fixed trunk tokens. ``rotation`` is the observed backbone's
column-basis frame, never a frame constructed from future coordinates. This
module returns raw CA; the existing bundle's external projection is unchanged.

The extra branch changes capacity. It does not establish a universal limitation
of D0, or an information bound for H. No checkpoint files are loaded here.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn

from protein_jepa_frame.models.geometric_delta_decoder import GeometricDeltaCADecoder


VERSION = 'geometric_local_residual_v1'


class GeometricLocalResidualCADecoderV1(GeometricDeltaCADecoder):
    """D0 raw CA plus ``R_observed @ W(z_future - z_anchor)`` on valid rows.

    ``hidden=128`` fixes the added parameter budget to 384. Old state keys are
    unchanged; the sole new key is ``local_residual.weight``. Restoring a new
    head uses ordinary strict ``load_state_dict``. Migrating D0 requires the
    explicitly named bridge, which rejects missing/extra keys and shape drift.
    """

    def __init__(self, latent_dim=128, hidden=128, depth=3):
        if type(hidden) is not int or hidden != 128:
            raise ValueError('v1 fixes the shared trunk width to 128 and adds exactly 384 parameters')
        if type(latent_dim) is not int or latent_dim < 1 or type(depth) is not int or depth < 0:
            raise ValueError('latent_dim must be positive and depth must be nonnegative integers')
        super().__init__(latent_dim=latent_dim, hidden=hidden, depth=depth)
        self.local_residual = nn.Linear(128, 3, bias=False)
        nn.init.zeros_(self.local_residual.weight)

    @classmethod
    def from_base_state_dict(cls, state, *, latent_dim=128, hidden=128, depth=3):
        """Explicitly copy a complete D0 state and append an exactly zero W.

        No filename, architecture inference, ignored key, or shared tensor is
        accepted. Construction preserves the caller's CPU RNG stream. All new
        parameters are trainable; freezing is the caller's explicit decision.
        """
        if not isinstance(state, Mapping) or not state:
            raise ValueError('a complete in-memory D0 state mapping is required')
        with torch.random.fork_rng(devices=[]):
            result = cls(latent_dim=latent_dim, hidden=hidden, depth=depth)
        expected = result.state_dict()
        base_keys = set(expected) - {'local_residual.weight'}
        if set(state) != base_keys:
            raise ValueError('D0 state keys differ; migration cannot drop or synthesize old weights')
        reference = state['input.weight']
        if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
            raise ValueError('D0 input.weight must be a floating tensor')
        for name in sorted(base_keys):
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                    or value.dtype != reference.dtype or value.device != reference.device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('D0 tensor shape/dtype/device/finite contract differs: '+name)
        result.to(device=reference.device, dtype=reference.dtype)
        migrated = {name: value.detach().clone() for name, value in state.items()}
        migrated['local_residual.weight'] = torch.zeros_like(result.local_residual.weight)
        result.load_state_dict(migrated, strict=True)
        return result

    @classmethod
    def from_base_decoder(cls, decoder):
        """Copy the exact production D0 class; reject subclasses with other semantics."""
        if type(decoder) is not GeometricDeltaCADecoder:
            raise TypeError('bridge requires the exact GeometricDeltaCADecoder class')
        if decoder.input.in_features % 2:
            raise ValueError('D0 latent-pair width must be even')
        result = cls.from_base_state_dict(decoder.state_dict(),
            latent_dim=decoder.input.in_features//2, hidden=decoder.input.out_features,
            depth=len(decoder.layers))
        result.train(decoder.training)
        return result

    def forward(self, future_h, anchor_h, aatype, residue_index, chain_id, rotation, anchor, mask):
        if (anchor.ndim != 3 or anchor.shape[-1] != 3 or mask.shape != anchor.shape[:2]
                or mask.dtype != torch.bool or mask.device != anchor.device
                or not bool(mask.any(-1).all())):
            raise ValueError('anchor [B,L,3] requires a nonempty boolean mask per window')
        if (rotation.shape != (*anchor.shape[:2],3,3) or rotation.dtype != anchor.dtype
                or rotation.device != anchor.device or not bool(torch.isfinite(rotation[mask]).all())):
            raise ValueError('observed rotation must be finite [B,L,3,3] on valid residues')

        # The D0 arithmetic below is intentionally identical to the production
        # implementation. Only the final token difference is additionally read.
        delta=future_h-anchor_h
        x=self.input(torch.cat((torch.cat((anchor_h,delta),-1),
                                     torch.cat((anchor_h,torch.zeros_like(delta)),-1)),0))
        x=x+self.aa(aatype).repeat(2,1,1)
        vectors=anchor[:,None,:,:]-anchor[:,:,None,:]
        distance=vectors.norm(dim=-1)
        sep=(residue_index[:,:,None]-residue_index[:,None,:]).abs().float()
        same=(chain_id[:,:,None]==chain_id[:,None,:]).float()
        rbf=torch.exp(-((distance[...,None]-self.centers)/2.).square())
        pair=self.pair(torch.cat((rbf,(sep/32).clamp(max=1)[...,None],same[...,None]),-1)).squeeze(-1)
        bias=pair.repeat(2,1,1)
        for layer in self.layers:
            attention_bias=bias.masked_fill(~mask.repeat(2,1)[:,None,:],float('-inf'))
            normalized=layer.norm1(x)
            attended=layer.self_attn(normalized,normalized,normalized,
                attn_mask=attention_bias.repeat_interleave(4,0),need_weights=False)[0]
            x=x+layer.dropout1(attended)
            normalized=layer.norm2(x)
            x=x+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(normalized)))))
        logits=(self.query(x)@self.key(x).transpose(-1,-2))/math.sqrt(32)+bias
        logits=logits.masked_fill(~mask.repeat(2,1)[:,None,:],float('-inf'))
        weights=logits.softmax(-1)*torch.tanh(self.gate(x))
        changed,fixed=weights.chunk(2)
        basis=vectors*(distance.clamp_min(1e-6).clamp(max=20)/distance.clamp_min(1e-6))[...,None]
        displacement=((changed-fixed)[...,None]*basis).sum(-2)*mask[...,None]
        raw=anchor+displacement

        changed_token,fixed_token=x.chunk(2)
        token_delta=torch.where(mask[...,None],changed_token-fixed_token,0.)
        # Padding frames are irrelevant; isolate even NaN padding rotations
        # before multiplication so they cannot contaminate outputs or gradients.
        observed_rotation=torch.where(mask[...,None,None],rotation,0.)
        local=self.local_residual(token_delta)
        residual=torch.einsum('blij,blj->bli',observed_rotation,local)
        return raw+residual
