"""Equivariant latent-delta CA probe with explicit anchor pair geometry.

Scalar attention coefficients multiply anchor inter-residue vectors. The decoder
can express cooperative motion in an anchor-derived basis without learning the
relationship between independent residue frames from sequence alone.
"""
import math
import torch
from torch import nn


class GeometricDeltaCADecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden=128, depth=3):
        super().__init__()
        self.input = nn.Linear(2*latent_dim,hidden)
        self.aa = nn.Embedding(21,hidden)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            hidden,4,hidden*2,dropout=0.,batch_first=True,norm_first=True)
            for _ in range(depth)])
        self.query = nn.Linear(hidden,32,bias=False)
        self.key = nn.Linear(hidden,32,bias=False)
        self.pair = nn.Sequential(nn.Linear(18,32),nn.SiLU(),nn.Linear(32,1))
        self.gate = nn.Linear(hidden,1)
        nn.init.normal_(self.gate.weight,std=1e-3)
        nn.init.zeros_(self.gate.bias)
        self.register_buffer('centers',torch.linspace(2.,30.,16))

    def forward(self,future_h,anchor_h,aatype,residue_index,chain_id,rotation,anchor,mask):
        delta=future_h-anchor_h
        x=self.input(torch.cat((torch.cat((anchor_h,delta),-1),
                                     torch.cat((anchor_h,torch.zeros_like(delta)),-1)),0))
        x=x+self.aa(aatype).repeat(2,1,1)
        vectors=anchor[:,None,:,:]-anchor[:,:,None,:]  # vector i -> j
        distance=vectors.norm(dim=-1)
        sep=(residue_index[:,:,None]-residue_index[:,None,:]).abs().float()
        same=(chain_id[:,:,None]==chain_id[:,None,:]).float()
        rbf=torch.exp(-((distance[...,None]-self.centers)/2.).square())
        pair=self.pair(torch.cat((rbf,(sep/32).clamp(max=1)[...,None],same[...,None]),-1)).squeeze(-1)
        bias=pair.repeat(2,1,1)
        for layer in self.layers:
            # Bias shape [2B*nheads,L,L]; geometry comes only from observed anchor.
            attention_bias=bias.masked_fill(~mask.repeat(2,1)[:,None,:],float('-inf'))
            # Avoid eval fastpath coercing continuous attention biases to bool.
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
        # Bounded vector lengths keep a few remote pairs from dominating.
        basis=vectors*(distance.clamp_min(1e-6).clamp(max=20)/distance.clamp_min(1e-6))[...,None]
        displacement=((changed-fixed)[...,None]*basis).sum(-2)*mask[...,None]
        return anchor+displacement


class BoundedEdgeCADecoder(GeometricDeltaCADecoder):
    """Train through bounded variable-length edges, with exact anchor identity.

    A smooth log-length map retains the anchor length at zero displacement.
    Segment means fix the translation gauge. Chain reconstruction still has
    length-dependent conditioning; this representation does not claim to remove it.
    """
    def forward(self, future_h, anchor_h, aatype, residue_index, chain_id, rotation, anchor, mask):
        raw = super().forward(future_h, anchor_h, aatype, residue_index, chain_id, rotation, anchor, mask)
        valid = (mask[:, 1:] & mask[:, :-1]
                 & (chain_id[:, 1:] == chain_id[:, :-1])
                 & (residue_index[:, 1:]-residue_index[:, :-1] == 1))
        edge = raw.diff(dim=1)
        reference = anchor.diff(dim=1).norm(dim=-1)
        distance = edge.norm(dim=-1)
        if bool(((reference <= 2.5) | (reference >= 5.) | (distance <= 1e-6))[valid].any()):
            raise ValueError('bounded edge requires nonsingular edges and anchor lengths inside (2.5,5)')
        # Logistic in log-length ratio: derivative of output length w.r.t.
        # input length is one at the anchor, rather than an arbitrary scale.
        fraction = ((reference-2.5)/2.5).clamp(1e-6, 1-1e-6)
        slope = reference.clamp_min(1e-6)/(2.5*fraction*(1-fraction))
        log_ratio = torch.log(distance.clamp_min(1e-6)/reference.clamp_min(1e-6))
        base_logit = torch.logit(fraction)
        length = reference+2.5*(torch.sigmoid(base_logit+slope*log_ratio)-torch.sigmoid(base_logit))
        # Residual form gives bitwise anchor identity when raw equals anchor.
        correction = edge*(length/distance.clamp_min(1e-6)-1)[..., None]
        batches = []
        for b in range(raw.shape[0]):
            segments = []
            start = 0
            for end in (~valid[b]).nonzero().flatten().tolist()+[raw.shape[1]-1]:
                original = raw[b, start:end+1]
                shift = torch.cat((torch.zeros_like(original[:1]), correction[b, start:end].cumsum(0)), 0)
                segments.append(original+shift-shift.mean(0))
                start = end+1
            batches.append(torch.cat(segments, 0))
        return torch.stack(batches)
