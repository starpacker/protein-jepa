"""Cross-Residue Interactive Decoder (CRIDecoder) for the from-none Stage B pipeline.

Design rationale
----------------
``SimpleCADecoder`` and the original ``MessagePassingCADecoder`` share two
structural weaknesses identified in the diagnosis (skill plateaued at ~0.055):

1. **S is not used** — ``bath_dim`` is passed in the config but the default run
   has bath_dim=8; yet the feature list ``[H_future, H_anchor, dH]`` never
   appends S unless ``bath_dim > 0`` AND ``S_future`` is passed.  The GLE bath S
   carries temperature and dissipation information that shapes which modes relax
   and how fast.

2. **H and S are treated independently** — even when S is concatenated, it is
   just appended as extra features at the input.  The physical picture is that H
   and S are coupled: S is the "heat bath" that drives H.  A decoder that never
   cross-attends H to S loses that coupling.

3. **Cross-residue interaction is missing from the simple decoder** — proteins
   move collectively (domain motions, hinge bending, loop breathing), so the
   correct displacement of residue i depends on the state of ALL residues j.
   The original MP decoder added self-attention over residues but dropped S.

CRIDecoder fixes all three simultaneously:

  * **Joint node init**: project ``[H_future, H_anchor, dH, S_future]`` together
    so H and S interact from the very first linear layer.  ``S_future`` is
    optional; when absent the node dim is still ``3*dH``.
  * **H↔S cross-attention** in every layer: H attends to S and S attends to H
    before the standard residue self-attention step.  This lets each residue's
    latent state be modulated by the bath dynamics at that residue, and vice
    versa.
  * **Geometry-aware residue self-attention**: edge bias carries CA-distance RBF,
    sequence-separation RBF, ``cos(dH_i, dH_j)``, and ``S_i·S_j`` (bath
    coupling) so the attention pattern respects both sequence topology and the
    dynamic coupling between residues.
  * **SE(3) equivariance and zero-init** preserved: the output is a bounded
    local-frame displacement (``max_update * tanh``), mapped to the global frame
    via ``R_anchor``, zero-initialised so training starts from persistence.

It is a drop-in replacement: same ``forward`` signature and same output keys as
``SimpleCADecoder``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .decoder import FromNoneConfig


@dataclass
class MessagePassingConfig(FromNoneConfig):
    """Sizing for the message-passing CA decoder head."""

    mp_depth: int = 2            # number of residue message-passing layers
    mp_heads: int = 4            # attention heads per MP layer
    num_rbf: int = 16            # radial basis on anchor CA distance
    seqsep_rbf: int = 8          # radial basis on |i-j| (sequence separation)
    use_dh_cosine: bool = True   # edge feature: cos(dH_i, dH_j) (invariant)
    mp_chunk: int = 32           # chunk size over the flattened batch dim
    mp_dropout: float = 0.0
    # --- CRIDecoder-specific options (ignored by MessagePassingCADecoder) ---
    use_hs_cross: bool = True    # H<->S cross-attention inside every layer
    use_bath_coupling: bool = True   # edge feature: S_i . S_j
    hs_heads: int = 4            # heads for the H<->S cross-attention


def _pairwise_distance(x: torch.Tensor) -> torch.Tensor:
    """Safe pairwise Euclidean distance [N, L, L] with gradient-safe eps.

    ``torch.cdist`` produces NaN gradients when any pair has distance exactly
    zero (self-pairs, or padded residues that share the (0,0,0) fill value).
    We use the squared-distance formulation and add eps before sqrt to avoid
    the 1/0 at zero distance, then clamp the result to [0, 40] Å.
    """
    # [N, L, L] via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    x2 = (x * x).sum(dim=-1, keepdim=True)          # [N, L, 1]
    sq = x2 + x2.transpose(1, 2) - 2.0 * torch.bmm(x, x.transpose(1, 2))
    sq = sq.clamp(min=0.0)                           # numerical guard
    dist = torch.sqrt(sq + 1e-8) - (1e-8 ** 0.5)   # subtract residual so d(i,i)≈0
    return dist.clamp(max=40.0)


class _MaskedMPLayer(nn.Module):
    """Distance-biased masked attention + MLP over residues (invariant trunk)."""

    def __init__(self, dim: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        N, L, D = h.shape
        x = self.norm1(h)
        q = self.q(x).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + self.edge_bias(edge_feat).permute(0, 3, 1, 2)
        key_mask = valid[:, None, None, :]
        scores = scores.masked_fill(~key_mask, -1.0e4)
        attn = torch.softmax(scores, dim=-1) * key_mask.float()
        upd = torch.matmul(attn, v).transpose(1, 2).reshape(N, L, D)
        h = h + self.dropout(self.out(upd))
        h = h + self.dropout(self.mlp(self.norm2(h)))
        return h * valid.unsqueeze(-1).float()


class MessagePassingCADecoder(nn.Module):
    """Per-residue trunk + masked residue message passing -> local CA update."""

    def __init__(self, cfg: MessagePassingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.state_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.hidden), nn.GELU(),
            *[m for _ in range(max(cfg.depth - 1, 0)) for m in (nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())],
        )
        rbf_centers = torch.linspace(0.0, 40.0, cfg.num_rbf)
        self.register_buffer("rbf_centers", rbf_centers, persistent=False)
        self.rbf_width = 40.0 / max(cfg.num_rbf - 1, 1)
        seq_centers = torch.logspace(0.0, 3.0, cfg.seqsep_rbf)  # 1..1000 residues
        self.register_buffer("seq_centers", seq_centers, persistent=False)
        self.seq_width = (3.0 - 0.0) / max(cfg.seqsep_rbf - 1, 1) * (1.0 / 2.302585)

        edge_dim = cfg.num_rbf + cfg.seqsep_rbf + (1 if cfg.use_dh_cosine else 0)
        self.layers = nn.ModuleList(
            [_MaskedMPLayer(cfg.hidden, cfg.mp_heads, edge_dim, cfg.mp_dropout)
             for _ in range(cfg.mp_depth)]
        )
        self.norm_out = nn.LayerNorm(cfg.hidden)
        out_dim = 4 if cfg.predict_log_scale else 3
        self.head = nn.Linear(cfg.hidden, out_dim)
        # Zero-init the point head: dx ~ 0 at step 0 => training starts exactly at
        # persistence (same rationale as SimpleCADecoder).
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def _edge_features(self, X_context: torch.Tensor, dH: torch.Tensor,
                       residue_mask: torch.Tensor) -> torch.Tensor:
        N, L = X_context.shape[:2]
        dist = _pairwise_distance(X_context)                             # [N,L,L] invariant
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2)
        idx = torch.arange(L, device=X_context.device, dtype=torch.float32)
        sep = (idx[:, None] - idx[None, :]).abs()                         # [L,L]
        sep_rbf = torch.exp(-((sep.unsqueeze(-1).log1p() - self.seq_centers.log()) / self.seq_width) ** 2)
        sep_rbf = sep_rbf.unsqueeze(0).expand(N, L, L, -1)
        feats = [rbf, sep_rbf]
        if self.cfg.use_dh_cosine:
            n = dH / dH.norm(dim=-1, keepdim=True).clamp_min(1e-6)        # [N,L,d]
            cos = torch.matmul(n, n.transpose(-1, -2)).unsqueeze(-1)      # [N,L,L,1]
            feats.append(cos)
        edge = torch.cat(feats, dim=-1)
        pair_valid = residue_mask[:, :, None] & residue_mask[:, None, :]
        return edge * pair_valid.unsqueeze(-1).float()

    def forward(
        self,
        H_future: torch.Tensor,        # [N, L, dH]
        H_context: torch.Tensor,       # [N, L, dH]
        X_context: torch.Tensor,       # [N, L, 3] anchor CA (global gauge)
        residue_mask: torch.Tensor,    # [N, L]
        R_anchor: torch.Tensor,        # [N, L, 3, 3] observed anchor frame
        S_future: torch.Tensor | None = None,  # [N, L, dS] optional bath
    ) -> dict[str, torch.Tensor]:
        if R_anchor is None:
            raise ValueError("MessagePassingCADecoder requires the observed R_anchor")
        expected = (*X_context.shape[:-1], 3, 3)
        if tuple(R_anchor.shape) != tuple(expected):
            raise ValueError(f"R_anchor must have shape {expected}, got {tuple(R_anchor.shape)}")

        feats = [H_future, H_context, H_future - H_context]
        if self.cfg.bath_dim > 0:
            if S_future is None:
                raise ValueError("bath_dim > 0 but S_future was not provided")
            feats.append(S_future)
        state = torch.cat(feats, dim=-1)
        dH = H_future - H_context
        valid = residue_mask.bool()

        outs = []
        chunk = max(int(self.cfg.mp_chunk), 1)
        for s in range(0, state.shape[0], chunk):
            sl = slice(s, min(s + chunk, state.shape[0]))
            h = self.trunk(state[sl])
            edge = self._edge_features(X_context[sl], dH[sl], valid[sl])
            for layer in self.layers:
                h = layer(h, edge, valid[sl])
            outs.append(self.head(self.norm_out(h)))
        raw = torch.cat(outs, dim=0)

        mask = residue_mask.unsqueeze(-1).to(raw.dtype)
        m = self.cfg.max_update
        if self.cfg.predict_log_scale:
            dx_local, log_s = raw[..., :3], raw[..., 3:]
            dx_local = m * torch.tanh(dx_local / max(m, 1e-6))
        else:
            dx_local, log_s = m * torch.tanh(raw / max(m, 1e-6)), None

        dx_local = dx_local * mask
        dx = (R_anchor @ dx_local.unsqueeze(-1)).squeeze(-1) * mask
        ca_pred = (X_context + dx) * mask
        out = {"dx_local": dx_local, "dx": dx, "ca_pred": ca_pred}
        if log_s is not None:
            out["log_scale"] = log_s * mask
        return out


# =============================================================================
# CRIDecoder: cross-residue + H<->S interactive decoder
# =============================================================================


class _MultiHeadAttention(nn.Module):
    """Plain masked multi-head attention over the residue axis.

    ``bias`` is an optional additive attention bias ``[N, heads, L, L]`` (e.g. a
    geometry/edge-feature bias); when it is None this is standard attention.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_in: torch.Tensor,                 # [N, L, D]
        kv_in: torch.Tensor,                # [N, L, D]
        valid: torch.Tensor,                # [N, L] bool
        bias: torch.Tensor | None = None,   # [N, heads, L, L] additive
    ) -> torch.Tensor:
        N, L, D = q_in.shape
        q = self.q(q_in).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(kv_in).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(kv_in).view(N, L, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if bias is not None:
            scores = scores + bias
        key_mask = valid[:, None, None, :]
        scores = scores.masked_fill(~key_mask, -1.0e4)
        attn = torch.softmax(scores, dim=-1) * key_mask.float()
        attn = self.dropout(attn)
        upd = torch.matmul(attn, v).transpose(1, 2).reshape(N, L, D)
        return self.out(upd)


class _CRILayer(nn.Module):
    """One layer of: H<->S cross-attention, residue self-attention, FFN.

    Order matters.  H<->S first lets the bath state modulate the latent state
    per residue *before* residues exchange information, so the residue-level
    message passing operates on bath-aware features.  Both sublayers are
    pre-norm residual, and every output is re-masked.
    """

    def __init__(self, dim: int, heads: int, edge_dim: int, hs_heads: int,
                 dropout: float, use_hs_cross: bool):
        super().__init__()
        self.use_hs_cross = use_hs_cross
        self.norm_hs_h = nn.LayerNorm(dim)
        self.norm_hs_s = nn.LayerNorm(dim)
        self.norm_hs_c = nn.LayerNorm(dim)
        if use_hs_cross:
            # H attends to S and S attends to H (both directions of the coupling).
            self.h_from_s = _MultiHeadAttention(dim, hs_heads, dropout)
            self.s_from_h = _MultiHeadAttention(dim, hs_heads, dropout)
            # Gated fusion: the decoder chooses how much bath signal to inject.
            self.hs_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = _MultiHeadAttention(dim, heads, dropout)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,            # [N, L, D] latent state (H + S fused)
        s: torch.Tensor | None,     # [N, L, D] bath branch, projected; None if no S
        edge_feat: torch.Tensor,    # [N, L, L, E]
        valid: torch.Tensor,        # [N, L] bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        m = valid.unsqueeze(-1).float()

        # --- H <-> S cross attention -------------------------------------
        if self.use_hs_cross and s is not None:
            dn = self.h_from_s(self.norm_hs_h(h), self.norm_hs_s(s), valid)
            sn = self.s_from_h(self.norm_hs_s(s), self.norm_hs_h(h), valid)
            gate = self.hs_gate(self.norm_hs_c(h))
            h = h + self.dropout(dn) * gate
            s = s + self.dropout(sn)
            h, s = h * m, s * m

        # --- residue self attention with geometric bias -------------------
        x = self.norm1(h)
        bias = self.edge_bias(edge_feat).permute(0, 3, 1, 2)   # [N, heads, L, L]
        h = h + self.dropout(self.self_attn(x, x, valid, bias=bias))
        h = h * m

        # --- FFN ----------------------------------------------------------
        h = h + self.dropout(self.mlp(self.norm2(h)))
        h = h * m
        return h, s


class CRIDecoder(nn.Module):
    """Cross-Residue Interactive Decoder.

    Same ``forward`` contract as ``SimpleCADecoder``.  The trunk is a stack of
    ``_CRILayer`` layers over an invariant residue graph; H and S are kept as
    two separate streams and mutually modulate each other at every layer.
    """

    def __init__(self, cfg: MessagePassingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.state_dim
        self.use_s = cfg.bath_dim > 0
        # Joint per-residue encoding of (H_future, H_anchor, dH, S_future).
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.hidden),
            nn.GELU(),
            *[m for _ in range(max(cfg.depth - 1, 0))
              for m in (nn.Linear(cfg.hidden, cfg.hidden), nn.GELU())],
        )
        # Separate S stream so the cross-attention has a bath-specific subspace
        # rather than sharing the fused trunk representation.
        if self.use_s and cfg.use_hs_cross:
            self.s_proj = nn.Sequential(
                nn.LayerNorm(cfg.bath_dim),
                nn.Linear(cfg.bath_dim, cfg.hidden),
                nn.GELU(),
                nn.Linear(cfg.hidden, cfg.hidden),
            )
        else:
            self.s_proj = None

        rbf_centers = torch.linspace(0.0, 40.0, cfg.num_rbf)
        self.register_buffer("rbf_centers", rbf_centers, persistent=False)
        self.rbf_width = 40.0 / max(cfg.num_rbf - 1, 1)
        seq_centers = torch.logspace(0.0, 3.0, cfg.seqsep_rbf)
        self.register_buffer("seq_centers", seq_centers, persistent=False)
        self.seq_width = (3.0 - 0.0) / max(cfg.seqsep_rbf - 1, 1) * (1.0 / 2.302585)

        edge_dim = cfg.num_rbf + cfg.seqsep_rbf
        if cfg.use_dh_cosine:
            edge_dim += 1
        if self.use_s and cfg.use_bath_coupling:
            edge_dim += 1
        self.edge_dim = edge_dim

        self.layers = nn.ModuleList(
            [_CRILayer(cfg.hidden, cfg.mp_heads, edge_dim, cfg.hs_heads,
                       cfg.mp_dropout, cfg.use_hs_cross)
             for _ in range(cfg.mp_depth)]
        )
        self.norm_out = nn.LayerNorm(cfg.hidden)
        out_dim = 4 if cfg.predict_log_scale else 3
        self.head = nn.Linear(cfg.hidden, out_dim)
        # Zero-ish init: dx ~ 0 at step 0 => training starts exactly at persistence.
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def _edge_features(self, X_context: torch.Tensor, dH: torch.Tensor,
                       S_future: torch.Tensor | None,
                       residue_mask: torch.Tensor) -> torch.Tensor:
        N, L = X_context.shape[:2]
        dist = _pairwise_distance(X_context)
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.rbf_centers) / self.rbf_width) ** 2)
        idx = torch.arange(L, device=X_context.device, dtype=torch.float32)
        sep = (idx[:, None] - idx[None, :]).abs()
        sep_rbf = torch.exp(-((sep.unsqueeze(-1).log1p() - self.seq_centers.log())
                              / self.seq_width) ** 2)
        sep_rbf = sep_rbf.unsqueeze(0).expand(N, L, L, -1)
        feats = [rbf, sep_rbf]
        if self.cfg.use_dh_cosine:
            n = dH / dH.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cos = torch.matmul(n, n.transpose(-1, -2)).unsqueeze(-1)
            feats.append(cos)
        if self.use_s and self.cfg.use_bath_coupling and S_future is not None:
            sn = S_future / S_future.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scos = torch.matmul(sn, sn.transpose(-1, -2)).unsqueeze(-1)
            feats.append(scos)
        edge = torch.cat(feats, dim=-1)
        pair_valid = residue_mask[:, :, None] & residue_mask[:, None, :]
        return edge * pair_valid.unsqueeze(-1).float()

    def forward(
        self,
        H_future: torch.Tensor,        # [N, L, dH]
        H_context: torch.Tensor,       # [N, L, dH]
        X_context: torch.Tensor,       # [N, L, 3] anchor CA (global gauge)
        residue_mask: torch.Tensor,    # [N, L]
        R_anchor: torch.Tensor,        # [N, L, 3, 3] observed anchor frame
        S_future: torch.Tensor | None = None,  # [N, L, dS] optional bath
    ) -> dict[str, torch.Tensor]:
        if R_anchor is None:
            raise ValueError("CRIDecoder requires the observed R_anchor")
        expected = (*X_context.shape[:-1], 3, 3)
        if tuple(R_anchor.shape) != tuple(expected):
            raise ValueError(f"R_anchor must have shape {expected}, got {tuple(R_anchor.shape)}")
        if self.use_s and S_future is None:
            raise ValueError("bath_dim > 0 but S_future was not provided")

        feats = [H_future, H_context, H_future - H_context]
        if self.use_s:
            feats.append(S_future)
        state = torch.cat(feats, dim=-1)
        dH = H_future - H_context
        valid = residue_mask.bool()

        outs = []
        chunk = max(int(self.cfg.mp_chunk), 1)
        for k in range(0, state.shape[0], chunk):
            sl = slice(k, min(k + chunk, state.shape[0]))
            v = valid[sl]
            h = self.trunk(state[sl])
            s = self.s_proj(S_future[sl]) if self.s_proj is not None else None
            edge = self._edge_features(X_context[sl], dH[sl], S_future[sl]
                                       if self.use_s else None, v)
            for layer in self.layers:
                h, s = layer(h, s, edge, v)
            outs.append(self.head(self.norm_out(h)))
        raw = torch.cat(outs, dim=0)

        mask = residue_mask.unsqueeze(-1).to(raw.dtype)
        m = self.cfg.max_update
        if self.cfg.predict_log_scale:
            dx_local, log_s = raw[..., :3], raw[..., 3:]
            dx_local = m * torch.tanh(dx_local / max(m, 1e-6))
        else:
            dx_local, log_s = m * torch.tanh(raw / max(m, 1e-6)), None

        dx_local = dx_local * mask
        dx = (R_anchor @ dx_local.unsqueeze(-1)).squeeze(-1) * mask
        ca_pred = (X_context + dx) * mask
        out = {"dx_local": dx_local, "dx": dx, "ca_pred": ca_pred}
        if log_s is not None:
            out["log_scale"] = log_s * mask
        return out
