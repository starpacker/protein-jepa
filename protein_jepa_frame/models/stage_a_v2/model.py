"""Stage A V2: clean JEPA-Anything-aligned Stage A for protein dynamics.

Architecture (per STAGE_ABC_REDESIGN_PLAN.md §1):
  - Online encoder f_θ  (ProteinGeometryEncoder, trained by gradient)
  - EMA teacher  f_θ̄  (same arch, exponential moving average, always eval+no_grad)
  - OPF projectors  P_j ∈ R^{r×dH}  (learnable, soft-Gram orthogonality)
  - Causal Transformer predictor  g_ω  (history K frames → J factor predictions)

Forward contracts:
  - forward_spatial(batch):  spatial mask on a single frame; returns per-residue
    factor predictions and teacher targets.
  - forward_temporal(batch): temporal mask; context K frames predict target frame
    at lag τ.
  Both return (pred_factors, target_factors, context_H, valid_mask).

EMA update rule (§1.3):
  θ̄ ← m·θ̄ + (1-m)·θ   applied AFTER each optimizer.step(), not inside forward.
  Non-float buffers are copied verbatim.

stop_grad placement (§1.4):
  target_factors = P_j · stop_grad(h̄)
  P_j itself is NOT detached — projector gets gradients from factor prediction loss.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path


def _find_jepa_anything_src() -> Path:
    """Locate jepa-anything-core/src by walking up from this file."""
    rel = Path("jepa_anything") / "JEPA-Anything-main" / "jepa-anything-core" / "src"
    for parent in Path(__file__).resolve().parents:
        candidate = parent / rel
        if candidate.is_dir():
            return candidate
    raise ImportError(
        f"could not locate {rel} in any parent of {Path(__file__).resolve()}"
    )


_ja_path = _find_jepa_anything_src()
if str(_ja_path) not in sys.path:
    sys.path.insert(0, str(_ja_path))
from jepa_anything_core.opf import OrthogonalFactorProjection  # noqa: E402

from protein_jepa_frame.models.encoder import ProteinGeometryEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageAV2Config:
    latent_dim: int = 128         # dH; must equal num_factors * factor_dim
    encoder_depth: int = 4
    num_heads: int = 4
    num_rbf: int = 16
    dropout: float = 0.1
    max_residues: int = 4096
    orientation_features: bool = True
    # Scale on the time-invariant identity embeddings (aa / residue-index /
    # chain).  Kept small so the geometry signal drives H rather than a static
    # identity code (see encoder.py note on time-invariant collapse).
    identity_embed_scale: float = 0.1

    # OPF
    num_factors: int = 4          # J; factor_dim r = latent_dim // num_factors
    opf_learnable: bool = True

    # Predictor
    predictor_depth: int = 4
    predictor_heads: int = 4
    max_frames: int = 256         # for positional encoding

    # EMA schedule: m stays at ema_start for ema_warmup_steps, then increases
    # linearly to ema_end over the *remaining* training.  Two properties matter
    # for avoiding the trivial-copy shortcut (predictor copies the last frame
    # instead of reasoning about dynamics):
    #   * a fast-moving teacher stays close to the online encoder, so
    #     predicting it requires no temporal reasoning;
    #   * a teacher that has not yet converged to a temporally-stable
    #     representation is itself a moving target.
    # The warm-up keeps the teacher slow from the very first steps (the older
    # schedule reached 0.9999 only after ~10k steps, so for most of a 25k run
    # the teacher tracked the online encoder closely).
    ema_start: float = 0.998
    ema_end: float = 0.9999
    ema_warmup_steps: int = 2500

    # Spatial mask
    spatial_mask_ratio: float = 0.40   # fraction of residues masked per frame

    # Structure-reconstruction auxiliary head.  The delta-target temporal
    # objective makes H time-varying but, on its own, lets the SE(3)-invariant
    # encoder discard the absolute internal fold (measured: a probe on H cannot
    # recover CA-CA distances even overfit on one protein, ~9 A RMSE).  The
    # downstream oracle decoder then has no structural signal to read.  This
    # head forces H to retain the full internal geometry by regressing pairwise
    # CA-CA distances (and, when enabled, local backbone geometry) directly from
    # H residue pairs.  It is trained jointly with the encoder and discarded
    # after Stage A.
    struct_recon_dim: int = 256        # hidden width of the distance head
    struct_recon_pairs: int = 256      # CA-CA pairs sampled per frame per step

    # Delta-target toggle.  When True the temporal objective predicts the
    # teacher-space displacement  P_j·(h̄_target − h̄_anchor)  (the original fix
    # for time-invariant collapse).  That objective is, however, *antagonistic*
    # to struct_recon: it rewards the encoder for discarding the static fold,
    # which is exactly the information the decoder needs.  With struct_recon
    # enabled H is already time-varying (structure changes between frames), so
    # the absolute target  P_j·h̄_target  suffices and keeps H structural.  Set
    # False to use the absolute teacher target.
    delta_target: bool = True

    @property
    def factor_dim(self) -> int:
        return self.latent_dim // self.num_factors

    def __post_init__(self) -> None:
        if self.latent_dim % self.num_factors != 0:
            raise ValueError(
                f"latent_dim ({self.latent_dim}) must be divisible by "
                f"num_factors ({self.num_factors})"
            )


def _context_to_target_lag_ps(batch: dict, K: int) -> torch.Tensor:
    """Physical interval from the last context frame to the target frame.

    The paired dataset reports `horizon_time_ps` as
    (first future frame) - (first context frame).  After the context is
    truncated to its last K frames the predictor's summary is anchored at the
    *last* context frame, so the lag it must bridge is
    (first future frame) - (last context frame).  Falls back to the reported
    horizon when time stamps are unavailable (e.g. synthetic batches).
    """
    fallback = batch["horizon_time_ps"]
    ctx_t = batch.get("context_time_ps")
    fut_t = batch.get("future_time_ps")
    if ctx_t is None or fut_t is None:
        return fallback
    if ctx_t.shape[1] > K:
        ctx_t = ctx_t[:, -K:]
    lag = fut_t[:, 0] - ctx_t[:, -1]
    # Guard against malformed/synthetic time stamps: a non-positive gap means
    # the stamps carry no usable information, so keep the reported horizon.
    bad = ~torch.isfinite(lag) | (lag <= 0)
    return torch.where(bad, fallback.to(lag.dtype), lag)


# ─────────────────────────────────────────────────────────────────────────────
# Causal Transformer predictor
# ─────────────────────────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    """Learned absolute frame-index positional encoding."""

    def __init__(self, dim: int, max_frames: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_frames, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, L, D]
        K = x.shape[1]
        pos = torch.arange(K, device=x.device)
        return x + self.pe(pos)[None, :, None, :]   # broadcast over B, L


class _TimeEmbedding(nn.Module):
    """Log-time sinusoidal projection for conditioning on τ and T."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, lag_ps: torch.Tensor, temperature_k: torch.Tensor) -> torch.Tensor:
        # lag_ps: [B], temperature_k: [B]
        log_lag = torch.log1p(lag_ps.clamp(min=0.0)).unsqueeze(-1)
        log_temp = torch.log(temperature_k.clamp(min=1.0)).unsqueeze(-1)
        feat = torch.cat([log_lag, log_temp], dim=-1)   # [B, 2]
        return self.proj(feat)                           # [B, dim]


class CausalTransformerPredictor(nn.Module):
    """Causal Transformer that takes H_{≤t} history and outputs J factor predictions.

    Input:  H_ctx  [B, K, L, dH]  — context frames (K frames, L residues)
            lag_ps     [B]         — physical lag τ in ps
            temperature_k  [B]     — temperature in Kelvin
    Output: pred_factors  [B, L, J, r]  — predicted factor coordinates
    """

    def __init__(
        self,
        latent_dim: int,
        num_factors: int,
        factor_dim: int,
        depth: int = 4,
        num_heads: int = 4,
        max_frames: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim

        self.pos_enc = _PositionalEncoding(latent_dim, max_frames)
        self.time_embed = _TimeEmbedding(latent_dim)

        # Per-frame causal self-attention over frames (K-dim axis)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,   # [B*L, K, dH]
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

        # Output head: per-residue, J×r prediction
        self.out_head = nn.Linear(latent_dim, num_factors * factor_dim)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def _causal_mask(self, K: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular float mask for causal attention over K frames."""
        mask = torch.full((K, K), float("-inf"), device=device)
        mask.triu_(diagonal=1)
        return mask

    def forward(
        self,
        H_ctx: torch.Tensor,
        lag_ps: torch.Tensor,
        temperature_k: torch.Tensor,
        target_frame: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, K, L, D = H_ctx.shape

        # Optional masked-frame input: rebuild the causal sequence with the
        # target frame standing in for the last context frame.  This prevents
        # the predictor from satisfying the short-lag objective by simply
        # copying the un-masked last context frame, which encodes the target's
        # *own* geometry and carries no predictive content.
        if target_frame is not None:
            x_hist = torch.cat([H_ctx[:, :-1], target_frame], dim=1)   # [B,K,L,D]
        else:
            x_hist = H_ctx

        # Time conditioning added to the last frame token
        time_cond = self.time_embed(lag_ps, temperature_k)   # [B, D]

        # Add positional encoding
        x = self.pos_enc(x_hist)                              # [B, K, L, D]
        x[:, -1] = x[:, -1] + time_cond[:, None, :]          # inject into last frame

        # Reshape: treat each residue position independently over K frames
        # [B, K, L, D] → [B*L, K, D]
        x = x.permute(0, 2, 1, 3).reshape(B * L, K, D)

        causal = self._causal_mask(K, x.device)
        x = self.transformer(x, mask=causal, is_causal=True)  # [B*L, K, D]

        # Use last-frame output as the context summary
        summary = x[:, -1, :]                                  # [B*L, D]
        summary = summary.reshape(B, L, D)

        # Project to J×r factor predictions
        out = self.out_head(summary)                           # [B, L, J*r]
        return out.reshape(B, L, self.num_factors, self.factor_dim)  # [B, L, J, r]


# ─────────────────────────────────────────────────────────────────────────────
# Main Stage A V2 model
# ─────────────────────────────────────────────────────────────────────────────

class StageAV2(nn.Module):
    """JEPA-Anything-aligned Stage A model for protein dynamics.

    Freeze contract:
      - context_encoder: online, trained by gradient
      - teacher_encoder: EMA shadow of context_encoder, always eval+no_grad
      - opf: learnable OPF projectors (soft-Gram)
      - predictor: causal Transformer, trained by gradient

    The teacher_encoder must NEVER be called inside a gradient-enabled context.
    Call update_ema() after every optimizer.step().
    """

    def __init__(self, config: StageAV2Config) -> None:
        super().__init__()
        self.config = config

        # Online encoder
        self.context_encoder = ProteinGeometryEncoder(
            latent_dim=config.latent_dim,
            depth=config.encoder_depth,
            num_heads=config.num_heads,
            num_rbf=config.num_rbf,
            dropout=config.dropout,
            max_residues=config.max_residues,
            orientation_features=config.orientation_features,
            identity_embed_scale=config.identity_embed_scale,
        )

        # EMA teacher (deep copy, no grad)
        self.teacher_encoder = copy.deepcopy(self.context_encoder)
        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)

        # OPF projectors: shape (J, r, dH), i.e. (num_factors, factor_dim, latent_dim)
        self.opf = OrthogonalFactorProjection(
            state_dim=config.latent_dim,
            num_factors=config.num_factors,
            factor_dim=config.factor_dim,
            learnable=config.opf_learnable,
            orthogonality_mode="soft_gram",
        )

        # Causal predictor
        self.predictor = CausalTransformerPredictor(
            latent_dim=config.latent_dim,
            num_factors=config.num_factors,
            factor_dim=config.factor_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            max_frames=config.max_frames,
            dropout=config.dropout,
        )

        # EMA step counter (used externally to compute momentum schedule)
        self.register_buffer("_ema_step", torch.zeros((), dtype=torch.long))

        # Structure-reconstruction head: regress CA-CA distance from a pair of
        # H vectors.  Symmetric in (i, j) by construction (concat both orders
        # and average).  Trained jointly with the encoder; keeps H structurally
        # faithful so the downstream decoder has a signal to read.
        dH = config.latent_dim
        self.struct_recon_head = nn.Sequential(
            nn.Linear(2 * dH, config.struct_recon_dim),
            nn.GELU(),
            nn.Linear(config.struct_recon_dim, config.struct_recon_dim),
            nn.GELU(),
            nn.Linear(config.struct_recon_dim, 1),
        )

    # ── train() override: keep teacher always in eval ────────────────────────

    def train(self, mode: bool = True) -> "StageAV2":
        super().train(mode)
        self.teacher_encoder.eval()
        return self

    # ── EMA update ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_ema(self, momentum: float | None = None) -> float:
        """Apply one EMA step.  Returns the momentum value used.

        Call this after every optimizer.step(), not inside forward().
        """
        self._ema_step += 1
        if momentum is None:
            t = self._ema_step.item()
            cfg = self.config
            warm = max(1, int(cfg.ema_warmup_steps))
            if t <= warm:
                momentum = cfg.ema_start
            else:
                progress = min((t - warm) / (2.0 * warm), 1.0)
                momentum = cfg.ema_start + progress * (cfg.ema_end - cfg.ema_start)

        for online, shadow in zip(
            self.context_encoder.parameters(),
            self.teacher_encoder.parameters(),
        ):
            shadow.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

        for online_buf, shadow_buf in zip(
            self.context_encoder.buffers(),
            self.teacher_encoder.buffers(),
        ):
            if online_buf.dtype.is_floating_point:
                shadow_buf.data.mul_(momentum).add_(online_buf.data, alpha=1.0 - momentum)
            else:
                shadow_buf.data.copy_(online_buf.data)

        return float(momentum)

    @torch.no_grad()
    def sync_teacher(self) -> None:
        """Hard-copy online weights to teacher (use at initialization)."""
        self.teacher_encoder.load_state_dict(self.context_encoder.state_dict())

    # ── Structure reconstruction auxiliary ───────────────────────────────────

    def struct_recon_loss(
        self,
        context_H: torch.Tensor,     # [B, K, L, dH]  online H for context frames
        coords: torch.Tensor,         # [B, K, L, 4, 3]  backbone coords (N,CA,C,O)
        residue_mask: torch.Tensor,   # [B, K, L]
        n_pairs: int | None = None,
    ) -> torch.Tensor:
        """Regress CA-CA distances from H residue pairs; MSE loss in Å².

        Samples n_pairs random (i,j) pairs per (batch, frame) and regresses
        the Euclidean CA-CA distance.  Trains the online encoder and
        struct_recon_head jointly so H retains internal geometry.

        Returns a scalar MSE loss (Å²).
        """
        if n_pairs is None:
            n_pairs = self.config.struct_recon_pairs
        B, K, L, dH = context_H.shape
        # Flatten batch × frame dimension for efficiency
        H_flat = context_H.reshape(B * K, L, dH)           # [BK, L, dH]
        ca_flat = coords[:, :, :, 1, :].reshape(B * K, L, 3)  # [BK, L, 3]
        mask_flat = residue_mask.reshape(B * K, L).bool()   # [BK, L]
        BK = B * K
        dev = H_flat.device

        # Sample random residue index pairs
        idx_i = torch.randint(L, (BK, n_pairs), device=dev)   # [BK, n_pairs]
        idx_j = torch.randint(L, (BK, n_pairs), device=dev)

        ar = torch.arange(BK, device=dev)[:, None]  # [BK, 1]

        # H at each end of the pair — symmetric concat
        Hi = H_flat[ar, idx_i]   # [BK, n_pairs, dH]
        Hj = H_flat[ar, idx_j]
        feat_ij = torch.cat([Hi, Hj], dim=-1)  # [BK, n_pairs, 2*dH]
        feat_ji = torch.cat([Hj, Hi], dim=-1)

        # Average over both orderings for symmetry
        pred = 0.5 * (
            self.struct_recon_head(feat_ij) + self.struct_recon_head(feat_ji)
        ).squeeze(-1)  # [BK, n_pairs]

        # Ground-truth CA-CA distance
        cai = ca_flat[ar, idx_i]   # [BK, n_pairs, 3]
        caj = ca_flat[ar, idx_j]
        dist = (cai - caj).norm(dim=-1)  # [BK, n_pairs]

        # Mask: both residues must be valid
        valid = mask_flat[ar, idx_i] & mask_flat[ar, idx_j]  # [BK, n_pairs]
        valid_f = valid.float()

        n_valid = valid_f.sum().clamp(min=1.0)
        return ((pred - dist) ** 2 * valid_f).sum() / n_valid

    # ── Teacher forward (always no_grad, eval mode) ──────────────────────────

    def _teacher_encode(self, batch_frame: dict) -> torch.Tensor:
        """Encode a single frame with the teacher; returns [B, L, dH], no grad."""
        assert not self.teacher_encoder.training, "teacher_encoder must stay in eval"
        with torch.no_grad():
            return self.teacher_encoder(
                coords=batch_frame["coords"],
                residue_mask=batch_frame["residue_mask"],
                aatype=batch_frame["aatype"],
                residue_index=batch_frame["residue_index"],
                chain_id=batch_frame["chain_id"],
            )

    # ── OPF factor helpers ───────────────────────────────────────────────────

    def _project_to_factors(self, h: torch.Tensor) -> torch.Tensor:
        """h: [..., dH]  →  factors: [..., J, r]  via OPF decompose."""
        # opf.decompose: (..., d) → (..., K, r)
        return self.opf.decompose(h)   # [..., J, r]

    # ── Spatial mask forward ─────────────────────────────────────────────────

    def forward_spatial(
        self,
        batch: dict,
        spatial_mask: torch.Tensor | None = None,
    ) -> dict:
        """Spatial (within-frame) mask: predict masked residues' teacher factors.

        Args:
            batch: collate_clips output.  coords [B,T,L,4,3].
                   Uses the *last* frame as both context and target.
            spatial_mask: bool tensor [B, L]; True = masked (hidden from online encoder).
                          If None, generated randomly at config.spatial_mask_ratio.

        Returns dict with keys:
            pred_factors:   [B, L_masked, J, r]
            target_factors: [B, L_masked, J, r]  (stop_grad on h̄ but NOT on P_j)
            context_H:      [B, L, dH]            (full online representation)
            valid_mask:     [B, L]                 (residue_mask of the frame)
            spatial_mask:   [B, L]                 (mask used)
        """
        coords = batch["coords"]           # [B, T, L, 4, 3]
        residue_mask = batch["residue_mask"]  # [B, T, L]  (all frames same residues)
        aatype = batch["aatype"]           # [B, L]
        residue_index = batch["residue_index"]
        chain_id = batch["chain_id"]
        B, T, L = coords.shape[:3]

        # Use last frame for both context (masked) and teacher target (full)
        last_coords = coords[:, -1:]       # [B, 1, L, 4, 3] — keep time dim
        last_rmask = residue_mask[:, -1:]  # [B, 1, L]
        valid = residue_mask[:, -1]        # [B, L]  per-residue validity

        # Build spatial mask if not provided
        if spatial_mask is None:
            spatial_mask = self._sample_spatial_mask(valid, self.config.spatial_mask_ratio)

        # Online encoder: sees only unmasked residues.
        # residue_mask is [B,T,L]; token_mask must match that leading shape.
        online_token_mask = (valid & ~spatial_mask).unsqueeze(1)  # [B, 1, L]
        context_H = self.context_encoder(
            coords=last_coords,
            residue_mask=last_rmask,
            aatype=aatype,
            residue_index=residue_index,
            chain_id=chain_id,
            token_mask=online_token_mask,
        )[:, 0]  # [B, L, dH]

        # Teacher encoder: full frame, no grad
        target_frame = {
            "coords": last_coords,
            "residue_mask": last_rmask,
            "aatype": aatype,
            "residue_index": residue_index,
            "chain_id": chain_id,
        }
        h_teacher = self._teacher_encode_clip(target_frame)   # [B, L, dH]

        # Predictor input: use context as a 1-frame "history"
        H_ctx = context_H.unsqueeze(1)  # [B, 1, L, dH]
        lag_ps = torch.zeros(B, device=coords.device)
        temp_k = batch.get("temperature_k", torch.full((B,), 348.0, device=coords.device))

        pred_full = self.predictor(H_ctx, lag_ps, temp_k)   # [B, L, J, r]

        # Compute factors only on masked (target) residues
        mask_idx = spatial_mask & valid   # [B, L]

        # target: P_j · stop_grad(h̄)   — stop_grad on h̄, NOT on P_j
        h_teacher_sg = h_teacher.detach()
        target_full = self._project_to_factors(h_teacher_sg)   # [B, L, J, r]

        # Gather masked positions
        pred_factors = pred_full[mask_idx]      # [N_masked, J, r]
        target_factors = target_full[mask_idx]  # [N_masked, J, r]

        return {
            "pred_factors": pred_factors,
            "target_factors": target_factors,
            "context_H": context_H,
            "valid_mask": valid,
            "spatial_mask": spatial_mask,
            "h_teacher_full": h_teacher_sg,  # for variance tracking
        }

    # ── Temporal (causal) mask forward ───────────────────────────────────────

    def forward_temporal(
        self,
        batch: dict,
    ) -> dict:
        """Temporal mask: K context frames → predict target frame H at lag τ.

        Expects a *paired* batch from collate_paired_windows, with keys:
          context_coords [B,K,L,4,3], future_coords [B,Tf,L,4,3],
          horizon_time_ps [B], temperature_k [B].

        Returns dict with keys:
            pred_factors:   [B, L, J, r]
            target_factors: [B, L, J, r]
            context_H:      [B, K, L, dH]
            valid_mask:     [B, L]
        """
        ctx_coords = batch["context_coords"]       # [B, K, L, 4, 3]
        ctx_rmask = batch["context_residue_mask"]  # [B, K, L]
        aatype = batch["context_aatype"]
        residue_index = batch["context_residue_index"]
        chain_id = batch["context_chain_id"]

        fut_coords = batch["future_coords"]
        fut_rmask = batch["future_residue_mask"]

        B, K, L = ctx_coords.shape[:3]
        valid = ctx_rmask[:, 0]   # [B, L] — residue existence fixed across frames

        # τ must be measured from the LAST context frame to the target frame.
        # The predictor's causal summary is formed at the last context frame, so
        # the interval it has to bridge is (t_target - t_last_context).  The raw
        # `horizon_time_ps` is (first future frame - first context frame), which
        # equals this only when the context is a single frame.  With K > 1 it
        # overstates τ by the context span (e.g. 8 ns instead of 1 ns for an
        # 8-frame context), which makes the time conditioning unlearnable.
        lag_ps = _context_to_target_lag_ps(batch, K)   # [B]
        temp_k = batch.get("temperature_k", torch.full((lag_ps.shape[0],), 348.0, device=lag_ps.device))

        # Online encoder: all K context frames
        context_H = self.context_encoder(
            coords=ctx_coords,
            residue_mask=ctx_rmask,
            aatype=aatype,
            residue_index=residue_index,
            chain_id=chain_id,
        )   # [B, K, L, dH]

        # Teacher encodes first frame of future window (the target frame)
        target_frame = {
            "coords": fut_coords[:, :1],
            "residue_mask": fut_rmask[:, :1],
            "aatype": batch["future_aatype"],
            "residue_index": batch["future_residue_index"],
            "chain_id": batch["future_chain_id"],
        }
        h_teacher = self._teacher_encode_clip(target_frame)   # [B, L, dH]

        # Predictor
        pred = self.predictor(context_H, lag_ps, temp_k)   # [B, L, J, r]

        # target factors: P_j · stop_grad(h̄)
        target_factors = self._project_to_factors(h_teacher.detach())  # [B, L, J, r]

        return {
            "pred_factors": pred,
            "target_factors": target_factors,
            "context_H": context_H,
            "valid_mask": valid,
            "h_teacher_full": h_teacher.detach(),
        }

    # ── Multi-lag temporal forward ────────────────────────────────────────────

    def forward_temporal_multilag(
        self,
        batch: dict,
        lags: tuple[int, ...],
        recent_frac: float = 0.5,
    ) -> dict:
        """Multi-lag causal prediction from ONE context window.

        For every lag L in `lags`, the target frame is context frame index
        ``K-1-L`` while the visible context is ``context[:, :K-L]``.  The
        predictor's last token is the *masked* online encoding of the target
        frame, so it cannot resolve the target by copying its own geometry at
        the last position; it must integrate the preceding history.

        Supervising several lags (in particular the short, near-consecutive
        gap where MD actually carries signal) forces H to be temporally
        informative instead of a static fold fingerprint.

        Args:
            batch: paired-window batch (only ``context_*`` keys are used).
            lags: positive frame gaps, each < K.
            recent_frac: fraction of each batch (by sample index) restricted to
                the two *shortest* lags.  Long horizons are intrinsically
                unpredictable; without this the predictor can minimise the loss
                by ignoring history and copying the recent frame.

        Returns dict with:
            pred_factors:   [N, L_max, J, r]  (padded across lags)
            target_factors: [N, L_max, J, r]
            valid_mask:     [N, L_max]
            lag_ps:         [N]
            lag_slot:       [N] long, index into `lags`
        """
        ctx_coords = batch["context_coords"]       # [B, K, L, 4, 3]
        ctx_rmask = batch["context_residue_mask"]  # [B, K, L]
        aatype = batch["context_aatype"]
        residue_index = batch["context_residue_index"]
        chain_id = batch["context_chain_id"]
        ctx_t = batch.get("context_time_ps")
        temp_all = batch.get("temperature_k")

        B, K, L = ctx_coords.shape[:3]
        valid = ctx_rmask[:, 0]                     # residues fixed across frames
        lags = tuple(sorted({int(l) for l in lags}))
        for l in lags:
            if l < 1 or l >= K:
                raise ValueError(f"lag {l} out of range for context K={K}")

        # Encode all context frames with the online encoder.  The anchor and
        # target frames are additionally re-encoded with the teacher below to
        # form the delta-target.
        context_H = self.context_encoder(
            coords=ctx_coords,
            residue_mask=ctx_rmask,
            aatype=aatype,
            residue_index=residue_index,
            chain_id=chain_id,
        )   # [B, K, L, dH]

        # Ratio-based sampling of short vs long lags.
        n_short = max(1, min(len(lags), 2))
        short_slots = torch.arange(n_short, dtype=torch.long)          # sorted asc
        long_slots = torch.arange(n_short, len(lags), dtype=torch.long)
        n_recent = int(round(B * recent_frac))
        slot = torch.empty(B, dtype=torch.long)
        if n_recent > 0:
            slot[:n_recent] = short_slots[
                torch.randint(0, len(short_slots), (n_recent,))
            ]
        if B - n_recent > 0 and long_slots.numel() > 0:
            slot[n_recent:] = long_slots[
                torch.randint(0, long_slots.numel(), (B - n_recent,))
            ]
        elif B - n_recent > 0:
            slot[n_recent:] = short_slots[
                torch.randint(0, len(short_slots), (B - n_recent,))
            ]
        slot = slot.to(ctx_coords.device)

        # Select the target frame index for each sample.
        lag_sel = torch.tensor([lags[int(s)] for s in slot.tolist()],
                               dtype=torch.long, device=ctx_coords.device)
        # Target frame sits `lag_sel` frames *after* the last visible frame.
        anchor_idx = (K - 1) - lag_sel                # last visible frame
        tgt_idx = anchor_idx + 1                      # target inside [1, K-1]
        ar = torch.arange(B, device=ctx_coords.device)
        tgt_idx = tgt_idx.clamp(max=K - 1)
        tgt_coords = ctx_coords[ar, tgt_idx][:, None]       # [B,1,L,4,3]
        tgt_rmask = ctx_rmask[ar, tgt_idx][:, None]         # [B,1,L]

        # Visible history: frames [0 .. anchor_idx], right-aligned so the last
        # slot of the predictor input is the anchor frame.  Slots before the
        # anchor that fall outside the window are zero-filled.
        idx = torch.arange(K, device=ctx_coords.device)[None, :]     # [1,K]
        # Output slot j takes source frame (j - (K-1-anchor_idx)); require src>=0.
        src = idx - (K - 1 - anchor_idx[:, None])                    # [B,K]
        valid_src = (src >= 0) & (src <= anchor_idx[:, None])
        H_hist = torch.gather(
            context_H, 1,
            src.clamp(min=0, max=K - 1)[..., None, None].expand(
                -1, -1, L, context_H.shape[-1]),
        )
        H_target_in = torch.where(
            valid_src[..., None, None], H_hist, torch.zeros_like(H_hist)
        )

        # Physical lag in ps: interval from the anchor to the target frame.
        if ctx_t is not None:
            lag_ps = (ctx_t[ar, tgt_idx] - ctx_t[ar, anchor_idx]).to(torch.float32)
            bad = ~torch.isfinite(lag_ps) | (lag_ps <= 0)
            lag_ps = torch.where(bad, lag_sel.float() * 1000.0, lag_ps)
        else:
            lag_ps = lag_sel.float()
        if temp_all is not None:
            temp_k = temp_all
        else:
            temp_k = torch.full((B,), 348.0, device=ctx_coords.device)

        pred = self.predictor(H_target_in, lag_ps, temp_k)  # [B, L, J, r]

        # ── Delta-target (the fix for time-invariant collapse) ───────────────
        # H is ~98% static fold-identity code and only ~2% time-varying, so an
        # absolute target P_j·h̄(X_target) lets the predictor reach near-zero
        # loss by copying the anchor (cos(H_t,H_{t+1})≈0.98) — zero temporal
        # reasoning.  Instead we predict the *displacement* in teacher space:
        #
        #     target = P_j · ( h̄(X_target) - stop_grad(h̄(X_anchor)) )
        #
        # The static fold code cancels in the difference, so 100% of the target
        # variance — and therefore 100% of the prediction gradient — comes from
        # the conformational change between anchor and target.  Copying the
        # anchor now yields target ≈ 0 and large error; the predictor is forced
        # to model where the protein moves.  Both teacher encodings are detached
        # (stop_grad); only the online encoder + predictor + projectors learn.
        anchor_coords = ctx_coords[ar, anchor_idx][:, None]     # [B,1,L,4,3]
        anchor_rmask = ctx_rmask[ar, anchor_idx][:, None]       # [B,1,L]
        tgt_frame_dict = {
            "coords": tgt_coords,
            "residue_mask": tgt_rmask,
            "aatype": aatype,
            "residue_index": residue_index,
            "chain_id": chain_id,
        }
        anchor_frame_dict = {
            "coords": anchor_coords,
            "residue_mask": anchor_rmask,
            "aatype": aatype,
            "residue_index": residue_index,
            "chain_id": chain_id,
        }
        h_teacher = self._teacher_encode_clip(tgt_frame_dict)        # [B, L, dH]
        h_teacher_anchor = self._teacher_encode_clip(anchor_frame_dict)  # [B, L, dH]
        if getattr(self.config, "delta_target", True):
            delta_teacher = (h_teacher - h_teacher_anchor).detach()     # [B, L, dH]
            target_factors = self._project_to_factors(delta_teacher)
        else:
            # Absolute teacher target.  Structurally-faithful H (driven by
            # struct_recon) is already time-varying, so no delta is needed and
            # H keeps the fold the decoder must read.
            delta_teacher = (h_teacher - h_teacher_anchor).detach()
            target_factors = self._project_to_factors(h_teacher.detach())

        return {
            "pred_factors": pred,
            "target_factors": target_factors,
            "context_H": H_target_in,
            # Full unmasked context trajectory [B,K,L,dH] for the temporal
            # contrast loss: the contrast must measure the true H trajectory.
            "context_H_full": context_H,
            "valid_mask": valid,
            "h_teacher_full": h_teacher.detach(),
            "h_teacher_anchor": h_teacher_anchor.detach(),
            "delta_teacher": delta_teacher,
            "lag_ps": lag_ps,
            "lag_slot": slot,
            "lag_sel": lag_sel,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _teacher_encode_clip(self, frame_dict: dict) -> torch.Tensor:
        """Encode a clip dict (coords [B,T,L,4,3]) with teacher → [B, L, dH]."""
        assert not self.teacher_encoder.training
        with torch.no_grad():
            h = self.teacher_encoder(
                coords=frame_dict["coords"],
                residue_mask=frame_dict["residue_mask"],
                aatype=frame_dict["aatype"],
                residue_index=frame_dict["residue_index"],
                chain_id=frame_dict["chain_id"],
            )   # [B, 1, L, dH]
        return h[:, 0]   # [B, L, dH]

    @staticmethod
    def _sample_spatial_mask(valid: torch.Tensor, ratio: float) -> torch.Tensor:
        """Sample contiguous-block spatial mask for valid residues.

        Args:
            valid: [B, L] bool
            ratio: fraction of valid residues to mask

        Returns:
            mask: [B, L] bool (True = masked/hidden from encoder)
        """
        B, L = valid.shape
        mask = torch.zeros(B, L, dtype=torch.bool, device=valid.device)
        for b in range(B):
            valid_idx = valid[b].nonzero(as_tuple=False).squeeze(1)
            n_valid = valid_idx.shape[0]
            if n_valid < 2:
                continue
            n_mask = max(1, int(n_valid * ratio))
            # Random contiguous block
            start = torch.randint(0, n_valid - n_mask + 1, ()).item()
            mask[b, valid_idx[start: start + n_mask]] = True
        return mask
