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
from contextlib import contextmanager
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
from protein_jepa_frame.data.protocol import validate_endpoint_times


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
    # Within-frame canonical-frame position channel (2026-10-18).  Adds a
    # per-residue CA position in the frame's own PCA frame, giving H a reference
    # in which "how residue i moved relative to the anchor" is expressible.
    # Without it H is a purely local, position-blind invariant descriptor and a
    # probe cannot generalise the displacement to unseen frames (measured
    # R2=0.09 held-out vs 1.0 overfit).
    relative_position_features: bool = True
    # Scale on the time-invariant identity embeddings (aa / residue-index /
    # chain).  Kept small so the geometry signal drives H rather than a static
    # identity code (see encoder.py note on time-invariant collapse).
    identity_embed_scale: float = 0.1

    # OPF
    num_factors: int = 4          # J; factor_dim r = latent_dim // num_factors
    opf_learnable: bool = False
    opf_mode: str = "fixed_orthogonal"
    deterministic_temporal: bool = True
    masked_identity_queries: bool = True

    # Predictor
    predictor_depth: int = 4
    predictor_spatial_depth: int = 2   # Fix 1: spatial stage depth
    predictor_heads: int = 4
    max_frames: int = 256         # for positional encoding

    # EMA schedule: m stays at ema_start for ema_warmup_steps, then increases
    # linearly to ema_end over the next 2*ema_warmup_steps. Two properties matter
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

    # Explicit output contract. All temporal APIs predict teacher-space changes;
    # synthesize H_future = compose(decompose(H_anchor) + delta_factors).
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
        if self.opf_mode not in {"fixed_orthogonal", "soft_gram", "qr_retraction"}:
            raise ValueError(f"unknown opf_mode: {self.opf_mode}")
        if self.opf_learnable and self.opf_mode == "fixed_orthogonal":
            self.opf_mode = "soft_gram"
        elif self.opf_mode != "fixed_orthogonal":
            self.opf_learnable = True
        if not self.delta_target:
            raise ValueError("Stage A v3 predictor emits delta_factors; delta_target must be True")


@contextmanager
def deterministic_module(module: nn.Module):
    """Disable stochastic layers while preserving autograd and prior modes."""
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield module
    finally:
        for child, mode in modes:
            child.training = mode


def _context_to_target_lag_ps(batch: dict, K: int) -> torch.Tensor:
    """Deprecated legacy first-future helper; new endpoint APIs do not call it.

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
    """Two-stage predictor: causal temporal attention then spatial residue attention.

    Stage 1 — temporal (per-residue, causal over K frames):
        Each residue independently aggregates its own history with causal
        self-attention.  The last-frame token becomes the temporal summary.

    Stage 2 — spatial (full residue attention on the temporal summary):
        Full bidirectional attention over all L residues lets the predictor
        model inter-residue correlations and collective motion modes — the
        information the purely per-residue temporal stage cannot capture.

    This mirrors the V-JEPA / I-JEPA predictor design: context tokens first
    summarise their own history, then attend across space to form the target
    prediction.  For protein dynamics the spatial stage is critical because
    correlated fluctuations (collective modes) carry most of the predictable
    signal, and they only emerge through inter-residue interactions.

    Input:  H_ctx  [B, K, L, dH]  — context frames
            lag_ps     [B]         — physical lag τ in ps
            temperature_k  [B]     — temperature in Kelvin
            target_frame   [B,1,L,dH] | None  — masked anchor slot
    Output: pred_factors  [B, L, J, r]
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
        spatial_depth: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_factors = num_factors
        self.factor_dim = factor_dim

        self.pos_enc = _PositionalEncoding(latent_dim, max_frames)
        self.time_embed = _TimeEmbedding(latent_dim)
        # Task embedding: distinguishes the two prediction regimes that share
        # this predictor.  Spatial masking uses lag_ps = 0 (same-frame
        # reconstruction) while the temporal objective uses lag_ps ~ 1e3–1e4 ps.
        # Without this the time-embedding MLP sees two disjoint input ranges and
        # the two tasks fight over the same weights.
        #   0 = spatial (within-frame masked reconstruction)
        #   1 = temporal (cross-frame causal prediction)
        self.task_embed = nn.Embedding(2, latent_dim)
        nn.init.normal_(self.task_embed.weight, std=latent_dim ** -0.5)
        # Queries retain known sequence identity without reading hidden geometry.
        self.query_aatype = nn.Embedding(21, latent_dim)
        self.query_residue = nn.Embedding(4096, latent_dim)
        self.query_chain = nn.Embedding(128, latent_dim)
        self.mask_query = nn.Parameter(torch.zeros(latent_dim))
        nn.init.normal_(self.mask_query, std=latent_dim ** -0.5)

        # Stage 1: causal temporal attention — per-residue over K frames
        # Input shape to transformer: [B*L, K, dH]
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=depth
        )

        # Stage 2: spatial attention — full bidirectional over L residues
        # Input shape: [B, L, dH] (the temporal summary at each position)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer, num_layers=spatial_depth
        )

        # Output head: per-residue, J×r prediction.
        # Small random init (not zero): with the delta-target objective the
        # initial target is near zero, and a zero-initialised head produces
        # exactly zero output and therefore zero gradient to the predictor on
        # the first steps.  A small normal keeps the initial loss bounded while
        # leaving the head trainable from step 0.
        self.out_head = nn.Linear(latent_dim, num_factors * factor_dim)
        nn.init.normal_(self.out_head.weight, std=0.02)
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
        valid_mask: torch.Tensor | None = None,
        task_id: int = 1,
        query_identity: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, K, L, D = H_ctx.shape
        if lag_ps.shape != (B,) or temperature_k.shape != (B,):
            raise ValueError("lag_ps and temperature_k must each have shape [B]")
        if not torch.isfinite(lag_ps).all() or not torch.isfinite(temperature_k).all():
            raise ValueError("physical lag and temperature must be finite")
        if (temperature_k <= 0).any() or (lag_ps < 0).any() or (task_id == 1 and (lag_ps <= 0).any()):
            raise ValueError("temperature must be positive and future lag must be strictly positive")

        # Optional masked-frame input: replace the last context slot with a
        # zeroed/masked token so the predictor cannot trivially copy anchor geometry.
        if target_frame is not None:
            x_hist = torch.cat([H_ctx[:, :-1], target_frame], dim=1)   # [B,K,L,D]
        else:
            x_hist = H_ctx
        if query_identity is not None:
            aa, position, chain = query_identity
            identity = (self.query_aatype(aa.clamp(0, 20))
                        + self.query_residue(position.clamp(0, self.query_residue.num_embeddings - 1))
                        + self.query_chain(chain.clamp(0, self.query_chain.num_embeddings - 1))
                        + self.mask_query)
            if query_mask is not None:
                identity = identity * query_mask.unsqueeze(-1)
            x_hist = x_hist + identity[:, None]

        # Time conditioning injected into the last frame token only.
        # task_id selects the regime: 0 = spatial (within-frame reconstruction),
        # 1 = temporal (cross-frame prediction).  The embedding is added on top
        # of the log-time projection so the two tasks share geometry but not the
        # conditioning range (spatial passes lag_ps=0, temporal ~1e3–1e4 ps).
        time_cond = self.time_embed(lag_ps, temperature_k)   # [B, D]
        task = self.task_embed(
            torch.full((B,), int(task_id), dtype=torch.long, device=H_ctx.device)
        )   # [B, D]
        cond = time_cond + task

        x = self.pos_enc(x_hist)                              # [B, K, L, D]
        x[:, -1] = x[:, -1] + cond[:, None, :]

        # ── Stage 1: temporal causal attention (per residue) ─────────────────
        # Reshape [B, K, L, D] → [B*L, K, D] so each residue runs independently
        x_t = x.permute(0, 2, 1, 3).reshape(B * L, K, D)
        causal = self._causal_mask(K, x_t.device)
        # Pass the explicit additive mask only.  Passing both `mask` and
        # `is_causal=True` is version-dependent in PyTorch (some releases ignore
        # the explicit mask), so we use the single unambiguous form.
        x_t = self.temporal_transformer(x_t, mask=causal)
        # Take the last-frame summary for each residue: [B*L, D] → [B, L, D]
        summary = x_t[:, -1, :].reshape(B, L, D)

        # ── Stage 2: spatial attention (full bidirectional over residues) ─────
        # Build key-padding mask from valid_mask so padding residues don't
        # contribute to or receive attention: True = ignore.
        key_padding_mask: torch.Tensor | None = None
        if valid_mask is not None:
            # valid_mask: [B, L] bool, True = valid; transformer expects True = pad
            key_padding_mask = ~valid_mask   # [B, L]
        summary = self.spatial_transformer(
            summary, src_key_padding_mask=key_padding_mask
        )   # [B, L, D]

        # Project to factor predictions
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
            relative_position_features=config.relative_position_features,
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
            orthogonality_mode=("soft_gram" if config.opf_mode == "fixed_orthogonal" else config.opf_mode),
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
            spatial_depth=config.predictor_spatial_depth,
        )

        # EMA step counter (used externally to compute momentum schedule)
        self.register_buffer("_ema_step", torch.zeros((), dtype=torch.long))

        # Structure-reconstruction head: regress CA-CA distance from a pair of
        # H vectors.  Symmetric in (i, j) by construction (concat both orders
        # and average).  Trained jointly with the encoder; keeps H structurally
        # faithful so the downstream decoder has a signal to read.
        #
        # Input is [H_i, H_j, H_mean]: the per-residue pair PLUS the frame's
        # global mean.  Per-residue H alone is only locally informative under
        # SE(3)-invariance (no canonical frame exists to place residue i in),
        # so a head reading [H_i, H_j] cannot recover the i-j distance and
        # collapses to predicting the batch mean.  The global mean supplies the
        # missing reference frame, exactly as it does for the downstream
        # decoder (per-residue alone → 4.38 Å; with global context → 1.17 Å).
        # The constraint on H is preserved: H_mean is identical for every pair
        # in a frame, so H_i and H_j must still carry pair-specific geometry.
        dH = config.latent_dim
        self.struct_recon_head = nn.Sequential(
            nn.Linear(3 * dH, config.struct_recon_dim),
            nn.GELU(),
            nn.Linear(config.struct_recon_dim, config.struct_recon_dim),
            nn.GELU(),
            nn.Linear(config.struct_recon_dim, 1),
        )

        # Structural-displacement readout head (P0-A, 2026-10-18).
        #
        # Reads the PREDICTED factor delta (+ anchor H + global mean H) and emits
        # a per-residue CA displacement expressed in the anchor residue-local
        # frame.  It is supervised by the TRUE future frame's Kabsch-aligned
        # internal CA displacement.  This is the minimal structure supervision
        # consistent with JEPA-Anything: the encoder is still trained by the
        # temporal factor-prediction loss, but that loss now also has to carry
        # enough information for this head to recover the real displacement.
        # The head is trained jointly and discarded after Stage A.
        #
        # Input: the predicted H change alone (dH-wide).  No anchor_H / H_mean
        # side-channel: those let the head recover the target's global structure
        # from static context and ignore the predicted change, so the displacement
        # signal never reached H (measured: within-protein probe ratio unchanged
        # at 0.946 when the aux was added).  Reading only the change forces the
        # gradient onto H itself.
        self.displacement_head = nn.Sequential(
            nn.Linear(dH, config.struct_recon_dim),
            nn.GELU(),
            nn.Linear(config.struct_recon_dim, config.struct_recon_dim),
            nn.GELU(),
            nn.Linear(config.struct_recon_dim, 3),
        )
        # Small-norm output init (NOT zero): a zero last layer would block all
        # gradient to its input at step 0 (dL/d_in = W_out^T @ ... = 0), so the
        # encoder/predictor would receive no displacement signal at init.  A
        # tiny nonzero init keeps the head near "no displacement" (= persistence)
        # while letting gradients flow through from the first step.
        nn.init.normal_(self.displacement_head[-1].weight, std=1e-2)
        nn.init.zeros_(self.displacement_head[-1].bias)

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

    def encode_temporal(self, **kwargs) -> torch.Tensor:
        """Deterministic, differentiable structural H for dynamic objectives."""
        if self.config.deterministic_temporal:
            with deterministic_module(self.context_encoder):
                return self.context_encoder(**kwargs)
        return self.context_encoder(**kwargs)

    @torch.no_grad()
    def opf_diagnostics(self) -> dict[str, float]:
        basis = self.opf.analysis_basis().reshape(self.config.latent_dim, -1)
        singular = torch.linalg.svdvals(basis.float())
        return {"opf/sv_min": singular.min().item(), "opf/sv_max": singular.max().item(),
                "opf/condition": (singular.max() / singular.min().clamp_min(1e-12)).item()}

    def after_optimizer_step(self) -> None:
        self.opf.after_optimizer_step(int(self._ema_step.item()) + 1)

    # ── Structure reconstruction auxiliary ───────────────────────────────────

    def struct_recon_loss(
        self,
        context_H: torch.Tensor,     # [B, K, L, dH]  online H for context frames
        coords: torch.Tensor,         # [B, K, L, 4, 3]  backbone coords (N,CA,C,O)
        residue_mask: torch.Tensor,   # [B, K, L]
        n_pairs: int | None = None,
    ) -> torch.Tensor:
        """Regress CA-CA distances from H residue pairs.

        Three design points, each fixing a measured failure of the naive form
        (raw MSE on [H_i, H_j]):

        * **Global context.**  The head sees [H_i, H_j, H_mean] where H_mean is
          the frame's mean over valid residues.  Under SE(3) invariance H_i
          carries only local geometry, so the i-j distance is unrecoverable
          from the pair alone; H_mean supplies the missing reference frame.
          The constraint on H is unchanged — H_mean is shared by every pair,
          so H_i and H_j must still encode pair-specific geometry.

        * **log1p target.**  Raw CA-CA distances span 0–55 Å with variance
          ≈ 63 Å².  Under squared error a handful of long-range pairs dominate
          the gradient and the head minimises loss by predicting the frame
          mean on every pair (measured: pred std 1.97 Å vs target std 7.92 Å,
          MSE/baseline = 1.007 — a dead head).  Compressing the target with
          log1p makes the residual distribution near-homoscedastic.

        * **Per-frame normalisation.**  The returned loss is divided by the
          per-frame variance of the (log1p) target, so it is O(1) — the same
          scale as the JEPA terms.  Without this the auxiliary entered the
          total loss at 0.3 × 100 = 31 against a JEPA total of 0.10 and
          consumed 99.7 % of the encoder's gradient.

        Returns a scalar loss O(1), not Å².
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
        ar = torch.arange(BK, device=dev)[:, None]  # [BK, 1]

        # Global reference: masked mean of H over valid residues, per frame.
        mf = mask_flat.float().unsqueeze(-1)                # [BK, L, 1]
        h_mean = (H_flat * mf).sum(dim=1, keepdim=True) / mf.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Stratified pair sampling by sequence separation.  Uniform sampling
        # buries the few local pairs (the ones carrying the tightest geometric
        # constraint) under an ocean of long-range pairs.  Three bins —
        # |i-j| <= 4 (backbone-local), 5–15 (medium), > 15 (tertiary contact)
        # — give each regime a third of the pairs.  All clamp to valid range.
        n_per = max(1, n_pairs // 3)
        sign = torch.where(torch.rand(BK, n_per, device=dev) < 0.5,
                           torch.ones(BK, n_per, dtype=torch.long, device=dev),
                           -torch.ones(BK, n_per, dtype=torch.long, device=dev))
        # local: offset in [1, 4]
        i_lo = torch.randint(0, L, (BK, n_per), device=dev)
        off_lo = torch.randint(1, 5, (BK, n_per), device=dev) * sign
        j_lo = (i_lo + off_lo).clamp(0, L - 1)
        # medium: offset in [5, 15]
        i_md = torch.randint(0, L, (BK, n_per), device=dev)
        off_md = torch.randint(5, 16, (BK, n_per), device=dev) * sign
        j_md = (i_md + off_md).clamp(0, L - 1)
        # long: offset >= 16, capped at L-1
        lo_lg = min(16, max(1, L - 1))      # guard for tiny proteins
        hi_lg = max(lo_lg + 1, L)
        i_lg = torch.randint(0, L, (BK, n_per), device=dev)
        off_lg = torch.randint(lo_lg, hi_lg, (BK, n_per), device=dev) * sign
        j_lg = (i_lg + off_lg).clamp(0, L - 1)
        idx_i = torch.cat([i_lo, i_md, i_lg], dim=1)        # [BK, 3*n_per]
        idx_j = torch.cat([j_lo, j_md, j_lg], dim=1)

        # H at each end of the pair — symmetric concat with global context.
        Hi = H_flat[ar, idx_i]   # [BK, n_pairs, dH]
        Hj = H_flat[ar, idx_j]
        g_i = h_mean.expand(-1, idx_i.shape[1], -1)         # [BK, n_pairs, dH]
        feat_ij = torch.cat([Hi, Hj, g_i], dim=-1)
        feat_ji = torch.cat([Hj, Hi, g_i], dim=-1)

        # Average over both orderings for symmetry
        pred = 0.5 * (
            self.struct_recon_head(feat_ij) + self.struct_recon_head(feat_ji)
        ).squeeze(-1)  # [BK, n_pairs]

        # Ground-truth CA-CA distance, log1p-compressed
        cai = ca_flat[ar, idx_i]   # [BK, n_pairs, 3]
        caj = ca_flat[ar, idx_j]
        dist = (cai - caj).norm(dim=-1)  # [BK, n_pairs]
        tgt = torch.log1p(dist)

        # Mask: both residues must be valid
        valid = mask_flat[ar, idx_i] & mask_flat[ar, idx_j]  # [BK, n_pairs]
        valid_f = valid.float()

        # Per-frame normalisation by the target variance, so the loss is O(1)
        # and comparable to the JEPA terms.  Mean-prediction ⇒ loss ≈ 1.
        tgt_mean = (tgt * valid_f).sum(dim=1, keepdim=True) / valid_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        tgt_dev = (tgt - tgt_mean) ** 2
        var = (tgt_dev * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
        var = var.clamp(min=1e-3)                            # [BK]
        se = (pred - tgt) ** 2 * valid_f                     # [BK, n_pairs]
        per_frame = se.sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)  # [BK]
        return (per_frame / var).mean()

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
        if batch.get("temperature_k") is None:
            raise ValueError("spatial JEPA requires an explicit temperature_k")
        temp_k = batch["temperature_k"]

        pred_full = self.predictor(
            H_ctx, lag_ps, temp_k, valid_mask=valid, task_id=0,
            query_identity=(aatype, residue_index, chain_id) if self.config.masked_identity_queries else None,
            query_mask=spatial_mask,
        )

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
        """Single endpoint API with the same delta contract as multi-lag training.

        The last future frame lies one declared horizon after the anchor.
        """
        return self.forward_temporal_multilag(batch, (0,), recent_frac=1.0)

    # ── Multi-lag temporal forward ────────────────────────────────────────────

    def forward_temporal_multilag(
        self,
        batch: dict,
        lags: tuple[int, ...],
        recent_frac: float = 0.5,
    ) -> dict:
        """Causal prediction: full context window → target frame OUTSIDE the window.

        The selected target comes from ``future_*`` and its physical timestamp
        must lie strictly after the last context frame. Unselected frames in a
        mirrored future clip can overlap context and never enter the teacher
        or predictor. No whole-window nonoverlap assumption is needed.

        ``lags`` controls which future offset is used for each sample.  When the
        batch supplies multiple horizons (``paired_all_horizons=True``), each
        sample already carries its own horizon via ``horizon_time_ps``; ``lags``
        then selects an index counted backward from the future clip's end.

        Args:
            batch: paired-window batch with ``context_*`` AND ``future_*`` keys.
            lags: future frame offsets within the future clip to supervise.
                  Each must be < T_fut. lag=0 selects its last frame, exactly
                  one declared horizon after the anchor for mirrored clips.
            recent_frac: fraction of each batch (by sample index) restricted to
                the two smallest backward offsets (closest to full horizon).

        Returns dict with the same keys as before so all downstream code is
        unchanged:
            pred_factors:    [B, L, J, r]
            target_factors:  [B, L, J, r]
            context_H:       [B, K, L, dH]  (full context, no zero-fill)
            context_H_full:  [B, K, L, dH]  (same; kept for variance/contrast)
            valid_mask:      [B, L]
            lag_ps:          [B]
            lag_slot:        [B] long
            lag_sel:         [B] long
            anchor_H:        [B, L, dH]  online encoding of last context frame
        """
        ctx_coords = batch["context_coords"]        # [B, K, L, 4, 3]
        ctx_rmask  = batch["context_residue_mask"]  # [B, K, L]
        fut_coords = batch["future_coords"]         # [B, T_fut, L, 4, 3]
        fut_rmask  = batch["future_residue_mask"]   # [B, T_fut, L]
        aatype        = batch["context_aatype"]
        residue_index = batch["context_residue_index"]
        chain_id      = batch["context_chain_id"]
        ctx_t    = batch.get("context_time_ps")
        fut_t    = batch.get("future_time_ps")
        temp_all = batch.get("temperature_k")
        if ctx_t is None or fut_t is None or temp_all is None:
            raise ValueError("temporal JEPA requires context_time_ps, future_time_ps, and temperature_k")

        B, K, L = ctx_coords.shape[:3]
        T_fut = fut_coords.shape[1]
        valid = ctx_rmask[:, 0]   # [B, L] — residues fixed across frames

        # ``lags`` are offsets counted from the END of the future clip.  The
        # future clip mirrors the context offsets shifted by ``horizon``, and
        # the context offsets are chronological with the anchor (= last context
        # frame) at the final offset.  Therefore the LAST future frame
        # (index T_fut-1) sits exactly ``horizon`` frames after the anchor — the
        # cleanest, longest-signal future target — and earlier future frames are
        # closer in absolute time to the anchor.  lag=0 selects the last future
        # frame (full horizon); larger lags step backward toward the anchor.
        lags = tuple(sorted({int(l) for l in lags}))
        if not lags or not 0 <= recent_frac <= 1:
            raise ValueError("lags must be nonempty and recent_frac must be in [0,1]")
        for l in lags:
            if l < 0 or l >= T_fut:
                raise ValueError(
                    f"lag {l} out of range for future clip T_fut={T_fut}"
                )

        # ── Encode full context window without stochastic differences ─────────
        context_H = self.encode_temporal(
            coords=ctx_coords,
            residue_mask=ctx_rmask,
            aatype=aatype,
            residue_index=residue_index,
            chain_id=chain_id,
        )   # [B, K, L, dH]

        # anchor = last visible context frame
        anchor_H = context_H[:, -1]   # [B, L, dH]

        # ── Sample which future offset to supervise per sample ────────────────
        # Slots are the sorted lag list; the two smallest lags (closest to the
        # full horizon) are the "short" slots that recent_frac prioritises.
        n_short  = max(1, min(len(lags), 2))
        short_slots = torch.arange(n_short, dtype=torch.long)
        long_slots  = torch.arange(n_short, len(lags), dtype=torch.long)
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

        # Map slot → future-clip index, counting from the end.
        lag_off = torch.tensor(
            [lags[int(s)] for s in slot.tolist()],
            dtype=torch.long, device=ctx_coords.device,
        )   # [B] — offset back from the last future frame
        lag_sel = (T_fut - 1 - lag_off).clamp(min=0)   # [B] index into future clip

        ar = torch.arange(B, device=ctx_coords.device)
        tgt_coords_sel = fut_coords[ar, lag_sel][:, None]   # [B,1,L,4,3]
        tgt_rmask_sel  = fut_rmask[ar, lag_sel][:, None]    # [B,1,L]

        # ── Physical lag (ps): anchor → selected future frame ─────────────────
        anchor_t = ctx_t[:, -1]
        fut_t_sel = fut_t[ar, lag_sel]
        validate_endpoint_times(ctx_t, fut_t_sel)
        lag_ps = (fut_t_sel - anchor_t).to(torch.float32)
        temp_k = temp_all

        # ── Predictor: full context window → future ───────────────────────────
        # The predictor receives the complete K-frame context (no zero-fill,
        # no masked anchor slot) so it can use the full history signal.
        # target_frame=None keeps the last slot as the real anchor encoding.
        pred = self.predictor(
            context_H, lag_ps, temp_k,
            valid_mask=valid, task_id=1,
        )   # [B, L, J, r]

        # ── Teacher target: future frame encoded by the EMA teacher ───────────
        tgt_frame_dict = {
            "coords": tgt_coords_sel,
            "residue_mask": tgt_rmask_sel,
            "aatype": aatype,
            "residue_index": residue_index,
            "chain_id": chain_id,
        }
        h_teacher = self._teacher_encode_clip(tgt_frame_dict)   # [B, L, dH]

        # Delta-target: predict displacement from anchor in teacher space.
        # Subtraction removes frame-invariant offsets from the target.
        # target = P_j · (h̄(future) - stop_grad(h̄(anchor)))
        # Delta losses still admit a temporally static encoder; structural
        # calibration and deterministic physical probes must detect that case.
        anchor_frame_dict = {
            "coords": ctx_coords[:, -1:],   # [B,1,L,4,3] last ctx frame
            "residue_mask": ctx_rmask[:, -1:],
            "aatype": aatype,
            "residue_index": residue_index,
            "chain_id": chain_id,
        }
        h_teacher_anchor = self._teacher_encode_clip(anchor_frame_dict)  # [B,L,dH]
        delta_teacher    = (h_teacher - h_teacher_anchor).detach()        # [B,L,dH]
        target_factors   = self._project_to_factors(delta_teacher)        # [B,L,J,r]

        # Anchor factor coordinates for anchor-calibrated reconstruction:
        #   H_hat(t) = compose( decompose(h_anchor) + pred_factors )
        # → pred_factors = 0 exactly reproduces the anchor (persistence fallback).
        #
        # IMPORTANT (P0-A fix, 2026-10-18): the anchor must come from the ONLINE
        # encoder and must NOT be detached.  Previously it was the *teacher*
        # encoding, detached, so the synthesized delta
        #   compose(anchor_factors + pred) - compose(anchor_factors)
        # had gradient ONLY through ``pred`` (the predictor) and NEVER through the
        # encoder's own anchor representation.  That let all displacement signal
        # live inside the predictor while the encoder's H stayed a purely local,
        # time-direction-agnostic descriptor (measured: H-delta -> true
        # displacement pattern R2 = 0.04).  Using the online anchor H
        # (= context_H[:, -1], already computed above and carrying gradient)
        # forces the encoder's anchor representation to be a usable origin for
        # the displacement readout.
        anchor_H_online = context_H[:, -1]                                # [B,L,dH]
        anchor_factors  = self._project_to_factors(anchor_H_online)       # [B,L,J,r] (grad)

        # ── P0-A target: TRUE internal CA displacement (anchor-local) ─────────
        # For each residue, the future CA cloud is Kabsch-aligned onto the anchor
        # CA cloud, and the displacement of the aligned target from the anchor is
        # expressed in the anchor's per-residue local frame.  This is exactly the
        # SE(3)-quotient quantity the decoder must reproduce (same definition as
        # B1a's target_delta_local), so supervising it here makes H's predicted
        # delta carry real structural change instead of a self-referential
        # teacher-latent delta.  Computed with the anchor geometry (no grad); the
        # selected future frame index is lag_sel per sample.
        target_delta_local = self._anchor_local_ca_displacement(
            target_coords=tgt_coords_sel[:, 0],     # [B, L, 4, 3]
            anchor_coords=ctx_coords[:, -1],        # [B, L, 4, 3]
            residue_mask=valid,                     # [B, L]
        )                                           # [B, L, 3]

        return {
            "pred_factors":    pred,
            "target_factors":  target_factors,
            # Feed the full context H (no zero-fill) so variance/contrast
            # losses see the real encoder outputs.
            "context_H":       context_H,
            "context_H_full":  context_H,
            "valid_mask":      valid,
            "h_teacher_full":  h_teacher.detach(),
            "h_teacher_anchor": h_teacher_anchor.detach(),
            "delta_teacher":   delta_teacher,
            # P0-A fix: anchor factors/H carry gradient to the online encoder.
            "anchor_factors":  anchor_factors,            # [B, L, J, r] (grad)
            "anchor_H":        anchor_H_online,           # [B, L, dH]  (grad)
            # Synthesized predicted future H and its factor delta, WITH grad:
            #   H_hat = compose(anchor_factors + pred_factors)
            # This is the actual full-magnitude predicted change the decomposition
            # is meant to encode, and is what the displacement readout reads.
            "synth_H":         self.opf.compose(anchor_factors + pred),
            "synth_delta_H":   self.opf.compose(anchor_factors + pred) - self.opf.compose(anchor_factors),
            "lag_ps":          lag_ps,
            "lag_slot":        slot,
            "lag_sel":         lag_sel,
            # P0-A structural displacement target + label for the readout head.
            "target_delta_local": target_delta_local,     # [B, L, 3]
            "anchor_coords":   ctx_coords[:, -1].detach(),  # [B, L, 4, 3]
        }

    @staticmethod
    def _anchor_local_ca_displacement(
        target_coords: torch.Tensor,   # [B, L, 4, 3]  target frame backbone (N, CA, C, O)
        anchor_coords: torch.Tensor,   # [B, L, 4, 3]  anchor frame backbone
        residue_mask: torch.Tensor,    # [B, L] bool
    ) -> torch.Tensor:
        """Kabsch-aligned internal CA displacement in the anchor local frames.

        Returns [B, L, 3]: for each residue, the future CA position after a
        single global Kabsch alignment of the target CA cloud onto the anchor CA
        cloud, minus the anchor CA, rotated into the anchor's per-residue local
        frame (x=C-CA, z=x×n, y=z×x).  A pure global rigid motion gives exactly
        zero; only internal conformational change is measured.  This is the same
        SE(3)-quotient target B1a supervises (``_kabsch_target_delta_local``).
        """
        B, L = anchor_coords.shape[:2]
        device = anchor_coords.device
        dtype = anchor_coords.dtype
        m = residue_mask.to(dtype).unsqueeze(-1)                # [B, L, 1]
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)       # [B, 1, 1]

        a_ca = anchor_coords[:, :, 1, :]                        # [B, L, 3]
        t_ca = target_coords[:, :, 1, :]
        a_c = (a_ca * m).sum(dim=1, keepdim=True) / denom
        t_c = (t_ca * m).sum(dim=1, keepdim=True) / denom
        P = (t_ca - t_c) * m                                    # target centred, [B,L,3]
        Q = (a_ca - a_c) * m                                    # anchor centred
        cov = P.transpose(1, 2) @ Q                             # [B, 3, 3]
        # Row-vector convention (see B1a): aligned = centred_target @ R_align ≈ Q,
        # with R_align = U D Vh.  Use float64 for a stable SVD.
        U, _, Vh = torch.linalg.svd(cov.to(torch.float64))
        d = torch.sign(torch.linalg.det(U @ Vh))
        d = torch.where(d == 0, torch.ones_like(d), d)
        D = torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0).expand(B, -1, -1).clone()
        D[:, 2, 2] = d
        R_align = (U @ D @ Vh).to(dtype)                        # [B, 3, 3]
        t_aligned = (t_ca - t_c) @ R_align + a_c                # target in anchor frame
        delta_global = t_aligned - a_ca                         # [B, L, 3]

        # Anchor per-residue local frame (matches encoder._local_backbone_frames).
        a_N = anchor_coords[:, :, 0, :]
        a_C = anchor_coords[:, :, 2, :]
        eps = 1e-8
        x_axis = F.normalize(a_C - a_ca, dim=-1, eps=eps)
        n_dir = F.normalize(a_N - a_ca, dim=-1, eps=eps)
        z_axis = F.normalize(torch.cross(x_axis, n_dir, dim=-1), dim=-1, eps=eps)
        y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=eps)
        R_local = torch.stack([x_axis, y_axis, z_axis], dim=-2)  # [B, L, 3, 3] rows=basis
        delta_local = (R_local @ delta_global.unsqueeze(-1)).squeeze(-1)  # [B, L, 3]
        return delta_local * m

    def displacement_readout_loss(
        self,
        pred_factors: torch.Tensor,      # [B, L, J, r]  predicted factor delta (with grad)
        target_delta_local: torch.Tensor,  # [B, L, 3]   true internal CA displacement
        residue_mask: torch.Tensor,      # [B, L] bool
        synth_delta_H: torch.Tensor,     # [B, L, dH]  predicted H change (with grad)
        direction_weight: float = 1.0,   # λ_dir on the per-residue direction term
        anchor_H: torch.Tensor | None = None,  # unused; kept for call-site compat
    ) -> torch.Tensor:
        """Supervise the predicted H change through a *structure-only* readout.

        P0-A rewrite (2026-10-18).  Two changes fix the measured failure where
        the displacement signal bypassed H and settled in the predictor:

        1. **No side-channel.**  The readout head reads ONLY the predicted H
           change ``synth_delta_H`` (dH-wide).  Previously the head also saw the
           anchored ``anchor_H`` and the frame-level ``H_mean``; those let the
           head reconstruct the *target pattern's* global structure from static
           context and ignore the predicted change (measured: adding the auxiliary
           left the within-protein probe ratio unchanged at 0.946).  With the
           change as the sole input, the ONLY way to lower this loss is for the
           predicted H change itself to carry the displacement information — and,
           because the anchor factors now come from the online encoder with grad,
           that gradient reaches H.

        2. **Direction term.**  The pattern MSE alone tolerates a head that gets
           the *magnitude* right but the *direction* wrong.  We add a per-residue
           cosine term ``(1 - cos(pred_c, tgt_c))`` so "which residue moves where"
           is supervised directly (measured H-delta vs displacement direction
           cosine was only 0.20).

        Both the predicted change and the target are made mean-free per frame and
        normalised by the target's residual variance, so the loss measures the
        *pattern* of the change (which residue moves where); a constant prediction
        scores ≈ 1 on the MSE term rather than 0.  The head is trained jointly and
        discarded after Stage A.
        """
        B, L, J, r = pred_factors.shape
        dH = J * r
        pf = synth_delta_H.reshape(B, L, dH)                    # ONLY input to head

        mf = residue_mask.to(pf.dtype).unsqueeze(-1)            # [B, L, 1]
        n = mf.sum(dim=1).squeeze(-1).clamp(min=1.0)            # [B]  (#valid residues)

        pred = self.displacement_head(pf)                       # [B, L, 3]

        # ── Mean-free (pattern) normalisation ────────────────────────────────
        # Remove the per-frame mean (common drift / horizon direction) so a head
        # that predicts only that constant scores ≈ 1 on the MSE term rather
        # than ~0.  See prior note: the raw form plateaued at 1.02 and never
        # moved because the constant-mean solution satisfied it.
        tgt_c = target_delta_local - (
            (target_delta_local * mf).sum(dim=1, keepdim=True) / n.view(B, 1, 1)
        )                                                      # [B, L, 3]
        pred_c = pred - (pred * mf).sum(dim=1, keepdim=True) / n.view(B, 1, 1)
        var = ((tgt_c ** 2) * mf).sum(dim=(1, 2)) / (n * 3.0)   # [B]
        var = var.clamp(min=1e-3)
        se = ((pred_c - tgt_c) ** 2 * mf).sum(dim=(1, 2)) / (n * 3.0)
        l_pattern = (se / var).mean()

        # ── Direction term: per-residue cosine between predicted and true ────
        # cosine_similarity over the last (3) axis, masked over valid residues.
        cos = F.cosine_similarity(pred_c, tgt_c, dim=-1)        # [B, L]
        l_dir = ((1.0 - cos) * mf.squeeze(-1)).sum() / mf.sum().clamp(min=1.0)

        return l_pattern + direction_weight * l_dir

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
