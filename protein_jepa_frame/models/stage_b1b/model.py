"""Autonomous H/S dynamics with a separate diagnostic Kalman likelihood.

Implements STAGE_ABC_REDESIGN_PLAN.md 2.2-2.4:
  - HSpaceEnergy:     scalar U(H;T,c), force = -grad U via autograd
  - StaticBathParams: D, C, Gamma depend on T and a fixed observed anchor feature
  - HistoryPosterior: causal GRU -> (m_0, chol_P_0) for bath initialisation
  - StageB1b:         differentiable autonomous sampling, diagnostic Kalman NLL
  - sample_step:      single-step Euler sampler (training/deployment identical)

Stop-grad contract (2.2):
  H_t = stop_grad(normalize_H(f_theta*(X_t)))  -- encoder frozen, normalizer frozen
  Energy U_eta and bath params (D, C, Gamma) are trainable.
  History posterior r_psi is trainable.
  R = 2*h*D (derived from D and h, NOT a separate learnable parameter).

Per-residue independent bath (2.1):
  Covariances are block-diagonal in L: each residue carries its own bath state
  S in R^{dS} with its own (dS x dS) covariance P.  The H transition noise R is
  diagonal in dH per residue.  This keeps the Kalman recursion O(L * dS^3)
  instead of O((L*dH)^3).

Notation:
  L  = residues (padded to max_residues),  dH = H latent dim,  dS = bath dim
  B  = batch,  N = observed future steps in the Kalman window
  h  = observation interval Delta_obs (ps), fixed, shared train/deploy
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class StageB1bConfig:
    latent_dim: int = 128        # dH; H is the state
    bath_dim: int = 8            # dS; 0 disables the bath (H-only ablation)

    energy_depth: int = 3
    energy_heads: int = 4
    energy_hidden: int = 128

    reference_temperature_k: float = 348.0

    # Bath parameter scales (static, (T,c)-conditioned).
    # Calibrated against the measured per-dim H innovation variance (~0.078 at
    # 1 ns on mdCATH): R = 2hD should be O(0.1-0.2).  The real prior bug was NOT
    # D but C: the bath's contribution to V = R + B_h P B_h^T is *quadratic* in
    # C (B_h = h*C), so with C initialised at ~0 its gradient w.r.t. C is exactly
    # zero and it can never grow.  We therefore initialise C at a value where
    # B_h P B_h^T is the same order as R, giving a non-zero gradient at step 0.
    diffusion_scale: float = 1.0e-4     # R_init = 2hD ~ 0.2 (true innovation var ~0.078)
    coupling_scale: float = 1.0e-3      # overall scale of C
    c_init_std: float = 1.0e-1          # C output init std: C_init ~ coupling_scale*std ~1e-4
    # The isolated bath Euler update A=1-h*Gamma is stable for 0<h*Gamma<2.
    # Negative A may oscillate without diverging. Coupled stability is stricter.
    # With h = 1000 ps the OLD gamma_floor=0.05 gave A = 1 - 50 = -49: the bath
    # covariance blew up (bath/R ~ 2e4).  Scale Gamma so h*Gamma stays in (0,1):
    # Gamma ~ 1e-4 -> A ~ 0.9, stable with a steady-state P_ss ~ O(1).
    gamma_scale: float = 2.0e-4
    gamma_floor: float = 2.0e-5         # minimum rate: h*gamma_floor = 0.02

    # History posterior
    history_hidden: int = 128

    # Integration: h = observation interval in picoseconds.
    # MDcath frame_interval_ps = 1000 (1 ns/frame).  The actual lag between the
    # last context frame and the first future frame is one frame.
    step_ps: float = 1000.0

    # Numerical jitter added to covariances for Cholesky stability
    jitter: float = 1.0e-6
    # Bounds the isolated bath Euler factor. Coupled stability still requires
    # held-out autonomous rollouts; this bound alone is not a stability proof.
    max_gamma_step: float | None = None  # legacy configs retain their original kernel
    # Optional elementwise clamp on the force magnitude |f| before it is scaled
    # by h*D in the drift.  A fresh (untrained) energy can emit large gradients,
    # and with h=1000 ps the drift h*D*f diverges within a few Euler steps.  The
    # clamp keeps the one-step update bounded so from-scratch rollouts are stable;
    # None preserves the legacy unbounded kernel exactly.
    force_clip: float | None = None
    # Quadratic-tail coefficient initialisation (see HSpaceEnergy).  Default 1e-3
    # is the legacy value; a larger init (e.g. 1.0) makes an untrained from-none
    # rollout mean-reverting.
    tail_init: float = 1.0e-3
    # Bath coupling convention.  This is the difference between a stable kernel
    # and a divergent one; see "why" below.
    #
    #   "force" (legacy, default for old checkpoints):
    #       mu_H = H + h*D*f + h*C*S,    S' = S + h*(C^T f - Gamma*S)
    #     At the bath steady state S -> C^T f / Gamma, so the H drift becomes
    #       h*(D + C Gamma^-1 C^T) * f
    #     The added term is POSITIVE DEFINITE: the bath *amplifies* mobility
    #     instead of providing friction.  It therefore adds a positive feedback
    #     loop (larger |H| -> larger |f| -> larger |S| -> larger |H|).  Measured
    #     on the 1000dom kernel: |C|_F/|D| = 4.2, i.e. a 15.4x mobility boost;
    #     |H| per residue runs 11 -> 3,969 over 64 one-ns steps.
    #
    #   "velocity" (FDT-correct):
    #       p  = f - C*S,                H' = H + h*D*p + sqrt(2hD) xi
    #       S' = S - h*Gamma*S + C^T*dH + sqrt(2hGamma) xi_S
    #     The bath is driven by the realised displacement dH rather than by the
    #     force, so it is anti-correlated with (H - H0) and acts as genuine
    #     friction; h*C*S is O(1) rather than O(|H|).  The fluctuation term is
    #     placed on H (the conservative FDT placement when D is singular), which
    #     keeps R = 2hD / W = 2hGamma exactly as the Kalman observer assumes.
    #     Measured: |H| 11 -> 18 and a 6.27 per-residue step against a true 6.87.
    # The default is "velocity": a checkpoint that predates this option carries
    # "force" in its saved kernel_config and therefore still loads with its
    # original (divergent) kernel, while every new run is stable by default.
    bath_coupling: str = "velocity"
    # Optional cap on the per-residue DRIFT norm |h*D*p| (latent units), applied
    # before the noise is added.  A one-ns frame genuinely moves H by ~6.9 per
    # residue, so a cap near 20 only ever fires on the runaway branch; it is a
    # guard, not a model term.  None off.
    max_step_norm: float | None = None
    # Truncated BPTT window (in rollout steps).  None keeps the full graph.
    # Long physical horizons need many steps: 64 ns is 64 steps at 1 ns/frame but
    # 640 at 100 ps/frame, and retaining the graph for all of them costs tens of
    # GB.  Only the last `rollout_grad_steps` transitions stay differentiable;
    # the forward pass and every horizon's value are unchanged.
    rollout_grad_steps: int | None = None

    def __post_init__(self):
        if self.latent_dim < 1 or self.bath_dim < 0:
            raise ValueError("latent_dim must be positive and bath_dim nonnegative")
        for name in ("step_ps", "diffusion_scale", "gamma_scale", "gamma_floor"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_gamma_step is not None:
            if not 0 < self.max_gamma_step < 2:
                raise ValueError("max_gamma_step must lie strictly between 0 and 2")
        if self.bath_coupling not in ("force", "velocity"):
            raise ValueError(
                f"bath_coupling must be 'force' or 'velocity', got {self.bath_coupling!r}")
        if self.bath_coupling == "force":
            # Old checkpoints carry this string in their saved kernel_config, so
            # they keep their original kernel instead of silently changing
            # behaviour.  That kernel diverges (|H| 11 -> 3,969 over 64 steps),
            # so loading one is a reproducibility aid, not a viable model.
            warnings.warn(
                "bath_coupling='force' is the legacy, structurally divergent "
                "kernel (the bath adds a positive-definite C*Gamma^-1*C^T "
                "mobility boost instead of friction). It is retained only to "
                "reproduce pre-2026-10-22 checkpoints; use 'velocity'.",
                DeprecationWarning, stacklevel=2)
        if self.max_step_norm is not None:
            if not math.isfinite(self.max_step_norm) or self.max_step_norm <= 0:
                raise ValueError("max_step_norm must be finite and positive")

    @property
    def use_bath(self) -> bool:
        return self.bath_dim > 0


# -----------------------------------------------------------------------------
# H-space energy: scalar potential U(H; T, c) via full self-attention
# -----------------------------------------------------------------------------

class HSpaceEnergy(nn.Module):
    """Scalar energy U(H; T, c) over H tokens; force = -grad_H U.

    Full (non-causal) residue self-attention: a permutation-equivariant
    function of the per-residue states with sequence-defined edges and no
    dependence on coordinates or an initial CA-distance graph (2.2).
    Temperature enters as a log-relative feature, not only a mobility scale.
    """

    def __init__(self, cfg: StageB1bConfig) -> None:
        super().__init__()
        d = cfg.latent_dim
        hid = cfg.energy_hidden
        self.reference_temperature_k = float(cfg.reference_temperature_k)

        self.in_proj = nn.Linear(d, hid)
        self.temperature_proj = nn.Sequential(
            nn.Linear(1, hid), nn.SiLU(), nn.Linear(hid, hid),
        )
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hid, nhead=cfg.energy_heads, dim_feedforward=hid * 2,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            for _ in range(cfg.energy_depth)
        ])
        # Per-residue energy contributions summed to a scalar.  Plus a quadratic
        # tail potential (2.2: "positive quadratic tail + bounded smooth
        # residual") to give a restoring trend without a force floor.
        self.energy_head = nn.Sequential(
            nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 1),
        )
        nn.init.normal_(self.energy_head[-1].weight, std=0.01)
        nn.init.zeros_(self.energy_head[-1].bias)
        # Quadratic tail coefficient (positive via softplus) per channel.  A
        # larger init gives an immediate restoring force, which keeps the Euler
        # rollout mean-reverting when the residual energy is untrained (from-none
        # training).  Default 1e-3 preserves the legacy kernel behaviour.
        tail0 = float(getattr(cfg, "tail_init", 1e-3))
        self.log_tail = nn.Parameter(torch.full((d,), math.log(math.expm1(max(tail0, 1e-12)))))

    def _temp_feature(self, temperature_k, B, device, dtype) -> torch.Tensor:
        if temperature_k is None:
            return torch.zeros(B, 1, device=device, dtype=dtype)
        temp = torch.as_tensor(temperature_k, device=device, dtype=dtype)
        if temp.ndim == 0:
            temp = temp.expand(B)
        temp = temp.reshape(B, -1)[:, -1]
        temp = torch.nan_to_num(
            temp, nan=self.reference_temperature_k,
            posinf=self.reference_temperature_k,
            neginf=self.reference_temperature_k,
        )
        return torch.log(temp.clamp(min=1.0) / self.reference_temperature_k).unsqueeze(-1)

    def per_residue_energy(
        self, H: torch.Tensor, residue_mask: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """H [B, L, dH] -> per-residue energy [B, L]."""
        B, L, _ = H.shape
        x = self.in_proj(H)
        temp = self._temp_feature(temperature_k, B, H.device, H.dtype)
        x = x + self.temperature_proj(temp).unsqueeze(1)
        pad = ~residue_mask                           # [B, L]; True = ignore
        for blk in self.blocks:
            x = blk(x, src_key_padding_mask=pad)
        e = self.energy_head(x).squeeze(-1)           # [B, L]
        tail = 0.5 * (F.softplus(self.log_tail) * H.pow(2)).sum(-1)   # [B, L]
        return (e + tail) * residue_mask.float()

    def forward(
        self, H: torch.Tensor, residue_mask: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Total scalar energy summed over valid residues."""
        return self.per_residue_energy(H, residue_mask, temperature_k).sum()


# -----------------------------------------------------------------------------
# Bath parameters held fixed throughout a rollout, conditioned on T and H0
# -----------------------------------------------------------------------------

class StaticBathParams(nn.Module):
    """D (dH diag), Gamma (dS diag), C (dH x dS) per residue.

    The fixed condition is the observed anchor H0 and temperature, not merely
    amino-acid sequence. The resulting process is explicitly conditioned on its
    origin conformation. H0 is held fixed, so there are no state-dependent
    mobility divergence terms within that conditional rollout. Detachment alone
    would not turn a conformation into a sequence feature.
    D and Gamma are positive diagonals; C is a full per-residue dH x dS matrix.
    """

    def __init__(self, cfg: StageB1bConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, dS = cfg.latent_dim, cfg.bath_dim
        hid = 64
        self.reference_temperature_k = float(cfg.reference_temperature_k)
        self.diffusion_scale = float(cfg.diffusion_scale)
        self.coupling_scale = float(cfg.coupling_scale)
        self.gamma_scale = float(cfg.gamma_scale)
        self.gamma_floor = float(cfg.gamma_floor)

        # Condition encoder: (sequence feature, temperature) -> hidden
        self.cond_proj = nn.Linear(d, hid)
        self.temp_proj = nn.Sequential(nn.Linear(1, hid), nn.SiLU(), nn.Linear(hid, hid))

        self.d_net = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, d))
        if cfg.use_bath:
            self.gamma_net = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, dS))
            self.c_net = nn.Sequential(nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, d * dS))
            # Non-zero init is essential: the bath term B_h P B_h^T is quadratic
            # in C, so C=0 gives zero gradient forever.  Start at O(c_init_std)
            # so the bath contributes at the same order as R from step 0.
            nn.init.normal_(self.c_net[-1].weight, std=float(getattr(cfg, "c_init_std", 1e-1)))
            nn.init.zeros_(self.c_net[-1].bias)
        # Bias D toward diffusion_scale and Gamma toward 1 at init.
        nn.init.zeros_(self.d_net[-1].weight); nn.init.zeros_(self.d_net[-1].bias)

    def _cond(self, seq_feature, temperature_k, B, L, device, dtype):
        # seq_feature: static per-residue feature [B, L, dH]; detached upstream.
        c = self.cond_proj(seq_feature)                                    # [B,L,hid]
        if temperature_k is None:
            temp = torch.zeros(B, 1, device=device, dtype=dtype)
        else:
            temp = torch.as_tensor(temperature_k, device=device, dtype=dtype)
            if temp.ndim == 0:
                temp = temp.expand(B)
            temp = temp.reshape(B, -1)[:, -1]
            temp = torch.nan_to_num(temp, nan=self.reference_temperature_k)
            temp = torch.log(temp.clamp(min=1.0) / self.reference_temperature_k).unsqueeze(-1)
        return F.silu(c + self.temp_proj(temp).unsqueeze(1))               # [B,L,hid]

    def forward(self, seq_feature, temperature_k):
        """Return (D, C, Gamma).

        D:     [B, L, dH]  positive diagonal mobility
        C:     [B, L, dH, dS]  per-residue coupling (or None if no bath)
        Gamma: [B, L, dS]  positive diagonal bath rate (or None if no bath)
        """
        B, L, _ = seq_feature.shape
        device, dtype = seq_feature.device, seq_feature.dtype
        cond = self._cond(seq_feature.detach(), temperature_k, B, L, device, dtype)

        D = self.diffusion_scale * F.softplus(self.d_net(cond) + _INV_SOFTPLUS_1)  # ~scale at init
        if not self.cfg.use_bath:
            return D, None, None
        dS = self.cfg.bath_dim
        gamma_raw = self.gamma_net(cond)
        gamma_unbounded = self.gamma_floor + self.gamma_scale * F.softplus(gamma_raw)
        Gamma = gamma_unbounded
        if self.cfg.max_gamma_step is not None:
            gamma_cap = self.cfg.max_gamma_step / self.cfg.step_ps
            if gamma_cap <= self.gamma_floor:
                raise ValueError("gamma_floor must be smaller than max_gamma_step / step_ps")
            # Smooth saturation; old checkpoints without this setting preserve
            # their original kernel and are diagnostic-only to the new trainer.
            Gamma = self.gamma_floor + (gamma_cap - self.gamma_floor) * torch.tanh(
                (gamma_unbounded - self.gamma_floor) / (gamma_cap - self.gamma_floor)
            )
        C = self.coupling_scale * self.c_net(cond).reshape(B, L, self.cfg.latent_dim, dS)
        return D, C, Gamma


_INV_SOFTPLUS_1 = math.log(math.expm1(1.0))   # softplus(x)=1


# -----------------------------------------------------------------------------
# History posterior r_psi: causal GRU -> (m_0, chol_P_0)
# -----------------------------------------------------------------------------

class HistoryPosterior(nn.Module):
    """r_psi(H_{<=0}, dt_{<=0}, T, c) -> (m_0, chol_P_0) per residue (2.3).

    Causal GRU over the history H tokens (+ dt + temperature features), run
    independently per residue (residues share GRU weights but not state).
    Outputs per-residue m_0 [.,dS] and a lower-triangular Cholesky factor
    chol_P_0 [.,dS,dS] with softplus-positive diagonal.  m_0 init ~ 0, P_0 ~ I.
    """

    def __init__(self, cfg: StageB1bConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, dS = cfg.latent_dim, cfg.bath_dim
        hid = cfg.history_hidden
        self.reference_temperature_k = float(cfg.reference_temperature_k)
        # Per-token input: H (dH) + log-dt (1) + log-rel-T (1)
        self.in_proj = nn.Linear(d + 2, hid)
        self.gru = nn.GRU(hid, hid, batch_first=True)
        n_tril = dS * (dS + 1) // 2
        self.head = nn.Linear(hid, dS + n_tril)
        # init: m_0 = 0, chol diag -> softplus=1 so P_0 = I
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self._tril_idx = torch.tril_indices(dS, dS)
        self._diag_mask = (self._tril_idx[0] == self._tril_idx[1])

    def forward(self, H_hist, dt_hist, temperature_k, residue_mask):
        """H_hist [B, K, L, dH], dt_hist [B, K] (ps), residue_mask [B, L].

        Returns m_0 [B, L, dS], chol_P_0 [B, L, dS, dS].
        """
        B, K, L, d = H_hist.shape
        dS = self.cfg.bath_dim
        device, dtype = H_hist.device, H_hist.dtype

        # dt feature (log ps), broadcast across residues
        dt = torch.as_tensor(dt_hist, device=device, dtype=dtype).reshape(B, K)
        dt = torch.log(dt.clamp(min=1.0)).unsqueeze(-1).unsqueeze(2).expand(B, K, L, 1)
        # temperature feature
        if temperature_k is None:
            tf = torch.zeros(B, K, L, 1, device=device, dtype=dtype)
        else:
            temp = torch.as_tensor(temperature_k, device=device, dtype=dtype)
            if temp.ndim == 0:
                temp = temp.expand(B)
            temp = temp.reshape(B, -1)[:, -1]
            temp = torch.log(temp.clamp(min=1.0) / self.reference_temperature_k)
            tf = temp.reshape(B, 1, 1, 1).expand(B, K, L, 1)

        x = torch.cat([H_hist, dt, tf], dim=-1)            # [B,K,L,d+2]
        x = self.in_proj(x)                                # [B,K,L,hid]
        # fold residues into batch, GRU over K
        x = x.permute(0, 2, 1, 3).reshape(B * L, K, -1)    # [B*L, K, hid]
        out, _ = self.gru(x)                               # [B*L, K, hid]
        last = out[:, -1]                                  # [B*L, hid]
        params = self.head(last)                           # [B*L, dS + n_tril]

        m0 = params[:, :dS].reshape(B, L, dS)
        tril = params[:, dS:]                              # [B*L, n_tril]
        chol = torch.zeros(B * L, dS, dS, device=device, dtype=dtype)
        i, j = self._tril_idx.to(device)
        chol[:, i, j] = tril
        # softplus the diagonal for positivity, offset so init diag=1
        diag = F.softplus(torch.diagonal(chol, dim1=1, dim2=2) + _INV_SOFTPLUS_1)
        chol = chol - torch.diag_embed(torch.diagonal(chol, dim1=1, dim2=2)) \
                    + torch.diag_embed(diag)
        chol = chol.reshape(B, L, dS, dS)
        return m0, chol


# -----------------------------------------------------------------------------
# Main Stage B1b model
# -----------------------------------------------------------------------------

class StageB1b(nn.Module):
    """Autonomous H/S dynamics; Kalman likelihood is diagnostic only.

    The encoder is NOT owned here: the training/eval script applies the frozen
    Stage A V2 encoder through the frozen B1a normalizer and passes H already
    stop_grad'ed.  All covariance algebra is per-residue (block diagonal in L).
    """

    def __init__(self, cfg: StageB1bConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.energy = HSpaceEnergy(cfg)
        self.bath = StaticBathParams(cfg)
        if cfg.use_bath:
            self.history_posterior = HistoryPosterior(cfg)
        # Source of S_0 when an observed history is supplied to rollout:
        #   'gru'   -- amortised history network (the historical default),
        #   'exact' -- closed-form Kalman posterior (exact_bath_posterior),
        #   'prior' -- unit-variance prior, i.e. ignore the history.
        # Set per run by the trainer so a checkpoint stays self-describing via
        # the recorded args; an unknown value is a hard error, never a silent
        # fallback to a different model.
        self.s0_mode = "gru"

    # -- Forces --------------------------------------------------------------

    def compute_forces(
        self, H: torch.Tensor, residue_mask: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """f = -grad_H U(H).  f has shape [B, L, dH]."""
        H_g = H if H.requires_grad else H.detach().requires_grad_(True)
        with torch.enable_grad():
            U = self.energy(H_g, residue_mask, temperature_k)
            (grad,) = torch.autograd.grad(
                U, H_g, grad_outputs=torch.ones_like(U),
                create_graph=create_graph, retain_graph=create_graph,
            )
        f = -grad * residue_mask.unsqueeze(-1).float()
        if self.cfg.force_clip is not None:
            c = float(self.cfg.force_clip)
            f = c * torch.tanh(f / c)
        return f

    # -- Mean one-step drift (used by sampler + Kalman predict) --------------

    def _drift_H(self, H, f, D, C, m_or_S, h):
        """mu_{t+1} = H + h*D*f + h*(C @ m_or_S).

        H,f,D: [B,L,dH]; C: [B,L,dH,dS] or None; m_or_S: [B,L,dS] or None.
        """
        mu = H + h * D * f
        if C is not None and m_or_S is not None:
            mu = mu + h * torch.einsum("blij,blj->bli", C, m_or_S)
        return mu

    # -- Versioned fixed-coefficient microstep sampler -----------------------

    def _collective_velocity_step_fixed(
        self, H, S, residue_mask, temperature_k, D, C, Gamma, dt,
        generator=None, create_graph=False, noise_H=None, noise_S=None,
        noise_factor=None,
    ):
        """One collective velocity step with anchor-fixed bath coefficients.

        This is the primitive used by ``microstep_v1``.  D/C/Gamma are computed
        once from the anchor feature for a physical macro path; only the force
        is reevaluated at each microstate.  ``noise_H`` and ``noise_S`` are
        standard-normal increments when supplied, which lets a refinement panel
        aggregate the same fine Brownian path at 2/4/8 microsteps.
        """
        if self.cfg.bath_coupling != "velocity":
            raise ValueError("microstep_v1 requires velocity bath coupling")
        rmask = residue_mask.unsqueeze(-1).float()
        if noise_H is None:
            noise_H = torch.randn(H.shape, device=H.device, dtype=H.dtype, generator=generator)
        if noise_factor is not None:
            noise_H = torch.einsum("bij,bjd->bid", noise_factor, noise_H)
        f = self.compute_forces(H, residue_mask, temperature_k, create_graph=create_graph)
        p = f if (not self.cfg.use_bath or S is None) else f - torch.einsum(
            "blij,blj->bli", C, S)
        root_D = D.sqrt()
        drive = root_D * p
        if noise_factor is not None:
            drive = torch.einsum("bij,bjd->bid", noise_factor.transpose(-1, -2), drive)
            drive = torch.einsum("bij,bjd->bid", noise_factor, drive)
        displacement = dt * root_D * drive + torch.sqrt(2.0 * dt * D) * noise_H
        H_next = (H + displacement) * rmask
        if not self.cfg.use_bath or S is None:
            return H_next, None
        if noise_S is None:
            noise_S = torch.randn(S.shape, device=S.device, dtype=S.dtype, generator=generator)
        S_next = S + torch.einsum("blij,bli->blj", C, displacement) \
            - dt * Gamma * S + torch.sqrt(2.0 * dt * Gamma) * noise_S
        return H_next, S_next * rmask

    def rollout_microstep_v1(
        self, H0, residue_mask, temperature_k, n_steps,
        H_hist=None, dt_hist=None, seq_feature=None, generator=None,
        S0=None, detach=True, return_bath=False, grad_steps=None,
        noise_factor=None, collective_mobility=False, microsteps=1,
        brownian_H=None, brownian_S=None, integrator="explicit",
        solver_tolerance=1e-6, solver_max_iterations=128, microstate_trace=None,
    ):
        """Roll out fixed physical macro steps using ``microsteps`` substeps.

        The macro observation interval remains ``cfg.step_ps``.  Bath
        coefficients are anchor-conditioned and held fixed over the complete
        path.  Optional Brownian tensors have shape
        ``[B, n_steps, microsteps, L, d]`` and contain standard-normal
        increments; callers can coarsen one fine path by summing and dividing
        by the square root of the aggregation factor.  This path is opt-in and
        does not alter the v1 Euler sampler.
        """
        if not collective_mobility:
            raise ValueError("microstep_v1 requires collective_mobility=True")
        if self.cfg.bath_coupling != "velocity":
            raise ValueError("microstep_v1 requires velocity bath coupling")
        if isinstance(microsteps, bool) or int(microsteps) != microsteps or microsteps < 1:
            raise ValueError("microsteps must be a positive integer")
        microsteps = int(microsteps)
        if integrator not in ("explicit", "coupled_midpoint_v1"):
            raise ValueError(f"unsupported microstep integrator: {integrator}")
        if integrator == "coupled_midpoint_v1":
            if not self.cfg.use_bath or self.cfg.force_clip is not None:
                raise ValueError("coupled midpoint requires a bath and an unclipped energy force")
        if seq_feature is None:
            seq_feature = H0
        B, L, dH = H0.shape
        if brownian_H is not None:
            expected = (B, int(n_steps), microsteps, L, dH)
            if tuple(brownian_H.shape) != expected:
                raise ValueError(f"brownian_H must have shape {expected}, got {tuple(brownian_H.shape)}")
        if brownian_S is not None:
            if not self.cfg.use_bath:
                raise ValueError("brownian_S requires a bath")
            expected = (B, int(n_steps), microsteps, L, self.cfg.bath_dim)
            if tuple(brownian_S.shape) != expected:
                raise ValueError(f"brownian_S must have shape {expected}, got {tuple(brownian_S.shape)}")

        S = None
        if self.cfg.use_bath:
            if S0 is not None:
                S = S0
            elif H_hist is None:
                dS = self.cfg.bath_dim
                m0 = torch.zeros(B, L, dS, device=H0.device, dtype=H0.dtype)
                chol0 = torch.eye(dS, device=H0.device, dtype=H0.dtype).expand(
                    B, L, dS, dS).contiguous()
                eps = torch.randn(m0.shape, device=H0.device, dtype=H0.dtype, generator=generator)
                S = m0 + torch.einsum("blij,blj->bli", chol0, eps)
            else:
                m0, chol0 = self.initial_bath_state(
                    H_hist, dt_hist, temperature_k, residue_mask,
                    seq_feature=seq_feature)
                eps = torch.randn(m0.shape, device=H0.device, dtype=H0.dtype, generator=generator)
                S = m0 + torch.einsum("blij,blj->bli", chol0, eps)

        # These are deliberately outside the time loop: mobility and coupling
        # are fixed by the observed anchor for the entire physical path.
        D, C, Gamma = self.bath(seq_feature, temperature_k)
        dt = self.cfg.step_ps / microsteps
        midpoint = None
        if integrator == "coupled_midpoint_v1":
            from protein_jepa_frame.models.lasp_midpoint import CoupledMidpoint
            midpoint = CoupledMidpoint(
                D, C, Gamma, F.softplus(self.energy.log_tail), residue_mask, dt,
                noise_factor, solver_tolerance, solver_max_iterations)
        traj = [H0]
        bath_traj = [S] if S is not None else None
        H = H0
        for i in range(int(n_steps)):
            for j in range(microsteps):
                previous_H, previous_S = H, S
                noise_H = None if brownian_H is None else brownian_H[:, i, j]
                noise_S = None if brownian_S is None else brownian_S[:, i, j]
                if midpoint is None:
                    H, S = self._collective_velocity_step_fixed(
                        H, S, residue_mask, temperature_k, D, C, Gamma, dt,
                        generator=generator, create_graph=not detach,
                        noise_H=noise_H, noise_S=noise_S, noise_factor=noise_factor,
                    )
                else:
                    if noise_H is None:
                        noise_H = torch.randn(H.shape, device=H.device, dtype=H.dtype, generator=generator)
                    if noise_S is None:
                        noise_S = torch.randn(S.shape, device=S.device, dtype=S.dtype, generator=generator)
                    force = self.compute_forces(H, residue_mask, temperature_k, create_graph=not detach)
                    H, S = midpoint.step(H, S, force, noise_H, noise_S)
                if microstate_trace is not None:
                    microstate_trace.append({
                        "macro_step": i, "micro_step": j,
                        "H_before": previous_H.detach().cpu(),
                        "S_before": previous_S.detach().cpu() if previous_S is not None else None,
                        "H_after": H.detach().cpu(),
                        "S_after": S.detach().cpu() if S is not None else None,
                        "solver": dict(midpoint.last_solve) if midpoint is not None else None,
                    })
            if not detach and grad_steps is not None and i < int(n_steps) - int(grad_steps):
                H = H.detach()
                if S is not None:
                    S = S.detach()
            traj.append(H)
            if bath_traj is not None:
                bath_traj.append(S)
        out = torch.stack(traj, dim=1)
        if detach:
            out = out.detach()
        if not return_bath:
            return out
        bath_out = torch.stack(bath_traj, dim=1) if bath_traj is not None else None
        if detach and bath_out is not None:
            bath_out = bath_out.detach()
        return out, bath_out

    # -- Sampler (single-step Euler; identical train/deploy) -----------------

    def sample_step(
        self, H, S, residue_mask, temperature_k, seq_feature=None, generator=None,
        create_graph: bool = False,
        noise_factor=None, collective_mobility: bool = False,
    ):
        """One Euler-Maruyama step of the GLE kernel (2.2).

        H [B,L,dH], S [B,L,dS] or None.  Returns (H_next, S_next).
        Uses R=2hD and W=2hGamma noise so it is identical to the Kalman kernel.
        """
        cfg = self.cfg
        h = cfg.step_ps
        if seq_feature is None:
            seq_feature = H
        D, C, Gamma = self.bath(seq_feature, temperature_k)       # D [B,L,dH]
        f = self.compute_forces(H, residue_mask, temperature_k, create_graph=create_graph)
        rmask = residue_mask.unsqueeze(-1).float()
        noise_H = torch.randn(H.shape, device=H.device, dtype=H.dtype, generator=generator)
        if noise_factor is not None:
            # Experimental spatial covariance; this changes the sampler covariance
            # and is not the block-diagonal Kalman/FDT model documented above.
            noise_H = torch.einsum('bij,bjd->bid', noise_factor, noise_H)

        if collective_mobility:
            if cfg.bath_coupling != 'velocity':
                raise ValueError('collective mobility requires velocity coupling')
            # Conditional on the observed anchor, M = sqrt(D) K sqrt(D).
            # The same M drives relaxation and fluctuations. With E=U+|S|²/2,
            # the joint mobility is [[M,MC],[C^T M,Gamma+C^T M C]].
            # It is PSD by factorization [I;C^T] M [I,C]+diag(0,Gamma).
            p = f if (not cfg.use_bath or S is None) else f-torch.einsum('blij,blj->bli',C,S)
            root_D=D.sqrt()
            drive=root_D*p
            if noise_factor is not None:
                drive=torch.einsum('bij,bjd->bid',noise_factor.transpose(-1,-2),drive)
                drive=torch.einsum('bij,bjd->bid',noise_factor,drive)
            drift=h*root_D*drive
            displacement=drift+torch.sqrt(2*h*D)*noise_H
            H_next=(H+displacement)*rmask
            if not cfg.use_bath or S is None:
                return H_next,None
            noise_S=torch.randn(S.shape,device=S.device,dtype=S.dtype,generator=generator)
            # Shared displacement includes the H noise and the compensating
            # friction C^T M C S; independent bath noise adds 2h Gamma.
            S_next=S+torch.einsum('blij,bli->blj',C,displacement)-h*Gamma*S+torch.sqrt(2*h*Gamma)*noise_S
            return H_next,S_next*rmask

        if cfg.bath_coupling == "velocity":
            # FDT-correct bath: driven by the realised displacement, so it is
            # anti-correlated with (H - H0) and damps rather than amplifies.
            p = f if (not cfg.use_bath or S is None) else f - torch.einsum(
                "blij,blj->bli", C, S)                            # [B,L,dH]
            dH = h * D * p
            if cfg.max_step_norm is not None:
                n = dH.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                dH = dH * (float(cfg.max_step_norm) / n).clamp(max=1.0)
            H_next = (H + dH + torch.sqrt(2.0 * h * D) * noise_H) * rmask
            if not cfg.use_bath or S is None:
                return H_next, None
            CtD = torch.einsum("blij,bli->blj", C, dH)            # [B,L,dS]
            noise_S = torch.randn(S.shape, device=S.device, dtype=S.dtype, generator=generator)
            S_next = S + CtD - h * Gamma * S + torch.sqrt(2.0 * h * Gamma) * noise_S
            return H_next, S_next * rmask

        mu_H = self._drift_H(H, f, D, C, S, h)
        dH = mu_H - H
        if cfg.max_step_norm is not None:
            n = dH.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            dH = dH * (float(cfg.max_step_norm) / n).clamp(max=1.0)
            mu_H = H + dH
        H_next = mu_H + torch.sqrt(2.0 * h * D) * noise_H
        H_next = H_next * residue_mask.unsqueeze(-1).float()

        if not cfg.use_bath or S is None:
            return H_next, None

        # S_{t+1} = S + h*(C^T f - Gamma*S) + sqrt(2 h Gamma) eps_S
        CtF = torch.einsum("blij,bli->blj", C, f)                 # [B,L,dS]
        mu_S = S + h * (CtF - Gamma * S)
        noise_S = torch.randn(mu_S.shape, device=S.device, dtype=S.dtype, generator=generator)
        S_next = mu_S + torch.sqrt(2.0 * h * Gamma) * noise_S
        S_next = S_next * residue_mask.unsqueeze(-1).float()
        return H_next, S_next

    # -- Exact bath posterior under the sampler dynamics ---------------------

    def exact_bath_posterior(
        self, H_hist, dt_hist, temperature_k, residue_mask, seq_feature=None,
        initial_mean=None, initial_var=None,
    ):
        """Exact p(S_0 | H_hist) as a Kalman filter over the sampler dynamics.

        ``sample_step`` (collective mobility) evolves

            H_{k+1} = H_k + h_k D_k (f_k - C_k S_k) + sqrt(2 h_k D_k) eps_H
            S_{k+1} = S_k + C_k^T (H_{k+1} - H_k) - h_k Gamma_k S_k + sqrt(2 h_k Gamma_k) eps_S

        Reading H_{k+1} as a (linear-Gaussian) observation of S_k,

            z_k := dH_k - h_k D_k f_k = -h_k D_k C_k S_k + sqrt(2 h_k D_k) eps_H,

        so the observation matrix is ``C_obs = -h D C`` with diagonal noise
        ``R = 2 h D``.  A standard Kalman update followed by the sampler's own
        time update yields the exact filtered posterior.  No amortised network
        and no training signal are needed.

        NOTE ON THE DRIVE-ONLY APPROXIMATION.  It is tempting to say that, since
        the H noise is revealed by the observed increment, the update collapses
        to

            S_{k+1} = (1 - h Gamma) S_k + C^T dH_k + eta,   eta ~ N(0, 2 h Gamma),

        with no Kalman gain.  That recursion propagates the MEAN correctly but it
        is not the posterior: it ignores the fact that observing H_{k+1} also
        CONSTRAINS S_k through the innovation (the information in z_k about S_k
        is ``h D C / sqrt(2 h D)`` per step, not zero).  Measured on real
        anchors, the drive-only mean disagrees with the exact filtered mean by
        76-85 % even at the trained scale, so it is not used here.  It is kept
        only as the ``mean_only`` branch below for diagnostics.

        Args:
          H_hist: [B, K, L, dH] observed history, oldest first, ending at H_0.
          dt_hist: [B, K] interval in ps between frame k and k+1 (last unused).
          seq_feature: [B, L, dH] anchor feature for the static bath params.
            Defaults to H_hist[:, -1], which is the rollout anchor.

        Returns:
          (m0, chol0) with shapes [B, L, dS] and [B, L, dS, dS]; chol0 is lower
          triangular, so ``m0 + chol0 @ eps`` samples the posterior.
        """
        cfg = self.cfg
        dS = cfg.bath_dim
        B, K, L, d = H_hist.shape
        if dt_hist.shape != (B, K):
            raise ValueError("dt_hist must be [B, K] matching H_hist")
        if not dt_hist[:, :K - 1].gt(0).all():
            raise ValueError("history intervals must be positive")
        device, dtype = H_hist.device, H_hist.dtype
        if seq_feature is None:
            seq_feature = H_hist[:, -1]
        D, C, Gamma = self.bath(seq_feature, temperature_k)       # [B,L,*]
        if C is None:
            raise ValueError("exact_bath_posterior requires a bath")

        # Prior: S is unknown at the oldest frame, so start from unit variance.
        m = (torch.zeros(B, L, dS, device=device, dtype=dtype)
             if initial_mean is None else initial_mean.expand(B, L, dS))
        P = (torch.eye(dS, device=device, dtype=dtype).expand(B, L, dS, dS).clone()
             if initial_var is None
             else torch.diag_embed(initial_var.expand(B, L, dS)))

        for k in range(K - 1):
            h_k = dt_hist[:, k].view(B, 1, 1).to(dtype)           # [B,1,1]
            dH = H_hist[:, k + 1] - H_hist[:, k]                  # [B,L,dH]
            Hk = H_hist[:, k]
            f = self.compute_forces(Hk, residue_mask, temperature_k, create_graph=False)

            # Observation: z = dH - h D f = C_obs S + noise, C_obs = -h D C.
            # h_k is [B,1,1]; promote to [B,1,1,1] to broadcast against C's 4D.
            R = 2.0 * h_k * D                                     # [B,L,dH]
            B_h = h_k.unsqueeze(-1) * D.unsqueeze(-1) * C         # [B,L,dH,dS]
            z = dH - h_k * D * f                                  # [B,L,dH]
            # Innovation for the current S estimate: e = z - C_obs m = z + B_h m.
            e = z + torch.einsum("blij,blj->bli", B_h, m)         # [B,L,dH]
            V = torch.diag_embed(R) + B_h @ P @ B_h.transpose(-1, -2)
            V = 0.5 * (V + V.transpose(-1, -2))
            # K = P C_obs^T V^-1 = -P B_h^T V^-1
            Pbt = P @ B_h.transpose(-1, -2)                        # [B,L,dS,dH]
            K = -torch.linalg.solve(V, Pbt.transpose(-1, -2)).transpose(-1, -2)
            m_plus = m + torch.einsum("blsk,blk->bls", K, e)
            # Joseph form; K C_obs = -K B_h.
            KC = -torch.einsum("blsk,blkt->blst", K, B_h)
            ImKC = torch.eye(dS, device=device, dtype=dtype) - KC
            P_plus = ImKC @ P @ ImKC.transpose(-1, -2) \
                + torch.einsum("blsk,blk,bltk->blst", K, R, K)
            P_plus = 0.5 * (P_plus + P_plus.transpose(-1, -2))

            # Sampler time update: S' = (1 - h Gamma) S + C^T dH + sqrt(2hG) eta.
            a = 1.0 - h_k * Gamma
            m = a * m_plus + torch.einsum("blij,bli->blj", C, dH)
            Am = torch.diag_embed(a)
            P = Am @ P_plus @ Am.transpose(-1, -2) + torch.diag_embed(2.0 * h_k * Gamma)
            P = 0.5 * (P + P.transpose(-1, -2))

        chol = torch.linalg.cholesky(P + cfg.jitter * torch.eye(
            dS, device=device, dtype=dtype))
        rm = residue_mask.unsqueeze(-1).to(dtype)
        return m * rm, chol * rm.unsqueeze(-1)

    def initial_bath_state(
        self, H_hist, dt_hist, temperature_k, residue_mask, seq_feature=None,
    ):
        """Return (m_0, chol_0) for the bath prior at the rollout anchor.

        Dispatches on ``self.s0_mode`` so every rollout path (``rollout`` and
        ``rollout_microstep_v1``) shares one definition:

          'gru'   -- amortised history network (historical default),
          'exact' -- closed-form Kalman posterior (exact_bath_posterior),
          'prior' -- unit-variance prior, history ignored.
        """
        mode = getattr(self, "s0_mode", "gru")
        if mode == "exact":
            return self.exact_bath_posterior(
                H_hist, dt_hist, temperature_k, residue_mask, seq_feature=seq_feature)
        if mode == "gru":
            return self.history_posterior(H_hist, dt_hist, temperature_k, residue_mask)
        if mode == "prior":
            B, L = residue_mask.shape
            dS = self.cfg.bath_dim
            m0 = torch.zeros(B, L, dS, device=H_hist.device, dtype=H_hist.dtype)
            chol0 = torch.eye(dS, device=H_hist.device,
                              dtype=H_hist.dtype).expand(B, L, dS, dS).contiguous()
            return m0, chol0
        raise ValueError(f"unknown s0_mode {mode!r}")

    # -- Diagnostic Kalman innovation NLL; never optimized by the trainer ---

    def kalman_nll(
        self, H_hist, dt_hist, H_seq, residue_mask, temperature_k,
        return_diagnostics: bool = False,
    ):
        """Diagnostic likelihood scoring H_{1:N} | H_{-K+1:0}.

        This observer-conditioned recursion is for density diagnostics only.
        Every optimized B1b loss instead scores autonomous history-only samples.

        Args:
          H_hist:  [B, K, L, dH]  history frames (stop_grad'ed), for r_psi only.
          dt_hist: [B, K]         history frame intervals (ps).
          H_seq:   [B, N+1, L, dH] observed states [H_0, H_1, ..., H_N].
          residue_mask: [B, L]    valid residues.
          temperature_k: [B] or scalar.

        Returns scalar NLL (summed over steps and batch, normalised by nothing
        here -- the trainer divides by effective dims).  If no bath, falls back
        to the diagonal Gaussian transition NLL with the same R=2hD.
        """
        cfg = self.cfg
        h = cfg.step_ps
        B, T, L, d = H_seq.shape
        if H_hist.ndim != 4 or H_hist.shape[0] != B or H_hist.shape[2:] != (L, d):
            raise ValueError("diagnostic history and sequence shapes must match")
        if not torch.equal(H_hist[:, -1], H_seq[:, 0]):
            raise ValueError("diagnostic history must end at the shared H_seq anchor")
        if dt_hist.shape != H_hist.shape[:2] or not torch.isfinite(dt_hist).all() or not (dt_hist > 0).all():
            raise ValueError("diagnostic history intervals must be finite and positive")
        N = T - 1
        rmask = residue_mask.unsqueeze(-1).float()               # [B,L,1]
        device, dtype = H_seq.device, H_seq.dtype

        # Static bath params from the anchor (last history / first obs) frame.
        seq_feature = H_seq[:, 0]                                 # [B,L,dH]
        D, C, Gamma = self.bath(seq_feature, temperature_k)
        R = 2.0 * h * D                                           # [B,L,dH] diag of H noise
        if not torch.isfinite(R).all() or not (R > 0).all():
            raise FloatingPointError("diagnostic likelihood needs finite positive transition diffusion")

        total_nll = H_seq.new_zeros(())
        jitter_hits = 0

        if not cfg.use_bath:
            # Diagonal Gaussian: V = R, independent per channel.
            logdetV = torch.log(R).sum(-1)                      # [B,L]
            for n in range(N):
                Hn, Hn1 = H_seq[:, n], H_seq[:, n + 1]
                f = self.compute_forces(Hn, residue_mask, temperature_k, create_graph=True)
                mu = Hn + h * D * f
                e = (Hn1 - mu)                                    # [B,L,dH]
                quad = (e * e / R).sum(-1)                      # [B,L]
                step_nll = 0.5 * (quad + logdetV + d * math.log(2 * math.pi))
                total_nll = total_nll + (step_nll * residue_mask.float()).sum()
            if return_diagnostics:
                return total_nll, {"jitter_hits": 0}
            return total_nll

        dS = cfg.bath_dim
        # Linear-Gaussian bath recursion coefficients, matched to sample_step.
        # The sampler's mean is H + h*D*(f - C S), so the mean effect of S on H
        # is B_k = h*D*C and NOT h*C.  The old h*C form omitted the mobility D,
        # which made the diagnostic prediction of the bath's influence wrong by
        # a factor 1/D (~5e4) and let a regime pass as "calibrated" that was in
        # fact inert in the deployed kernel.
        A_diag = 1.0 - h * Gamma                                  # [B,L,dS]
        W_diag = 2.0 * h * Gamma                                  # [B,L,dS]
        B_h = h * D.unsqueeze(-1) * C                             # [B,L,dH,dS]
        if not torch.isfinite(R).all() or not (R > 0).all():
            raise FloatingPointError("diagnostic likelihood needs finite positive transition diffusion")
        I_dS = torch.eye(dS, device=device, dtype=dtype).expand(B, L, dS, dS)
        innov_whitened = [] if return_diagnostics else None
        bath_ratios = [] if return_diagnostics else None

        # Initial bath posterior from history network.
        m0, chol0 = self.history_posterior(H_hist, dt_hist, temperature_k, residue_mask)
        m_n = m0                                                  # [B,L,dS]
        P_n = chol0 @ chol0.transpose(-1, -2)                    # same prior as sampler

        log2pi = math.log(2 * math.pi)
        for n in range(N):
            Hn, Hn1 = H_seq[:, n], H_seq[:, n + 1]
            f = self.compute_forces(Hn, residue_mask, temperature_k, create_graph=True)

            # Predicted mean of H, matching sample_step's `f - C S`: the bath
            # opposes the energy drift, so the coupling enters with a MINUS.
            # The previous version added it, which inverted the bath's mean
            # effect and made this likelihood unable to score the deployed model.
            mu = Hn + h * D * f - torch.einsum("blij,blj->bli", B_h, m_n)
            e = Hn1 - mu                                          # [B,L,dH]

            # V = diag(R) + F F^T, F = B_h chol(P). All solves and
            # determinants use only dS x dS matrices (Woodbury), avoiding a
            # dense dH x dH covariance per residue. Numerical failures are
            # explicit; the diagnostic must not silently score another kernel.
            Lp, p_info = torch.linalg.cholesky_ex(P_n)
            if (p_info != 0).any() or not torch.isfinite(Lp).all():
                raise FloatingPointError("non-positive bath covariance in diagnostic likelihood")
            F_low = B_h @ Lp
            invR_F = F_low / R.unsqueeze(-1)
            middle = I_dS + F_low.transpose(-1, -2) @ invR_F
            middle = 0.5 * (middle + middle.transpose(-1, -2))
            Lm, m_info = torch.linalg.cholesky_ex(middle)
            if (m_info != 0).any() or not torch.isfinite(Lm).all():
                raise FloatingPointError("nonfinite low-rank innovation covariance")

            def solve_innovation(rhs):
                invR_rhs = rhs / R.unsqueeze(-1)
                correction = torch.cholesky_solve(F_low.transpose(-1, -2) @ invR_rhs, Lm)
                return invR_rhs - invR_F @ correction

            if return_diagnostics:
                # Ratio of the bath's mean diagonal contribution to R's: should be
                # O(1) for the bath to matter.  ~0 => bath is numerically dead.
                bd = F_low.square().sum(-1).mean()
                rd = R.mean().clamp_min(1e-12)
                bath_ratios.append(float((bd / rd).detach()))

            e_col = e.unsqueeze(-1)                               # [B,L,dH,1]
            sol = solve_innovation(e_col)
            quad = (e_col * sol).sum((-1, -2))                    # [B,L]
            logdetV = R.log().sum(-1) + 2*Lm.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            step_nll = 0.5 * (quad + logdetV + d * log2pi)
            total_nll = total_nll + (step_nll * residue_mask.float()).sum()

            if return_diagnostics:
                # Dense Cholesky is diagnostic-only to expose a conventional
                # whitened innovation; it never changes the scored covariance.
                V = torch.diag_embed(R) + F_low @ F_low.transpose(-1, -2)
                Lc = torch.linalg.cholesky(V)
                ev = torch.linalg.solve_triangular(Lc, e_col, upper=False)[..., 0]  # [B,L,dH]
                innov_whitened.append(ev.detach())

            # Joseph covariance update of the bath posterior.
            # Observation model: dH_n - h*D*f_n = -B_h S_n + noise, so the
            # effective observation matrix is C_obs = -B_h (NOT +B_h).  The
            # Kalman gain therefore carries a MINUS:
            #   K = P C_obs^T V^-1 = -P B_h^T V^-1.
            # The covariance form is unchanged by that sign, because Joseph
            # uses K C_obs = (-P B_h^T V^-1)(-B_h) = P B_h^T V^-1 B_h, in which
            # the two negatives cancel.  The MEAN update, however, is signed:
            # adding +K e here instead of -K e propagated the bath posterior in
            # the wrong direction and made this likelihood disagree with the
            # exact joint Gaussian.
            #
            # Vinv_Bh = V^-1 B_h  [B,L,dH,dS]
            Vinv_Bh = solve_innovation(B_h)
            K = -torch.einsum("blsi,blij->blsj", P_n, Vinv_Bh.transpose(-1, -2))  # [B,L,dS,dH]
            m_plus = m_n + torch.einsum("blsk,blk->bls", K, e)   # [B,L,dS]
            # Joseph form uses K C_obs with C_obs = -B_h.  With the signed gain
            # above, K C_obs = -(K B_h), so negate to keep the covariance
            # identical to the previously-correct (unsigned-gain) version.
            KB = -torch.einsum("blsk,blkt->blst", K, B_h)       # [B,L,dS,dS] = K C_obs
            ImKB = I_dS - KB
            P_plus = ImKB @ P_n @ ImKB.transpose(-1, -2) \
                   + torch.einsum("blsk,blk,bltk->blst", K, R, K)  # K R K^T (R diagonal)
            P_plus = 0.5 * (P_plus + P_plus.transpose(-1, -2))

            # Bath time update.  Under sample_step the S increment is
            #   C^T dH - h Gamma S + sqrt(2 h Gamma) eps_S.
            # Conditioning on the observed transition removes the H noise from
            # S entirely (eps_H is revealed by dH), so the exact recursion is
            #   S_{n+1} = (1 - h Gamma) S_n + C^T (H_{n+1} - H_n) + sqrt(2hG) eps_S.
            # Using the observed increment here is the filtering update, not a
            # second use of the observation.
            u_n = torch.einsum("blij,bli->blj", C, Hn1 - Hn)     # [B,L,dS]
            m_n = A_diag * m_plus + u_n
            A_mat = torch.diag_embed(A_diag)
            P_n = A_mat @ P_plus @ A_mat.transpose(-1, -2) + torch.diag_embed(W_diag)
            P_n = 0.5 * (P_n + P_n.transpose(-1, -2))

        if return_diagnostics:
            diag = {"jitter_hits": jitter_hits}
            if innov_whitened:
                # [N, B, L, dH] stacked whitened innovations
                diag["innov_whitened"] = torch.stack(innov_whitened, dim=0)
            if bath_ratios:
                diag["bath_ratio"] = float(sum(bath_ratios) / len(bath_ratios))
            return total_nll, diag
        return total_nll

    # -- Deployment rollout (sampling) ---------------------------------------

    def rollout(
        self, H0, residue_mask, temperature_k, n_steps,
        H_hist=None, dt_hist=None, seq_feature=None, generator=None,
        S0=None, detach: bool = True,
        return_bath: bool = False,
        grad_steps: int | None = None,
        noise_factor=None, collective_mobility: bool = False,
    ):
        """Rollout from H0 for n_steps.

        Deployment path (detach=True, no grad): initialises S_0 by SAMPLING
        from the history posterior (2.3), not just its mean.

        Training path (detach=False): gradients flow through the history
        posterior and the dynamics, so the energy score in (2.5) can train
        both r_psi and the kernel.

        ``grad_steps`` truncates backprop-through-time: the FORWARD pass still
        runs all ``n_steps`` (so long-horizon targets and the energy score are
        genuinely evaluated at e.g. 64 ns), but once the rollout is more than
        ``grad_steps`` steps old the recurrent state is detached from the graph.
        Autograd therefore retains only a sliding window of activations instead
        of the whole trajectory, which is what makes a 640-step rollout at
        100 ps/frame fit in memory.  None keeps the full graph.

        Returns stacked H trajectory [B, n_steps+1, L, dH].
        """
        cfg = self.cfg
        if seq_feature is None:
            seq_feature = H0
        S = None
        if cfg.use_bath:
            if S0 is not None:
                S = S0
            else:
                if H_hist is None:
                    dS = cfg.bath_dim
                    m0 = torch.zeros(H0.shape[0], H0.shape[1], dS, device=H0.device, dtype=H0.dtype)
                    chol0 = torch.eye(dS, device=H0.device, dtype=H0.dtype).expand(
                        H0.shape[0], H0.shape[1], dS, dS).contiguous()
                else:
                    m0, chol0 = self.initial_bath_state(
                        H_hist, dt_hist, temperature_k, residue_mask,
                        seq_feature=seq_feature)
                eps = torch.randn(m0.shape, device=H0.device, dtype=H0.dtype, generator=generator)
                S = m0 + torch.einsum("blij,blj->bli", chol0, eps)

        traj = [H0]
        bath_traj = [S] if S is not None else None
        H = H0
        for i in range(n_steps):
            H, S = self.sample_step(
                H, S, residue_mask, temperature_k, seq_feature, generator,
                create_graph=not detach, noise_factor=noise_factor, collective_mobility=collective_mobility,
            )
            # Truncated BPTT aimed at the scored horizon: the losses of interest
            # live at the END of the rollout, so we keep the last `grad_steps`
            # transitions differentiable and detach everything before them.
            # Forward values are untouched, so every horizon is still evaluated
            # exactly; autograd just stops accumulating the early trajectory.
            if (not detach and grad_steps is not None
                    and i < n_steps - grad_steps):
                H = H.detach()
                if S is not None:
                    S = S.detach()
            traj.append(H)
            if bath_traj is not None:
                bath_traj.append(S)
        out = torch.stack(traj, dim=1)
        out = out.detach() if detach else out
        if not return_bath:
            return out
        bath_out = torch.stack(bath_traj, dim=1) if bath_traj is not None else None
        if detach and bath_out is not None:
            bath_out = bath_out.detach()
        return out, bath_out
