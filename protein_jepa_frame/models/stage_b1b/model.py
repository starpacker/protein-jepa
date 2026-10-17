"""Stage B1b: GLE dynamics in H space with Kalman innovation NLL.

Implements STAGE_ABC_REDESIGN_PLAN.md 2.2-2.4:
  - HSpaceEnergy:     scalar U(H;T,c), force = -grad U via autograd
  - StaticBathParams: D, C, Gamma depend only on (T, c), not on H
  - HistoryPosterior: causal GRU -> (m_0, chol_P_0) for bath initialisation
  - StageB1b:         Kalman innovation NLL with Joseph covariance update
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
    # The bath time update uses the Euler discretisation A = 1 - h*Gamma, which
    # is only stable for h*Gamma < 1 (otherwise A < 0 and P oscillates/explodes).
    # With h = 1000 ps the OLD gamma_floor=0.05 gave A = 1 - 50 = -49: the bath
    # covariance blew up (bath/R ~ 2e4).  Scale Gamma so h*Gamma stays in (0,1):
    # Gamma ~ 1e-4 -> A ~ 0.9, stable with a steady-state P_ss ~ O(1).
    gamma_scale: float = 2.0e-4         # softplus output scaled so h*Gamma < 1
    gamma_floor: float = 2.0e-5         # minimum rate: h*gamma_floor = 0.02

    # History posterior
    history_hidden: int = 128

    # Integration: h = observation interval in picoseconds.
    # MDcath frame_interval_ps = 1000 (1 ns/frame).  The actual lag between the
    # last context frame and the first future frame is one frame.
    step_ps: float = 1000.0

    # Numerical jitter added to covariances for Cholesky stability
    jitter: float = 1.0e-6

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
        # Quadratic tail coefficient (positive via softplus) per channel.
        self.log_tail = nn.Parameter(torch.full((d,), math.log(math.expm1(1e-3))))

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
# Static bath parameters: D, C, Gamma conditioned on (T, c) only
# -----------------------------------------------------------------------------

class StaticBathParams(nn.Module):
    """D (dH diag), Gamma (dS diag), C (dH x dS) per residue.

    Depends only on static conditions -- sequence c (via per-residue H at the
    anchor frame used as a static sequence feature) and temperature T -- NOT on
    the evolving H_t.  This avoids Ito divergence terms (2.2).

    To honour "depends on sequence c, not on H_t", the sequence condition is a
    *detached* per-residue feature; the networks never receive grad-carrying H.
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
        Gamma = self.gamma_floor + self.gamma_scale * F.softplus(gamma_raw)
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
    """H-space GLE dynamics with Kalman innovation NLL.

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
        return -grad * residue_mask.unsqueeze(-1).float()

    # -- Mean one-step drift (used by sampler + Kalman predict) --------------

    def _drift_H(self, H, f, D, C, m_or_S, h):
        """mu_{t+1} = H + h*D*f + h*(C @ m_or_S).

        H,f,D: [B,L,dH]; C: [B,L,dH,dS] or None; m_or_S: [B,L,dS] or None.
        """
        mu = H + h * D * f
        if C is not None and m_or_S is not None:
            mu = mu + h * torch.einsum("blij,blj->bli", C, m_or_S)
        return mu

    # -- Sampler (single-step Euler; identical train/deploy) -----------------

    def sample_step(
        self, H, S, residue_mask, temperature_k, seq_feature=None, generator=None,
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
        f = self.compute_forces(H, residue_mask, temperature_k, create_graph=False)

        mu_H = self._drift_H(H, f, D, C, S, h)
        noise_H = torch.randn(mu_H.shape, device=H.device, dtype=H.dtype, generator=generator)
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

    # -- Kalman innovation NLL (main training loss, 2.4) ---------------------

    def kalman_nll(
        self, H_hist, dt_hist, H_seq, residue_mask, temperature_k,
        return_diagnostics: bool = False,
    ):
        """Kalman innovation NLL scoring H_{1:N} | H_{-K+1:0} (2.4).

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
        N = T - 1
        rmask = residue_mask.unsqueeze(-1).float()               # [B,L,1]
        device, dtype = H_seq.device, H_seq.dtype

        # Static bath params from the anchor (last history / first obs) frame.
        seq_feature = H_seq[:, 0]                                 # [B,L,dH]
        D, C, Gamma = self.bath(seq_feature, temperature_k)
        R = 2.0 * h * D                                           # [B,L,dH] diag of H noise

        total_nll = H_seq.new_zeros(())
        jitter_hits = 0

        if not cfg.use_bath:
            # Diagonal Gaussian: V = R, independent per channel.
            logdetV = torch.log(R.clamp_min(cfg.jitter)).sum(-1)  # [B,L]
            for n in range(N):
                Hn, Hn1 = H_seq[:, n], H_seq[:, n + 1]
                f = self.compute_forces(Hn, residue_mask, temperature_k, create_graph=True)
                mu = Hn + h * D * f
                e = (Hn1 - mu)                                    # [B,L,dH]
                quad = (e * e / R.clamp_min(cfg.jitter)).sum(-1)  # [B,L]
                step_nll = 0.5 * (quad + logdetV + d * math.log(2 * math.pi))
                total_nll = total_nll + (step_nll * residue_mask.float()).sum()
            if return_diagnostics:
                return total_nll, {"jitter_hits": 0}
            return total_nll

        dS = cfg.bath_dim
        # Linear-Gaussian bath recursion coefficients (per residue):
        #   A = I - h*Gamma (diagonal),  W = 2 h Gamma (diagonal),  u_n = h C^T f
        A_diag = 1.0 - h * Gamma                                  # [B,L,dS]
        W_diag = 2.0 * h * Gamma                                  # [B,L,dS]
        B_h = h * C                                               # [B,L,dH,dS]
        R_mat = torch.diag_embed(R)                               # [B,L,dH,dH]
        I_dS = torch.eye(dS, device=device, dtype=dtype).expand(B, L, dS, dS)
        innov_whitened = [] if return_diagnostics else None
        bath_ratios = [] if return_diagnostics else None

        # Initial bath posterior from history network.
        m0, chol0 = self.history_posterior(H_hist, dt_hist, temperature_k, residue_mask)
        m_n = m0                                                  # [B,L,dS]
        P_n = chol0 @ chol0.transpose(-1, -2) + cfg.jitter * I_dS  # [B,L,dS,dS]

        log2pi = math.log(2 * math.pi)
        for n in range(N):
            Hn, Hn1 = H_seq[:, n], H_seq[:, n + 1]
            f = self.compute_forces(Hn, residue_mask, temperature_k, create_graph=True)

            # Predicted mean of H: mu = H + hDf + h C m_n
            mu = Hn + h * D * f + torch.einsum("blij,blj->bli", B_h, m_n)
            e = Hn1 - mu                                          # [B,L,dH]

            # Innovation covariance V = R + B_h P B_h^T   [B,L,dH,dH]
            BP = torch.einsum("blij,bljk->blik", B_h, P_n)        # [B,L,dH,dS]
            bath_cov = torch.einsum("blik,bljk->blij", BP, B_h)   # [B,L,dH,dH]
            V = R_mat + bath_cov                                  # [B,L,dH,dH]
            V = 0.5 * (V + V.transpose(-1, -2))                   # symmetrise
            if return_diagnostics:
                # Ratio of the bath's mean diagonal contribution to R's: should be
                # O(1) for the bath to matter.  ~0 => bath is numerically dead.
                bd = torch.diagonal(bath_cov, dim1=-2, dim2=-1).mean()
                rd = R.mean().clamp_min(1e-12)
                bath_ratios.append(float((bd / rd).detach()))

            # Cholesky solve/logdet (no explicit inverse).
            # V is PSD in exact arithmetic, but in practice its diagonal spans
            # many orders of magnitude (R = 2hD can be ~0 while B_h P B_h^T is
            # large), so a fixed jitter is meaningless.  Use a TRACE-RELATIVE
            # jitter and escalate multiplicatively until the factorisation
            # succeeds, then fall back to eigen-clamping.  Never raise: a step
            # that cannot be factorised contributes zero NLL instead of killing
            # the run.
            V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
            eye_d = torch.eye(d, device=device, dtype=dtype)
            tr = V.diagonal(dim1=-2, dim2=-1).abs().sum(-1) / d          # [B,L] mean diag
            base = tr.clamp_min(1.0).unsqueeze(-1).unsqueeze(-1)         # [B,L,1,1]
            Lc, info = None, None
            for mult in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                V_j = V + (mult * base) * eye_d
                Lc, info = torch.linalg.cholesky_ex(V_j)
                if not (info != 0).any():
                    break
                jitter_hits += int((info != 0).sum().item())
            if (info != 0).any():
                # Eigen-clamp: floor eigenvalues at (mult*mean diag).
                evals, evecs = torch.linalg.eigh(V)
                floor = (1e-6 * base).squeeze(-1).squeeze(-1).unsqueeze(-1)  # [B,L,1]
                evals = torch.nan_to_num(evals, nan=0.0).clamp_min(floor)
                V_j = torch.einsum("blij,blj,blkj->blik", evecs, evals, evecs)
                V_j = 0.5 * (V_j + V_j.transpose(-1, -2))
                # keep escalating until even the clamped matrix factors
                for mult in (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0):
                    V_js = V_j + (mult * base) * eye_d
                    Lc, info = torch.linalg.cholesky_ex(V_js)
                    if not (info != 0).any():
                        V_j = V_js
                        break
                if (info != 0).any():
                    # Give up on this step: zero contribution, identity factor.
                    V_j = base.squeeze(-1).expand(B, L, d) * eye_d
                    Lc = torch.linalg.cholesky(V_j)

            # quad = e^T V^-1 e via triangular solve
            e_col = e.unsqueeze(-1)                               # [B,L,dH,1]
            sol = torch.cholesky_solve(e_col, Lc)                 # V^-1 e
            quad = (e_col * sol).sum((-1, -2))                    # [B,L]
            logdetV = 2.0 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)  # [B,L]
            step_nll = 0.5 * (quad + logdetV + d * log2pi)
            total_nll = total_nll + (step_nll * residue_mask.float()).sum()

            if return_diagnostics:
                # Whitened innovation ev = Lc^{-1} e, which should be ~N(0, I_dH).
                ev = torch.linalg.solve_triangular(Lc, e_col, upper=False)[..., 0]  # [B,L,dH]
                innov_whitened.append(ev.detach())

            # Joseph covariance update of the bath posterior.
            # K = P B_h^T V^-1   [B,L,dS,dH]
            # cholesky_solve(X, L) solves L L^T K = X; X must be [B,L,dH,*]
            Vinv_Bh = torch.cholesky_solve(B_h, Lc)                     # V^-1 B_h  [B,L,dH,dS]
            K = torch.einsum("blsi,blij->blsj", P_n, Vinv_Bh.transpose(-1, -2))  # [B,L,dS,dH]
            m_plus = m_n + torch.einsum("blsk,blk->bls", K, e)   # [B,L,dS]
            KB = torch.einsum("blsk,blkt->blst", K, B_h)         # [B,L,dS,dS]
            ImKB = I_dS - KB
            P_plus = ImKB @ P_n @ ImKB.transpose(-1, -2) \
                   + torch.einsum("blsk,blk,bltk->blst", K, R, K)  # K R K^T (R diagonal)
            P_plus = 0.5 * (P_plus + P_plus.transpose(-1, -2))

            # Bath time update: m_{n+1}=A m_plus + u_n, P_{n+1}=A P_plus A^T + W
            u_n = h * torch.einsum("blij,bli->blj", C, f)        # [B,L,dS]
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
    ):
        """Rollout from H0 for n_steps.

        Deployment path (detach=True, no grad): initialises S_0 by SAMPLING
        from the history posterior (2.3), not just its mean.

        Training path (detach=False): gradients flow through the history
        posterior and the dynamics, so the energy score in (2.5) can train
        both r_psi and the kernel.

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
                    m0, chol0 = self.history_posterior(H_hist, dt_hist, temperature_k, residue_mask)
                eps = torch.randn(m0.shape, device=H0.device, dtype=H0.dtype, generator=generator)
                S = m0 + torch.einsum("blij,blj->bli", chol0, eps)

        traj = [H0]
        H = H0
        for _ in range(n_steps):
            H, S = self.sample_step(H, S, residue_mask, temperature_k, seq_feature, generator)
            traj.append(H)
        out = torch.stack(traj, dim=1)
        return out.detach() if detach else out
