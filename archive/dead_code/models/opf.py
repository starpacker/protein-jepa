"""Orthogonal Predictive Factorization (OPF) for the Stage-A JEPA target.

Adapted from JEPA-Anything (arXiv 2609.20800). OPF partitions the paired delta target
space R^d into K mutually orthogonal r-dimensional factors (K*r = d), predicts each
factor with a dedicated (or shared-trunk + branch) head, and synthesizes a complete
latent state via the Moore-Penrose pseudoinverse of the analysis map P^T.

    factor:     z^(k)_t = P_k^T @ sg(z_t),            P_k in R^(d x r)
    predict:    z_hat^(k)_t = q_k(context, s_t)
    synthesis:  z_hat = (P^T)^+ @ [z_hat^(1), ..., z_hat^(K)]

Loss terms (mirror Sec 2.2-2.3):
    L_pred : full regression of each predicted factor to its target.
    L_orth : within-factor `||P_k^T P_k - I_r||_F^2` + cross-factor `sum ||P_i^T P_j||_F^2`.
    L_fac  : per-factor coordinate activity floor `max(0, gamma_fac - sigma(coord))`.
    L_enc  : anti-collapse std floor on the online context state.

`num_factors == 1` reduces to the standard single-head predictor (matched baseline).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.time_features import (
    LogTimeEmbedding,
    ThermodynamicConditionEmbedding,
)


class _OrthogonalFactorHead(nn.Module):
    """Dedicated residual head predicting one r-dimensional factor from a shared input."""

    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int = 2,
        dropout: float = 0.1,
        zero_init: bool = True,
    ):
        super().__init__()
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, depth - 1)):
            layers.extend([nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout)])
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(width, input_dim)
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.trunk(x))


class OPFPredictor(nn.Module):
    """K orthogonal factor predictors sharing a common physical-time conditioning input.

    Parameters
    ----------
    latent_dim : int
        Full target dimension d. Each factor width is r = d // K.
    num_factors : int
        Number of orthogonal factors K. Must divide `latent_dim`; use 1 for baseline.
    shared_trunk : bool
        If True, all factors share one conditioning trunk and differ by branch head
        (parameter-matched against an unconstrained multi-head control). If False,
        each factor gets an independent head (the paper's default).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_factors: int = 4,
        shared_trunk: bool = True,
        depth: int = 2,
        dropout: float = 0.1,
        thermodynamic_conditioning: bool = True,
        activity_gamma: float = 0.5,
        encoder_activity_gamma: float = 0.5,
    ):
        super().__init__()
        if latent_dim <= 0 or num_factors <= 0:
            raise ValueError("OPF latent dim and factor count must be positive")
        if latent_dim % num_factors != 0:
            raise ValueError("OPF factor count must divide latent_dim")

        self.latent_dim = int(latent_dim)
        self.num_factors = int(num_factors)
        self.shared_trunk = bool(shared_trunk)
        self.factor_dim = int(latent_dim) // int(num_factors)
        self.activity_gamma = float(activity_gamma)
        self.encoder_activity_gamma = float(encoder_activity_gamma)

        # Learned projectors P_k in R^(d x r) initialized as an IDENTITY PARTITION
        # (block identity: P_k's non-zero rows are exactly one coordinate block).
        # The full concatenated P = [P_1..P_K] is therefore already the identity in
        # R^(d x d): globally orthonormal (kappa2 = 1) with exactly-zero cross-factor
        # overlap. This is the official JEPA-Anything default init; the Gram /
        # prediction / activity losses then rotate the blocks while soft-Gram keeps
        # them (nearly) orthonormal.
        self.projectors = nn.ParameterList()
        for k in range(self.num_factors):
            p = torch.zeros(self.latent_dim, self.factor_dim)
            p[k * self.factor_dim : (k + 1) * self.factor_dim, :] = torch.eye(
                self.factor_dim
            )
            self.projectors.append(nn.Parameter(p))

        # Shared physical-time conditioning → factor logits.
        self.time_embed = LogTimeEmbedding(self.latent_dim)
        self.condition_embed = (
            ThermodynamicConditionEmbedding(self.latent_dim)
            if thermodynamic_conditioning
            else None
        )

        # Factor predictors.
        if self.shared_trunk:
            # IMPORTANT (dead-saddle fix): the shared trunk must output NONZERO
            # features. If the trunk output layer were zero-initialized, the branch
            # heads would receive all-zero input and dL/d(head) = input^T = 0, so the
            # heads could never escape a trivial zero prediction. Match the standard
            # PairedDeltaPredictor init contract: the *final output layer* (branch
            # heads) starts at zero (so initial delta-prediction is identity/persistence-
            # preserving), but the trunk features give a nonzero gradient into the heads.
            self.shared_trunk_net = _OrthogonalFactorHead(
                input_dim=self.latent_dim,
                width=self.latent_dim * 2,
                depth=depth,
                dropout=dropout,
                zero_init=False,  # trunk outputs must be nonzero for head-gradient flow
            )
            self.branch_heads = nn.ModuleList(
                [
                    nn.Linear(self.factor_dim, self.factor_dim, bias=False)
                    for _ in range(self.num_factors)
                ]
            )
            for h in self.branch_heads:
                nn.init.zeros_(h.weight)
        else:
            # Independent heads: each head's output layer is zero-init (delta-prediction
            # starts at zero) but the internal trunk layers are random-init so gradient
            # flows through the head (dL/d(head) = input^T * dL/d(output)).
            self.factor_heads = nn.ModuleList(
                [
                    _OrthogonalFactorHead(
                        input_dim=self.factor_dim,
                        width=self.latent_dim * 2,
                        depth=depth,
                        dropout=dropout,
                        zero_init=True,   # output layer zero (persistence init)
                    )
                    for _ in range(self.num_factors)
                ]
            )

        # Project inputs into the shared trunk (identity projection of the raw state).
        self._eps = 1.0e-6

    # ------------------------------------------------------------------ #
    # Geometry / synthesis helpers
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def orthonormalize_projectors(self) -> None:
        """Project the FULL concatenated analysis map onto the Stiefel manifold.

        This QR-orthonormalizes P = [P_1,...,P_K] as a single d x d matrix, exactly
        matching the official JEPA-Anything `orthonormalize_basis` (which flattens
        the (K, r, d) analysis rows and QRs the whole system). Unlike per-factor QR,
        this enforces BOTH within-factor row orthonormality AND cross-factor
        orthogonality, so the concatenated analysis map is a global orthogonal matrix
        (kappa2 = 1, Prop-1). Run after each optimizer step to keep the factor
        interface unit-conditioned regardless of optimizer scale.
        """
        flat = torch.cat([p for p in self.projectors], dim=-1)  # (d, d)
        q, r = torch.linalg.qr(flat)
        # Canonicalize QR sign ambiguity: orientations by analyzer (like official).
        signs = torch.sign(torch.diagonal(r)).clamp_min(0.0)
        signs[signs == 0.0] = 1.0
        q = q * signs.unsqueeze(0)
        for k in range(self.num_factors):
            sl = slice(k * self.factor_dim, (k + 1) * self.factor_dim)
            self.projectors[k].copy_(q[:, sl])

    def concatenated_projector(self) -> torch.Tensor:
        """Return P = [P_1, ..., P_K] in R^(d x d)."""
        return torch.cat([p for p in self.projectors], dim=-1)

    @torch.no_grad()
    def projector_condition_number(self) -> torch.Tensor:
        """κ₂(P) = σ_max / σ_min of the concatenated analysis map."""
        P = self.concatenated_projector()
        sv = torch.linalg.svdvals(P)
        return (sv.max() / sv.min().clamp(min=self._eps)).detach()

    @torch.no_grad()
    def projector_min_singular_value(self) -> torch.Tensor:
        sv = torch.linalg.svdvals(self.concatenated_projector())
        return sv.min().detach()

    @torch.no_grad()
    def cross_factor_overlap(self) -> torch.Tensor:
        """Mean squared cross-subspace overlap `(2/(K(K-1))) sum_{i<j} ||P_i^T P_j||_F^2`."""
        overlaps = []
        for i in range(self.num_factors):
            for j in range(i + 1, self.num_factors):
                gram = self.projectors[i].T @ self.projectors[j]
                overlaps.append(gram.square().mean())
        if not overlaps:
            return self.projectors[0].new_tensor(0.0)
        return torch.stack(overlaps).mean().detach()

    def _synthesize(self, factor_preds: list[torch.Tensor]) -> torch.Tensor:
        """State synthesis z_hat = (P^T)^+ u where u = concat of factor preds.

        Uses the pseudoinverse of P^T (Moore-Penrose) so it remains well-defined
        even while P is not yet perfectly orthogonal (Corollary 1 setting).
        """
        P = self.concatenated_projector()  # [d, d]
        u = torch.cat(factor_preds, dim=-1)  # [B, d]
        # (P^T)^+ = P (P^T P)^-1 when P is square & full column rank. This equals
        # the transpose synthesis P u in the orthogonal limit (P^T P = I) but stays
        # well-defined before orthogonality converges (Corollary 1 setting).
        PTP = P.T @ P  # [d, d]
        eye = torch.eye(PTP.shape[0], device=PTP.device, dtype=PTP.dtype)
        try:
            inv = torch.linalg.inv(PTP + self._eps * eye)
        except RuntimeError:
            inv = torch.linalg.pinv(PTP)
        return (P @ inv @ u.T).T

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        endpoint_state: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor | None = None,
        dataset_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict each factor and synthesize a full latent state.

        Parameters
        ----------
        endpoint_state : [B, d]
            Online context transition state (used for conditioning and encoding activity).
        horizon_time_ps : [B] physical time horizon.

        Returns
        -------
        dict with keys:
          - `state_prediction` : [B,d] synthesized full latent delta.
          - `factor_preds`     : list of K [B, r] per-factor predictions.
          - `factor_cond`      : [B,d] shared conditioning (context + time) for activity stats.
        """
        if endpoint_state.dim() != 2:
            raise ValueError("OPF endpoint_state must be [B,d]")
        B = endpoint_state.shape[0]

        condition = self.time_embed(horizon_time_ps).to(dtype=endpoint_state.dtype)
        if self.condition_embed is not None:
            condition = condition + self.condition_embed(
                temperature_k, dataset_index, B, endpoint_state.device, endpoint_state.dtype
            )
        # Conditioning = context state + time, as in the single-head predictor.
        cond_input = endpoint_state + condition

        factor_preds = []
        if self.shared_trunk:
            shared = self.shared_trunk_net(cond_input)  # [B, d]
            shared = shared.view(B, self.num_factors, self.factor_dim)
            for k in range(self.num_factors):
                pred_k = self.branch_heads[k](
                    shared[:, k, :]
                )
                factor_preds.append(pred_k)
        else:
            inp_k = cond_input.view(B, self.num_factors, self.factor_dim)
            for k in range(self.num_factors):
                factor_preds.append(self.factor_heads[k](inp_k[:, k, :]))

        state_prediction = self._synthesize(factor_preds)
        return {
            "state_prediction": state_prediction,
            "factor_preds": factor_preds,
            "factor_cond": cond_input,
        }


def opf_terms(outputs: dict, config, loss_type: str = "smooth_l1") -> dict[str, torch.Tensor]:
    """Compute the additive OPF loss terms given forward_paired outputs.

    Expects `outputs` to contain (from forward_paired when OPF is enabled):
      - `opf_factor_preds` : list of K [B, r] factor predictions.
      - `opf_factor_targets` : list of K [B, r] projected target factors P_k^T delta_target.
      - `opf_projectors` : the OPFPredictor module (for L_orth, activity).
      - `opf_factor_cond` : [B, d] conditioning used for activity stats.
      - `online_context_state` : [B, d] online context (for L_enc).
    """
    preds = outputs["opf_factor_preds"]
    targets = outputs["opf_factor_targets"]
    opf: OPFPredictor = outputs["opf_projectors"]
    factor_cond = outputs["opf_factor_cond"]
    online_context = outputs["online_context_state"]

    zero = targets[0].new_tensor(0.0)

    # L_pred: mean squared error over all factors.
    # NOTE: the target latent was stopped in forward_paired (sg(z_t)); the
    # projector P_k stays in the graph so L_pred shapes P_k toward predictable
    # factor coordinates (paper Eq-3, "predictability drives factorization").
    mse_sum = zero
    n = 0
    for p, t in zip(preds, targets):
        mse_sum = mse_sum + F.mse_loss(p, t)
        n += 1
    pred_loss = mse_sum / max(n, 1)

    # L_orth: within-factor orthonormality + cross-factor orthogonality.
    within_sum = zero
    for k in range(opf.num_factors):
        gram = opf.projectors[k].T @ opf.projectors[k]
        within_sum = within_sum + (gram - torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )).square().sum() / (gram.shape[0] * gram.shape[1])
    cross_sum = zero
    for i in range(opf.num_factors):
        for j in range(i + 1, opf.num_factors):
            gram = opf.projectors[i].T @ opf.projectors[j]
            cross_sum = cross_sum + gram.square().mean()
    orth_loss = within_sum / max(opf.num_factors, 1) + cross_sum / max(opf.num_factors, 1)

    # L_fac: keep every projected target coordinate active across the mini-batch.
    # Population variance with sqrt(Var + eps), per-coordinate hinge, mean over all
    # Kr coordinates -- exactly the official jepa_anything_core.factor_activity_loss.
    # The target latent is stopped (from forward_paired), but the projector P_k is
    # in the graph, so this loss shapes the projectors (official Eq-9).
    factor_activity_loss = zero
    if targets and targets[0].shape[0] >= 2:
        eps = 1e-6
        centered = torch.stack(targets, dim=-2)  # (B, K, r)
        centered = centered - centered.mean(dim=0, keepdim=True)
        var = centered.square().mean(dim=0)  # population variance
        sigma = torch.sqrt(var + eps)
        factor_activity_loss = F.relu(opf.activity_gamma - sigma).mean()

    # L_enc: anti-collapse population std floor on the online context state.
    # (official jepa_anything_core.encoder_variance_loss)
    if online_context is not None and online_context.shape[0] >= 2:
        eps = 1e-6
        centered = online_context - online_context.mean(dim=0, keepdim=True)
        var = centered.square().mean(dim=0)
        sigma_enc = torch.sqrt(var + eps)
        encoder_activity_loss = F.relu(opf.encoder_activity_gamma - sigma_enc).mean()
    else:
        encoder_activity_loss = zero

    return {
        "opf_pred_loss": pred_loss,
        "opf_orth_loss": orth_loss,
        "opf_factor_activity_loss": factor_activity_loss,
        "opf_encoder_activity_loss": encoder_activity_loss,
        "opf_pairs": torch.tensor(len(targets), device=targets[0].device),
    }
