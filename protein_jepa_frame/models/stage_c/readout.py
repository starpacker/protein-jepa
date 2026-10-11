"""Lightweight physical observable readout from latent dynamics Q.

M04 (CVPR Section 4.4): Converts Q:[B,L,8] to physically meaningful protein
observables WITHOUT full structure reconstruction. This is the P0-level
evaluation backbone for all E01-E06 experiments.

Architecture:
  - Pairwise MLP: (q_i, q_j, rbf(|i-j|)) → d̂_ij (CA-CA distance)
  - Per-residue MLP: (q_i, masked_mean(Q)) → RMSF
  - Contact: sigmoid(τ - d̂_ij) with threshold τ = 8Å
  - All models share the same Q input — no H, no structure decoder

Training: supervised on mdCATH training set, frozen Stage B Q as input.
Evaluation: same readout used for all baselines (B0-B5) for fair comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReadoutConfig:
    """Configuration for M04 physical observable readout."""
    q_dim: int = 8          # Q latent dimension per residue
    hidden_dim: int = 64    # hidden layer dimension
    depth: int = 2          # MLP depth
    num_rbf: int = 16       # sequence-distance RBF features
    rbf_max_dist: int = 50  # max sequence separation for RBF
    contact_threshold: float = 8.0  # Å threshold for contact definition
    rmsf_hidden_dim: int = 32


class PhysicalObservableReadout(nn.Module):
    """M04: Lightweight readout from Q → physical protein observables.

    Output channels:
      - ca_distances: [B, L, L] mean CA-CA distance predictions (Å)
      - contact_prob: [B, L, L] P(distance < threshold)
      - rmsf: [B, L] per-residue root-mean-square fluctuation (Å)
      - global_rg: [B] radius of gyration estimate (derived from distances)

    All outputs are computed from Q alone. No structure decoder, no H access.
    This enables fair comparison across all dynamics models (B0-B5).
    """

    def __init__(self, config: ReadoutConfig | None = None):
        super().__init__()
        self.cfg = config or ReadoutConfig()
        c = self.cfg

        # Sequence-distance RBF encoding: |i-j| → num_rbf features
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.0, c.rbf_max_dist, c.num_rbf),
            persistent=False,
        )
        self.rbf_width = float(c.rbf_max_dist / max(1, c.num_rbf - 1))
        if self.rbf_width < 1e-6:
            self.rbf_width = 1.0

        # ── Pairwise distance predictor ──
        # Input: q_i [D] + q_j [D] + rbf(|i-j|) [num_rbf] = 2D + num_rbf
        pair_input = 2 * c.q_dim + c.num_rbf
        pair_layers = []
        in_dim = pair_input
        for _ in range(c.depth):
            pair_layers.extend([nn.Linear(in_dim, c.hidden_dim), nn.SiLU()])
            in_dim = c.hidden_dim
        pair_layers.append(nn.Linear(in_dim, 1))  # single scalar distance
        self.pair_net = nn.Sequential(*pair_layers)

        # ── Per-residue RMSF predictor ──
        # Input: q_i [D] + G (masked_mean Q) [D] = 2D
        rmsf_input = 2 * c.q_dim
        self.rmsf_net = nn.Sequential(
            nn.Linear(rmsf_input, c.rmsf_hidden_dim), nn.SiLU(),
            nn.Linear(c.rmsf_hidden_dim, c.rmsf_hidden_dim), nn.SiLU(),
            nn.Linear(c.rmsf_hidden_dim, 1),
        )

    def _seq_rbf(self, L: int, device: torch.device) -> torch.Tensor:
        """Build [L, L, num_rbf] RBF encoding of |i-j|."""
        ids = torch.arange(L, device=device, dtype=torch.float32)
        dists = (ids.unsqueeze(0) - ids.unsqueeze(1)).abs()  # [L, L]
        # RBF: exp(-((d - c)/w)²)
        diffs = dists.unsqueeze(-1) - self.rbf_centers.to(device)
        return torch.exp(-(diffs / self.rbf_width) ** 2)  # [L, L, num_rbf]

    def forward(
        self,
        q: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convert latent Q to physical observables.

        Args:
            q: [B, L, D] per-residue dynamics variables
            residue_mask: [B, L] valid residues

        Returns:
            Dict with keys:
              ca_distances:  [B, L, L] predicted mean CA-CA distances (Å)
              contact_prob:  [B, L, L] P(contact) = σ(τ - d̂)
              rmsf:          [B, L] predicted RMSF (Å)
              rmsf_angstrom: [B, L] same, clamped to ≥ 0
              global_rg:     [B] approximate radius of gyration
        """
        B, L, D = q.shape
        device = q.device

        # ── Pairwise distances ──
        rbf = self._seq_rbf(L, device)  # [L, L, num_rbf]

        # Build all pairwise inputs
        qi = q.unsqueeze(2).expand(-1, -1, L, -1)  # [B, L, L, D]
        qj = q.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, L, D]
        rbf_b = rbf.unsqueeze(0).expand(B, -1, -1, -1)  # [B, L, L, num_rbf]

        pair_input = torch.cat([qi, qj, rbf_b], dim=-1)  # [B, L, L, 2D+num_rbf]
        raw_dist = self.pair_net(pair_input).squeeze(-1)  # [B, L, L]

        # Enforce positivity and symmetry
        ca_distances = F.softplus(raw_dist) + 0.01  # guaranteed > 0
        ca_distances = 0.5 * (ca_distances + ca_distances.transpose(-1, -2))

        # Diagonal = 0 (self-distance)
        diag_mask = 1.0 - torch.eye(L, device=device, dtype=q.dtype).unsqueeze(0)
        ca_distances = ca_distances * diag_mask

        # ── Contact probability ──
        contact_prob = torch.sigmoid(
            (self.cfg.contact_threshold - ca_distances) / 1.0
        )

        # ── RMSF (per-residue) ──
        # G = masked_mean(Q) — global condition
        if residue_mask is not None:
            m_f = residue_mask.unsqueeze(-1).float()
            G = (q * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)  # [B, D]
        else:
            G = q.mean(dim=1)  # [B, D]
        G_exp = G.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]

        rmsf_input = torch.cat([q, G_exp], dim=-1)  # [B, L, 2D]
        rmsf_raw = self.rmsf_net(rmsf_input).squeeze(-1)  # [B, L]
        rmsf = F.softplus(rmsf_raw) + 1e-3  # positive, minimum ~0.001

        # ── Apply mask ──
        if residue_mask is not None:
            m = residue_mask.float()
            m2 = m.unsqueeze(1) * m.unsqueeze(2)  # [B, L, L]
            ca_distances = ca_distances * m2
            contact_prob = contact_prob * m2
            rmsf = rmsf * m

        # ── Global radius of gyration (from pairwise distances) ──
        # Rg² = (1/2N²) Σ_{i,j} d²_{ij}
        if residue_mask is not None:
            n_valid = m.sum(dim=1, keepdim=True).clamp(min=1.0)
            d_sq = (ca_distances ** 2) * m2
            rg_sq = 0.5 * d_sq.sum(dim=(1, 2)) / (n_valid.squeeze(-1) ** 2)
        else:
            d_sq = ca_distances ** 2
            rg_sq = 0.5 * d_sq.sum(dim=(1, 2)) / (L * L)
        global_rg = torch.sqrt(rg_sq.clamp(min=1e-8))

        return {
            "ca_distances": ca_distances,
            "contact_prob": contact_prob,
            "rmsf": rmsf,
            "global_rg": global_rg,
        }


# ── Readout loss functions ──

def readout_distance_loss(
    pred_distances: torch.Tensor,
    target_distances: torch.Tensor,
    mask: torch.Tensor | None = None,
    clamp_max: float = 50.0,
) -> torch.Tensor:
    """MSE on CA-CA distances, masked, with outlier clamping.

    Args:
        pred_distances: [B, L, L] predicted distances
        target_distances: [B, L, L] ground-truth CA-CA distances
        mask: [B, L] residue mask (broadcast to L×L)
        clamp_max: clip target distances (avoid long-range noise dominance)

    Returns:
        scalar MSE loss per valid pair
    """
    target = target_distances.clamp(max=clamp_max)
    diff = pred_distances - target

    if mask is not None:
        m2 = mask.unsqueeze(1).float() * mask.unsqueeze(2).float()
        loss = (diff ** 2 * m2).sum() / m2.sum().clamp(min=1.0)
    else:
        loss = diff.pow(2).mean()
    return loss


def readout_contact_loss(
    pred_contacts: torch.Tensor,
    target_contacts: torch.Tensor,
    mask: torch.Tensor | None = None,
    pos_weight: float = 3.0,
) -> torch.Tensor:
    """Binary cross-entropy on contact prediction.

    Args:
        pred_contacts: [B, L, L] predicted contact probabilities (0-1)
        target_contacts: [B, L, L] ground-truth binary contacts (0/1)
        mask: [B, L] residue mask
        pos_weight: weight for positive class (contacts are sparse)

    Returns:
        scalar BCE loss per valid pair
    """
    B, L = pred_contacts.shape[:2]
    device = pred_contacts.device
    # Don't count self-contacts (diagonal) — apply to full tensor first
    off_diag = 1.0 - torch.eye(L, device=device).unsqueeze(0)  # [1, L, L]
    weight_mask = off_diag > 0.5

    if mask is not None:
        m2 = mask.unsqueeze(1).float() * mask.unsqueeze(2).float()  # [B, L, L]
        valid_mask = (m2 > 0.5) & weight_mask
        pred = pred_contacts[valid_mask]
        target = target_contacts[valid_mask]
    else:
        pred = pred_contacts[weight_mask.expand(B, -1, -1)]
        target = target_contacts[weight_mask.expand(B, -1, -1)]

    if pred.numel() == 0:
        return torch.tensor(0.0, device=device)

    loss = F.binary_cross_entropy_with_logits(
        torch.logit(pred.clamp(1e-7, 1 - 1e-7)),
        target,
        pos_weight=torch.tensor(pos_weight, device=device),
    )
    return loss


def readout_rmsf_loss(
    pred_rmsf: torch.Tensor,
    target_rmsf: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE on per-residue RMSF.

    Args:
        pred_rmsf: [B, L] predicted RMSF
        target_rmsf: [B, L] ground-truth RMSF (Å)
        mask: [B, L]

    Returns:
        scalar MSE
    """
    diff = pred_rmsf - target_rmsf
    if mask is not None:
        m = mask.float()
        loss = (diff ** 2 * m).sum() / m.sum().clamp(min=1.0)
    else:
        loss = diff.pow(2).mean()
    return loss


def readout_total_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    mask: torch.Tensor | None = None,
    distance_weight: float = 1.0,
    contact_weight: float = 0.5,
    rmsf_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined readout training loss.

    Args:
        outputs: {ca_distances, contact_prob, rmsf} from PhysicalObservableReadout
        targets: {ca_distances, contacts, rmsf} ground truth
        mask: [B, L]
        *_weight: per-task loss weight

    Returns:
        total_loss, dict of per-task losses
    """
    components = {}
    total = torch.tensor(0.0, device=outputs["ca_distances"].device)

    if "ca_distances" in targets and "ca_distances" in outputs:
        l_d = readout_distance_loss(outputs["ca_distances"], targets["ca_distances"], mask)
        components["distance"] = l_d
        total = total + distance_weight * l_d

    if "contacts" in targets and "contact_prob" in outputs:
        l_c = readout_contact_loss(outputs["contact_prob"], targets["contacts"], mask)
        components["contact"] = l_c
        total = total + contact_weight * l_c

    if "rmsf" in targets and "rmsf" in outputs:
        l_r = readout_rmsf_loss(outputs["rmsf"], targets["rmsf"], mask)
        components["rmsf"] = l_r
        total = total + rmsf_weight * l_r

    if not components:
        total = outputs["ca_distances"].sum() * 0.0  # zero grad

    return total, components


def compute_target_contacts(
    ca_distances: torch.Tensor,
    threshold: float = 8.0,
) -> torch.Tensor:
    """Convert CA distances to binary contacts.

    Args:
        ca_distances: [B, L, L] CA-CA distances in Å
        threshold: contact threshold in Å

    Returns:
        [B, L, L] binary (0/1) contacts
    """
    return (ca_distances < threshold).float()


def compute_target_rmsf(
    ca_trajectory: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-residue RMSF from CA trajectory.

    Args:
        ca_trajectory: [B, T, L, 3] CA positions over time
        mask: [B, L]

    Returns:
        [B, L] RMSF in Å
    """
    # Mean position per residue
    if mask is not None:
        m_f = mask.unsqueeze(0).unsqueeze(-1).float()
        mean_pos = (ca_trajectory * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)
        sq_dev = ((ca_trajectory - mean_pos.unsqueeze(1)) ** 2).sum(dim=-1)
        msf = (sq_dev * m_f.squeeze(-1)).sum(dim=1) / m_f.squeeze(-1).sum(dim=1).clamp(min=1.0)
        rmsf = torch.sqrt(msf.clamp(min=1e-8))
        rmsf = rmsf * mask.float()
    else:
        mean_pos = ca_trajectory.mean(dim=1)
        sq_dev = ((ca_trajectory - mean_pos.unsqueeze(1)) ** 2).sum(dim=-1)
        msf = sq_dev.mean(dim=1)
        rmsf = torch.sqrt(msf.clamp(min=1e-8))
    return rmsf


__all__ = [
    "ReadoutConfig",
    "PhysicalObservableReadout",
    "readout_distance_loss",
    "readout_contact_loss",
    "readout_rmsf_loss",
    "readout_total_loss",
    "compute_target_contacts",
    "compute_target_rmsf",
]
