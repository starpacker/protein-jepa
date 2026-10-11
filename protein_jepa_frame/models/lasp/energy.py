"""Scalar potentials and energy functions (conservative forces).

Contains:
  - MeanForcePotential: global scalar PMF A(q)
  - ResidueMessagePassingPotential: per-residue GNN potential U(q_i)
  - ReversiblePotential: Helmholtz potential H(q) for divergence-free drift
  - SkewSymmetricOperator: low-rank skew-symmetric J = P Q^T - Q P^T
  - UnifiedEnergyGraph: hierarchical energy A(Q,G)
  - Sliced score matching / force direction / stationary losses
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.lasp.utils import _as_batch_temperature
from protein_jepa_frame.models.lasp.objectives import kde_mean_force


class MeanForcePotential(nn.Module):
    """A scalar dimensionless PMF whose force is always conservative.

    Temperature is an explicit conditioning variable. The feature is the
    dimensionless log-temperature ratio, so a multi-temperature dataset does
    not force all equilibrium PMFs to share one unconstrained surface.
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: int = 0,
        hidden_dim: int = 128,
        hidden_depth: int = 2,
        reference_temperature_k: float = 348.0,
    ) -> None:
        super().__init__()
        if state_dim < 1 or condition_dim < 0 or hidden_dim < 4 or reference_temperature_k <= 0.0:
            raise ValueError("invalid LSP-WM potential dimensions")
        if hidden_depth < 1:
            raise ValueError("hidden_depth must be at least 1")
        self.state_dim = int(state_dim)
        self.condition_dim = int(condition_dim)
        self.reference_temperature_k = float(reference_temperature_k)
        input_dim = self.state_dim + self.condition_dim + 1
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(hidden_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.normal_(self.network[-1].bias, std=0.01)

    def _condition(self, q: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.condition_dim == 0:
            return q.new_zeros(*q.shape[:-1], 0)
        if condition is None:
            return q.new_zeros(*q.shape[:-1], self.condition_dim)
        if condition.shape != (*q.shape[:-1], self.condition_dim):
            raise ValueError("condition shape does not match the potential")
        return condition.to(dtype=q.dtype, device=q.device)

    def forward(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if q.shape[-1] != self.state_dim:
            raise ValueError("q has the wrong state dimension")
        temperature = _as_batch_temperature(
            temperature_k, q.shape[0], q, self.reference_temperature_k,
        )
        temperature_feature = torch.log(
            temperature / self.reference_temperature_k
        ).reshape(*q.shape[:-1], 1)
        return self.network(
            torch.cat([q, self._condition(q, condition), temperature_feature], dim=-1)
        ).squeeze(-1)

    def mean_force(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | float | None = None,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return ``-grad A`` in the same dimensionless units as the PMF."""
        with torch.enable_grad():
            q_leaf = q if q.requires_grad else q.detach().requires_grad_(True)
            energy = self(q_leaf, condition, temperature_k)
            gradient = torch.autograd.grad(
                energy.sum(), q_leaf,
                create_graph=create_graph, retain_graph=create_graph,
            )[0]
        return -gradient


class ResidueMessagePassingPotential(nn.Module):
    """Per-residue scalar potential with distance-based message passing.

    Each residue has its own CV state q_i ∈ R^{cv_dim}. The total potential
    energy is a sum of per-residue contributions that depend on both the
    residue's own state AND messages from neighbouring residues weighted by
    Cα distances. The resulting force ∇_iU is conservative by construction
    (computed via autograd).

    This replaces the global MeanForcePotential when per-residue dynamics
    are enabled, solving the globality/permutation-invariance problems.
    """

    def __init__(
        self,
        cv_dim: int = 8,
        condition_dim: int = 64,
        hidden_dim: int = 64,
        depth: int = 2,
        num_rbf: int = 8,
        rbf_max_dist: float = 30.0,
        reference_temperature_k: float = 348.0,
        temperature_feature: bool = True,
    ) -> None:
        super().__init__()
        if cv_dim < 1 or depth < 1 or hidden_dim < 16:
            raise ValueError("invalid residue potential dimensions")
        self.cv_dim = int(cv_dim)
        self.condition_dim = int(condition_dim)
        self.depth = int(depth)
        self.num_rbf = int(num_rbf)
        self.reference_temperature_k = float(reference_temperature_k)
        self.temperature_feature = bool(temperature_feature)
        self.register_buffer(
            "rbf_centres",
            torch.linspace(1.0, rbf_max_dist, num_rbf),
            persistent=False,
        )
        self.rbf_width = float(rbf_max_dist / num_rbf)
        edge_in = 2 * cv_dim + num_rbf + condition_dim
        self.edge_net = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        temperature_dim = 1 if self.temperature_feature else 0
        self.node_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cv_dim + temperature_dim + hidden_dim + condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, cv_dim),
            )
            for _ in range(depth)
        ])
        self.energy_head = nn.Sequential(
            nn.Linear(cv_dim + temperature_dim + condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for net in self.node_nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        nn.init.normal_(self.energy_head[-1].weight, std=0.01)
        nn.init.normal_(self.energy_head[-1].bias, std=0.01)

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -((distances.unsqueeze(-1) - self.rbf_centres) / self.rbf_width) ** 2
        )

    def _condition_broadcast(
        self, q: torch.Tensor, condition: torch.Tensor | None
    ) -> torch.Tensor:
        """Return per-residue condition of shape [B, L, cond_dim] from [B, cond_dim]."""
        B, L = q.shape[0], q.shape[1]
        if self.condition_dim == 0:
            return q.new_zeros(B, L, 0)
        if condition is None:
            return q.new_zeros(B, L, self.condition_dim)
        cond_t = condition.to(dtype=q.dtype, device=q.device)
        if cond_t.dim() == 2 and cond_t.shape[1] == self.condition_dim:
            # Per-protein: [B, cond_dim] → [B, L, cond_dim]
            return cond_t.unsqueeze(1).expand(B, L, -1)
        # Already per-residue: [B, L, cond_dim]
        return cond_t

    def forward(
        self,
        q: torch.Tensor,
        ca_distances: torch.Tensor,
        residue_mask: torch.Tensor,
        condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        B, L, _ = q.shape
        cond = self._condition_broadcast(q, condition)        # [B, L, cond_dim]

        # ── Temperature conditioning ──
        # The dimensionless free-energy landscape itself is T-independent,
        # but the network needs a temperature signal so it can learn to shape
        # the potential differently across the temperature range.  Without
        # this, all temperatures produce identical forces and the model shows
        # zero temperature sensitivity in intervention experiments.
        t_feat = torch.zeros(
            B, L, 1 if self.temperature_feature else 0,
            device=q.device, dtype=q.dtype,
        )
        if self.temperature_feature and temperature_k is not None:
            t_k = torch.as_tensor(temperature_k, device=q.device, dtype=q.dtype)
            if t_k.ndim == 0:
                t_k = t_k.expand(B)
            t_feat = torch.log(t_k / self.reference_temperature_k).reshape(B, 1, 1).expand(B, L, 1)

        rbf = self._rbf(ca_distances)
        energies = []
        for b in range(B):
            qi = q[b]           # [L, cv_dim]
            ri = rbf[b]         # [L, L, num_rbf]
            mi = residue_mask[b]  # [L]
            ci = cond[b]        # [L, cond_dim]
            ti = t_feat[b]      # [L, 1]
            mask = mi.to(dtype=q.dtype)
            pair_mask = mask[:, None] * mask[None, :]  # [L, L], float
            # Remove self-edges
            self_mask = torch.eye(L, device=mi.device, dtype=mask.dtype)
            pair_mask = pair_mask * (1.0 - self_mask)
            h = qi.unsqueeze(0)  # [1, L, cv_dim]
            h_with_t = torch.cat([h, ti.unsqueeze(0)], dim=-1)  # [1, L, cv_dim+1]
            ri_b = ri.unsqueeze(0)  # [1, L, L, num_rbf]
            pair_mask_b = pair_mask.unsqueeze(0)  # [1, L, L]
            mask_b = mask.unsqueeze(0)  # [1, L]
            ci_b = ci.unsqueeze(0)  # [1, L, cond_dim]
            cond_edge = ci_b.unsqueeze(2).expand(-1, L, L, -1)  # [1, L, L, cond_dim]
            cond_node = ci_b.expand(-1, L, -1)  # [1, L, cond_dim]
            for layer_idx in range(self.depth):
                # Extract q-part from full node state (last dim == cv_dim + 1)
                h_q = h_with_t[:, :, :self.cv_dim]  # [1, L, cv_dim]
                qi_exp = h_q[:, :, None, :].expand(-1, -1, L, -1)
                qj_exp = h_q[:, None, :, :].expand(-1, L, -1, -1)
                edge_ij = torch.cat([qi_exp, qj_exp, ri_b, cond_edge], dim=-1)
                edge_ji = torch.cat([qj_exp, qi_exp, ri_b, cond_edge], dim=-1)
                msg_ij = self.edge_net(edge_ij)
                msg_ji = self.edge_net(edge_ji)
                messages = 0.5 * (msg_ij + msg_ji) * pair_mask_b.unsqueeze(-1)
                denom = pair_mask_b.sum(dim=2, keepdim=True).clamp(min=1.0)
                aggregated = messages.sum(dim=2) / denom
                # Node input includes temperature: [q, messages, cond, T_feat]
                node_in = torch.cat([h_with_t, aggregated, cond_node], dim=-1)
                delta = self.node_nets[layer_idx](node_in) * mask_b.unsqueeze(-1)
                # Only the q-part receives the node net output
                h_q = h_q + delta
                h_with_t = torch.cat([h_q, ti.unsqueeze(0)], dim=-1)
            # Energy head sees [h_q, cond, T_feat]
            energy_in = torch.cat([h_with_t[:, :, :self.cv_dim], cond_node, ti.unsqueeze(0)], dim=-1)
            e = self.energy_head(energy_in).squeeze(-1) * mask_b
            energies.append(e)
        result = torch.cat(energies, dim=0)  # [B, L]
        return result

    def _forward_single(
        self,
        q: torch.Tensor,
        ca_distances: torch.Tensor,
        residue_mask: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = q.shape
        assert B == 1
        cond = self._condition_broadcast(q, condition)
        rbf = self._rbf(ca_distances)
        mask = residue_mask.to(dtype=q.dtype)
        pair_mask = mask[:, :, None] * mask[:, None, :]
        self_mask = torch.eye(L, device=residue_mask.device, dtype=mask.dtype).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - self_mask)
        h = q
        for layer_idx in range(self.depth):
            qi_exp = h[:, :, None, :].expand(-1, -1, L, -1)
            qj_exp = h[:, None, :, :].expand(-1, L, -1, -1)
            # cond is [B, L, cond_dim]; make per-edge [B, L, L, cond_dim]
            cond_edge = cond[:, :, None, :].expand(-1, -1, L, -1)
            edge_ij = torch.cat([qi_exp, qj_exp, rbf, cond_edge], dim=-1)
            edge_ji = torch.cat([qj_exp, qi_exp, rbf, cond_edge], dim=-1)
            msg_ij = self.edge_net(edge_ij)
            msg_ji = self.edge_net(edge_ji)
            messages = 0.5 * (msg_ij + msg_ji) * pair_mask.unsqueeze(-1)
            denom = pair_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
            aggregated = messages.sum(dim=2) / denom
            node_in = torch.cat([h, aggregated, cond], dim=-1)
            delta = self.node_nets[layer_idx](node_in) * mask.unsqueeze(-1)
            h = h + delta
        energy_in = torch.cat([h, cond], dim=-1)
        energy = self.energy_head(energy_in).squeeze(-1)
        return energy * mask

    def force(
        self,
        q: torch.Tensor,
        ca_distances: torch.Tensor,
        residue_mask: torch.Tensor,
        condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | float | None = None,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return the conservative force from the canonical energy path.

        Keep this helper on the same implementation as ``forward``.  The old
        private fast path predated explicit temperature features, had stale
        input dimensions, and either ignored temperature or failed with a
        shape error when called by score/direction objectives.
        """
        forces = []
        for b in range(q.shape[0]):
            qi = q[b:b+1]
            ri = ca_distances[b:b+1]
            mi = residue_mask[b:b+1]
            ci = condition[b:b+1] if condition is not None else None
            ti = None
            if temperature_k is not None:
                temp = torch.as_tensor(temperature_k, device=q.device, dtype=q.dtype)
                ti = temp if temp.ndim == 0 else temp.reshape(q.shape[0], -1)[b:b+1, -1]
            q_leaf = qi.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                energy = self.forward(q_leaf, ri, mi, ci, temperature_k=ti)
                gradient = torch.autograd.grad(
                    outputs=energy.sum(), inputs=q_leaf,
                    create_graph=create_graph, retain_graph=create_graph,
                    allow_unused=True,
                )[0]
            if gradient is None:
                forces.append(torch.zeros_like(q_leaf))
            else:
                forces.append(-gradient)
        return torch.cat(forces, dim=0)


def sliced_score_matching_loss(
    potential: ResidueMessagePassingPotential,
    q: torch.Tensor,
    ca_distances: torch.Tensor,
    residue_mask: torch.Tensor,
    condition: torch.Tensor | None = None,
    *,
    noise_std: float = 0.05,
    num_projections: int = 4,
) -> torch.Tensor:
    B, L, d = q.shape
    mask = residue_mask.to(dtype=q.dtype)
    n_valid = mask.sum(dim=1).clamp(min=1.0)
    q_flat = q.reshape(B, L * d)
    mask_flat = mask.unsqueeze(-1).expand(-1, -1, d).reshape(B, L * d)
    noise = torch.randn_like(q_flat) * noise_std
    q_noisy = q_flat + noise
    q_noisy = q_noisy.reshape(B, L, d)
    proj = torch.randn(num_projections, L * d, device=q.device, dtype=q.dtype)
    with torch.no_grad():
        force_clean = potential.force(q, ca_distances, residue_mask, condition)
        force_clean_flat = force_clean.reshape(B, L * d)
        force_clean_proj = torch.einsum("pd,bd->bp", proj, force_clean_flat)
    force_noisy = potential.force(q_noisy, ca_distances, residue_mask, condition, create_graph=True)
    force_noisy_flat = force_noisy.reshape(B, L * d)
    force_noisy_proj = torch.einsum("pd,bd->bp", proj, force_noisy_flat)
    diff = (force_noisy_proj - force_clean_proj) / noise_std
    per_sample = diff.square().mean(dim=1)
    weight = n_valid / n_valid.sum()
    return (per_sample * weight).sum()


def residue_force_direction_loss(
    potential: ResidueMessagePassingPotential,
    q_current: torch.Tensor,
    q_next: torch.Tensor,
    ca_distances: torch.Tensor,
    residue_mask: torch.Tensor,
    condition: torch.Tensor | None = None,
    *,
    hinge_margin: float = 0.0,
) -> torch.Tensor:
    mask = residue_mask.to(dtype=q_current.dtype)
    force = potential.force(q_current, ca_distances, residue_mask, condition, create_graph=True)
    displacement = (q_next - q_current).detach()
    force_norm = force.norm(dim=-1).clamp(min=1e-8)
    disp_norm = displacement.norm(dim=-1).clamp(min=1e-8)
    cos_sim = (force * displacement).sum(dim=-1) / (force_norm * disp_norm)
    hinge = torch.relu(-cos_sim + hinge_margin)
    per_sample = (hinge * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return per_sample.mean()


class ReversiblePotential(nn.Module):
    """A scalar Helmholtz potential H(q) for the reversible (divergence-free) drift.

    ``ReversiblePotential`` uses the same architecture as ``MeanForcePotential``
    but is a separate instance so the reversible free-energy landscape is
    parameterized independently from the dissipative one.
    """

    def __init__(
        self,
        state_dim: int,
        condition_dim: int = 0,
        hidden_dim: int = 128,
        hidden_depth: int = 2,
        reference_temperature_k: float = 348.0,
    ) -> None:
        super().__init__()
        if state_dim < 1 or condition_dim < 0 or hidden_dim < 4 or reference_temperature_k <= 0.0:
            raise ValueError("invalid reversible potential dimensions")
        if hidden_depth < 1:
            raise ValueError("hidden_depth must be at least 1")
        self.state_dim = int(state_dim)
        self.condition_dim = int(condition_dim)
        self.reference_temperature_k = float(reference_temperature_k)
        input_dim = self.state_dim + self.condition_dim + 1
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(hidden_depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.normal_(self.network[-1].bias, std=0.01)

    def _condition(self, q: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.condition_dim == 0:
            return q.new_zeros(*q.shape[:-1], 0)
        if condition is None:
            return q.new_zeros(*q.shape[:-1], self.condition_dim)
        if condition.shape != (*q.shape[:-1], self.condition_dim):
            raise ValueError("condition shape does not match the reversible potential")
        return condition.to(dtype=q.dtype, device=q.device)

    def forward(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if q.shape[-1] != self.state_dim:
            raise ValueError("q has the wrong state dimension for the reversible potential")
        temperature = _as_batch_temperature(
            temperature_k, q.shape[0], q, self.reference_temperature_k,
        )
        temperature_feature = torch.log(
            temperature / self.reference_temperature_k
        ).reshape(*q.shape[:-1], 1)
        return self.network(
            torch.cat([q, self._condition(q, condition), temperature_feature], dim=-1)
        ).squeeze(-1)

    def mean_force(
        self,
        q: torch.Tensor,
        condition: torch.Tensor | None = None,
        temperature_k: torch.Tensor | float | None = None,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Return ``-grad H`` (reversible force)."""
        with torch.enable_grad():
            q_leaf = q if q.requires_grad else q.detach().requires_grad_(True)
            energy = self(q_leaf, condition, temperature_k)
            gradient = torch.autograd.grad(
                energy.sum(), q_leaf,
                create_graph=create_graph, retain_graph=create_graph,
            )[0]
        return -gradient


class SkewSymmetricOperator(nn.Module):
    """Low-rank skew-symmetric operator J = P Q^T - Q P^T.

    The low-rank parameterization ensures that the operator does not introduce
    more than ``2 * rank`` effective degrees of freedom, which acts as an
    implicit regularizer against learning spurious high-frequency rotation.
    """

    def __init__(self, state_dim: int, rank: int = 16) -> None:
        super().__init__()
        if state_dim < 1 or rank < 1 or rank > state_dim:
            raise ValueError(
                f"skew operator rank must be between 1 and state_dim ({state_dim}), got {rank}"
            )
        self.state_dim = int(state_dim)
        self.rank = int(rank)
        self.P = nn.Parameter(torch.zeros(state_dim, rank))
        self.Q = nn.Parameter(torch.zeros(state_dim, rank))

    def forward(self, batch: int | None = None) -> torch.Tensor:
        J = self.P @ self.Q.T - self.Q @ self.P.T
        if batch is not None:
            J = J.unsqueeze(0).expand(batch, -1, -1)
        return J

    def apply(self, gradient: torch.Tensor) -> torch.Tensor:
        J = self.forward(batch=gradient.shape[0])
        return torch.einsum("bqd,bd->bq", J, gradient)


class UnifiedEnergyGraph(nn.Module):
    """Conservative hierarchical energy A(Q,G) with local, pair, global and coupling terms."""

    def __init__(self, cv_dim=8, global_dim=8, condition_dim=64, num_rbf=8, hidden_dim=64):
        super().__init__()
        self.cv_dim = cv_dim
        self.global_dim = global_dim
        self.condition_dim = condition_dim
        self.num_rbf = num_rbf
        self.local_energy = nn.Sequential(
            nn.Linear(cv_dim + condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.pair_energy = nn.Sequential(
            nn.Linear(2 * cv_dim + num_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.global_pmf = MeanForcePotential(global_dim, condition_dim, hidden_dim=hidden_dim)
        self.coupling_energy = nn.Sequential(
            nn.Linear(cv_dim + global_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.register_buffer('rbf_centres', torch.linspace(1., 30., num_rbf), persistent=False)

    def forward(self, q, G, ca_distances, residue_mask, condition=None, temperature_k=None):
        B, L, _ = q.shape
        m = residue_mask.to(q.dtype)
        c = q.new_zeros(B, L, self.condition_dim) if condition is None else condition[:, None, :].expand(B, L, -1)
        local = self.local_energy(torch.cat([q, c], -1)).squeeze(-1) * m
        rbf = torch.exp(-((ca_distances[..., None] - self.rbf_centres) / 3.) ** 2)
        qi = q[:, :, None, :].expand(-1, -1, L, -1)
        qj = q[:, None, :, :].expand(-1, L, -1, -1)
        pair = self.pair_energy(torch.cat([qi, qj, rbf], -1)).squeeze(-1)
        pm = m[:, :, None] * m[:, None, :]
        pair = (pair * pm).sum(-1) / pm.sum(-1).clamp_min(1)
        coupling = self.coupling_energy(torch.cat([q, G[:, None, :].expand(-1, L, -1)], -1)).squeeze(-1) * m
        glob = self.global_pmf(G, condition, temperature_k)
        total_energy = local + pair + coupling
        return {
            'energy_per_residue': total_energy,
            'energy_total': total_energy.sum(1) + glob,
            'energy_global': glob,
            'energy_local': local,
            'energy_pair': pair,
            'energy_coupling': coupling,
        }

    def force_q(self, q, *args, **kwargs):
        q = q.requires_grad_(True)
        return -torch.autograd.grad(self(q, *args, **kwargs)['energy_total'].sum(), q, create_graph=True)[0]

    def force_G(self, q, G, *args, **kwargs):
        G = G.requires_grad_(True)
        return -torch.autograd.grad(self(q, G, *args, **kwargs)['energy_total'].sum(), G, create_graph=True)[0]


def force_score_G_loss(
    unified_energy: UnifiedEnergyGraph,
    q: torch.Tensor,
    G: torch.Tensor,
    ca_distances: torch.Tensor,
    residue_mask: torch.Tensor,
    condition: torch.Tensor | None,
    temperature_k: torch.Tensor | None,
    reference_G: torch.Tensor,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    if reference_G.shape[0] < 2:
        return G.new_zeros(())
    kde_target = kde_mean_force(G.detach(), reference_G.detach(), bandwidth)
    global_force = unified_energy.force_G(q, G, ca_distances, residue_mask, condition, temperature_k)
    return (global_force - kde_target.detach()).square().sum(dim=-1).mean()


def force_score_Q_loss(
    unified_energy: UnifiedEnergyGraph,
    q: torch.Tensor,
    G: torch.Tensor,
    ca_distances: torch.Tensor,
    residue_mask: torch.Tensor,
    condition: torch.Tensor | None,
    temperature_k: torch.Tensor | None,
    noise_std: float = 0.05,
    num_projections: int = 4,
) -> torch.Tensor:
    B, L, D = q.shape
    q_detached = q.detach()
    noise = torch.randn(B, L, D, device=q.device, dtype=q.dtype) * noise_std
    q_noisy = q_detached + noise
    sigma2 = noise_std ** 2
    if num_projections <= 0:
        return q.new_zeros(())
    loss = q.new_zeros(())
    n_valid = residue_mask.sum().clamp_min(1.0)
    score_noisy = unified_energy.force_q(
        q_noisy, G, ca_distances, residue_mask, condition, temperature_k
    )
    scaled_score = score_noisy * sigma2
    target_noise = -noise
    for _ in range(num_projections):
        v = torch.randn(D, device=q.device, dtype=q.dtype)
        v = v / v.norm().clamp(min=1e-8)
        proj_noisy = (scaled_score * v).sum(-1)
        proj_target = (target_noise * v).sum(-1)
        loss = loss + ((proj_noisy - proj_target).square() * residue_mask).sum()
    return (loss / n_valid) / num_projections


def self_consistency_loss(
    unified_energy: UnifiedEnergyGraph,
    q: torch.Tensor,
    G: torch.Tensor,
    ca_distances: torch.Tensor,
    residue_mask: torch.Tensor,
    condition: torch.Tensor | None,
    temperature_k: torch.Tensor | None,
    reference_G: torch.Tensor,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    if reference_G.shape[0] < 2 or G.shape[0] < 2:
        return G.new_zeros(())
    B, L, D = q.shape
    mask = residue_mask.to(q.dtype)
    # 1. A_g alone vs KDE target
    G_ag = G.detach().requires_grad_(True)
    E_global = unified_energy.global_pmf(G_ag, condition, temperature_k)
    grad_A_g = torch.autograd.grad(E_global.sum(), G_ag, create_graph=True)[0]
    kde_target = kde_mean_force(G.detach(), reference_G.detach(), bandwidth).detach()
    L_ag_kde = (grad_A_g - kde_target).square().sum(dim=-1).mean()
    # 2. Per-residue coupling force variance
    G_phi = G.detach().requires_grad_(True)
    G_expanded = G_phi[:, None, :].expand(-1, L, -1)
    coupling_raw = unified_energy.coupling_energy(
        torch.cat([q.detach(), G_expanded], -1)
    ).squeeze(-1) * mask
    grad_coupling = torch.autograd.grad(
        coupling_raw.sum(), G_phi, create_graph=True, retain_graph=True
    )[0]
    grad_coupling_norm = grad_coupling.norm(dim=-1).clamp(min=1e-8)
    # 3. Full force vs A_g-only force: coupling contribution ratio
    G_full = G.detach().requires_grad_(True)
    E_full = unified_energy(q.detach(), G_full, ca_distances, residue_mask,
                             condition, temperature_k)
    grad_full = torch.autograd.grad(
        E_full["energy_total"].sum(), G_full, create_graph=True
    )[0]
    coupling_ratio = grad_coupling_norm / grad_full.norm(dim=-1).clamp(min=1e-8)
    L_ratio_reg = (torch.relu(coupling_ratio - 0.5)).mean()
    # 4. Coupling non-degeneracy: energy variance across residues
    with torch.no_grad():
        coupling_std_res = coupling_raw.std(dim=1).mean()
    L_coupling_entropy = torch.relu(0.01 - coupling_std_res)
    return L_ag_kde + 0.1 * L_ratio_reg + 0.05 * L_coupling_entropy


def stationary_distribution_loss(
    rollout_samples: torch.Tensor,
    md_samples: torch.Tensor,
    sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    if rollout_samples.shape[0] < 2 or md_samples.shape[0] < 2:
        return rollout_samples.new_zeros(())
    if sigma is None:
        all_samples = torch.cat([rollout_samples, md_samples], dim=0)
        pairwise_dist = torch.cdist(all_samples, all_samples)
        sigma = pairwise_dist.median() * 0.5
    xx = (rollout_samples ** 2).sum(-1, keepdim=True)
    yy = (md_samples ** 2).sum(-1, keepdim=True)
    xy = rollout_samples @ md_samples.T
    k_xy = torch.exp(-(xx + yy.T - 2 * xy) / (2 * sigma ** 2 + 1e-8)).mean()
    x2 = (rollout_samples ** 2).sum(-1, keepdim=True)
    k_xx = torch.exp(-(x2 + x2.T - 2 * rollout_samples @ rollout_samples.T) / (2 * sigma ** 2 + 1e-8)).mean()
    y2 = (md_samples ** 2).sum(-1, keepdim=True)
    k_yy = torch.exp(-(y2 + y2.T - 2 * md_samples @ md_samples.T) / (2 * sigma ** 2 + 1e-8)).mean()
    return (k_xx + k_yy - 2 * k_xy).clamp(min=0.0)
