from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


K_B_KCAL_PER_MOL_K = 0.0019872041


def _last_valid(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    index = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
    last = torch.where(mask, index, -1).max(dim=1).values.clamp(min=0)
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, last]


def _first_valid(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    index = torch.arange(mask.shape[1], device=mask.device).view(1, -1)
    first = torch.where(mask, index, mask.shape[1]).min(dim=1).values.clamp(
        max=mask.shape[1] - 1
    )
    rows = torch.arange(values.shape[0], device=values.device)
    return values[rows, first]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.dim() < values.dim():
        weight = weight.unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp(min=1.0)


def _dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    b0 = b - a
    b1 = c - b
    b2 = d - c
    b1_unit = F.normalize(b1, dim=-1, eps=1.0e-8)
    v = b0 - (b0 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1_unit, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


class FixedBackboneCollectiveVariables(nn.Module):
    """Differentiable, SE(3)-invariant backbone collective variables.

    For three contiguous multiscale segments this produces 16 coordinates:
    four global shape/contact values, two values per segment, two adjacent
    segment distances, and four global phi/psi Fourier moments.
    """

    def __init__(
        self,
        num_segments: int = 3,
        contact_cutoff_angstrom: float = 8.0,
        contact_width_angstrom: float = 0.75,
        distance_scale_angstrom: float = 10.0,
    ):
        super().__init__()
        if num_segments < 2:
            raise ValueError("TM-JEPA requires at least two physical segments")
        if contact_cutoff_angstrom <= 0 or contact_width_angstrom <= 0:
            raise ValueError("TM-JEPA contact scales must be positive")
        if distance_scale_angstrom <= 0:
            raise ValueError("TM-JEPA distance scale must be positive")
        self.num_segments = int(num_segments)
        self.contact_cutoff_angstrom = float(contact_cutoff_angstrom)
        self.contact_width_angstrom = float(contact_width_angstrom)
        self.distance_scale_angstrom = float(distance_scale_angstrom)
        self.output_dim = 3 * self.num_segments + 7
        names = ["global_rg", "end_to_end", "global_contact", "contact_order"]
        for segment in range(self.num_segments):
            names.extend([f"segment_{segment}_rg", f"segment_{segment}_contact"])
        names.extend(
            f"segment_{segment}_{segment + 1}_distance"
            for segment in range(self.num_segments - 1)
        )
        names.extend(["mean_sin_phi", "mean_cos_phi", "mean_sin_psi", "mean_cos_psi"])
        self.channel_names = tuple(names)

    def to_unconstrained(self, physical_cv: torch.Tensor) -> torch.Tensor:
        """Map positive and bounded observables to a fixed Lebesgue coordinate."""
        eps = 1.0e-4
        positive = {0, 1}
        bounded = {2, 3}
        cursor = 4
        for _ in range(self.num_segments):
            positive.add(cursor)
            bounded.add(cursor + 1)
            cursor += 2
        positive.update(range(cursor, cursor + self.num_segments - 1))
        signed_start = cursor + self.num_segments - 1
        values = []
        for index in range(self.output_dim):
            value = physical_cv[..., index]
            if index in positive:
                value = value.clamp(min=eps).log()
            elif index in bounded:
                value = torch.logit(value.clamp(min=eps, max=1.0 - eps))
            elif index >= signed_start:
                value = torch.atanh(value.clamp(min=-1.0 + eps, max=1.0 - eps))
            values.append(value)
        return torch.stack(values, dim=-1)

    def to_physical(self, unconstrained_cv: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`to_unconstrained` for reporting and structural checks."""
        positive = {0, 1}
        bounded = {2, 3}
        cursor = 4
        for _ in range(self.num_segments):
            positive.add(cursor)
            bounded.add(cursor + 1)
            cursor += 2
        positive.update(range(cursor, cursor + self.num_segments - 1))
        signed_start = cursor + self.num_segments - 1
        values = []
        for index in range(self.output_dim):
            value = unconstrained_cv[..., index]
            if index in positive:
                value = value.exp()
            elif index in bounded:
                value = value.sigmoid()
            elif index >= signed_start:
                value = value.tanh()
            values.append(value)
        return torch.stack(values, dim=-1)

    def log_abs_det_to_unconstrained(self, physical_cv: torch.Tensor) -> torch.Tensor:
        """Return log|dq/dy| for physical y and unconstrained q coordinates."""
        eps = 1.0e-6
        positive = {0, 1}
        bounded = {2, 3}
        cursor = 4
        for _ in range(self.num_segments):
            positive.add(cursor)
            bounded.add(cursor + 1)
            cursor += 2
        positive.update(range(cursor, cursor + self.num_segments - 1))
        signed_start = cursor + self.num_segments - 1
        terms = []
        for index in range(self.output_dim):
            value = physical_cv[..., index]
            if index in positive:
                term = -value.clamp(min=eps).log()
            elif index in bounded:
                value = value.clamp(min=eps, max=1.0 - eps)
                term = -value.log() - torch.log1p(-value)
            elif index >= signed_start:
                term = -(1.0 - value.square()).clamp(min=eps).log()
            else:
                term = torch.zeros_like(value)
            terms.append(term)
        return torch.stack(terms, dim=-1).sum(dim=-1)

    def forward(
        self,
        coords: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        chain_id: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> torch.Tensor:
        if coords.dim() != 5 or coords.shape[-2:] != (4, 3):
            raise ValueError("TM-JEPA coordinates must have shape [B,T,L,4,3]")
        if atom_mask.shape != coords.shape[:-1] or residue_mask.shape != coords.shape[:3]:
            raise ValueError("TM-JEPA coordinate masks do not match coordinates")
        B, T, L = residue_mask.shape
        ca = coords[..., 1, :]
        valid = residue_mask & atom_mask[..., 1]
        weight = valid.to(ca.dtype).unsqueeze(-1)
        count = weight.sum(dim=2).clamp(min=1.0)
        center = (ca * weight).sum(dim=2) / count
        rg = torch.sqrt(
            ((ca - center.unsqueeze(2)).square() * weight).sum(dim=(2, 3))
            / count.squeeze(-1)
        ).clamp(min=1.0e-8) / self.distance_scale_angstrom

        position = torch.arange(L, device=coords.device).view(1, 1, L)
        first_index = torch.where(valid, position, L).min(dim=2).values.clamp(max=L - 1)
        last_index = torch.where(valid, position, -1).max(dim=2).values.clamp(min=0)
        rows = torch.arange(B, device=coords.device)[:, None]
        times = torch.arange(T, device=coords.device)[None, :]
        first_ca = ca[rows, times, first_index]
        last_ca = ca[rows, times, last_index]
        end_to_end = torch.linalg.vector_norm(last_ca - first_ca, dim=-1)
        end_to_end = end_to_end / self.distance_scale_angstrom

        distances = torch.cdist(ca.reshape(B * T, L, 3), ca.reshape(B * T, L, 3))
        distances = distances.reshape(B, T, L, L)
        sequence_separation = (residue_index[:, :, None] - residue_index[:, None, :]).abs()
        same_chain = chain_id[:, :, None] == chain_id[:, None, :]
        upper = torch.triu(torch.ones(L, L, dtype=torch.bool, device=coords.device), diagonal=1)
        pair_valid = (
            valid[:, :, :, None]
            & valid[:, :, None, :]
            & same_chain[:, None]
            & (sequence_separation[:, None] > 2)
            & upper
        )
        soft_contact = torch.sigmoid(
            (self.contact_cutoff_angstrom - distances) / self.contact_width_angstrom
        )
        pair_weight = pair_valid.to(ca.dtype)
        pair_count = pair_weight.sum(dim=(-1, -2)).clamp(min=1.0)
        global_contact = (soft_contact * pair_weight).sum(dim=(-1, -2)) / pair_count
        normalized_separation = sequence_separation[:, None].to(ca.dtype) / valid.sum(
            dim=2
        ).clamp(min=1).to(ca.dtype)[:, :, None, None]
        contact_mass = (soft_contact * pair_weight).sum(dim=(-1, -2)).clamp(min=1.0e-6)
        contact_order = (
            soft_contact * pair_weight * normalized_separation
        ).sum(dim=(-1, -2)) / contact_mass

        valid_length = valid.sum(dim=2).clamp(min=1)
        segment_id = torch.div(
            position * self.num_segments,
            valid_length.unsqueeze(-1),
            rounding_mode="floor",
        ).clamp(max=self.num_segments - 1)
        segment_values: list[torch.Tensor] = []
        segment_centers: list[torch.Tensor] = []
        segment_valids: list[torch.Tensor] = []
        for segment in range(self.num_segments):
            segment_mask = valid & (segment_id == segment)
            segment_weight = segment_mask.to(ca.dtype).unsqueeze(-1)
            segment_count = segment_weight.sum(dim=2).clamp(min=1.0)
            segment_center = (ca * segment_weight).sum(dim=2) / segment_count
            segment_rg = torch.sqrt(
                ((ca - segment_center.unsqueeze(2)).square() * segment_weight).sum(dim=(2, 3))
                / segment_count.squeeze(-1)
            ).clamp(min=1.0e-8) / self.distance_scale_angstrom
            segment_pair = pair_valid & segment_mask[:, :, :, None] & segment_mask[:, :, None, :]
            segment_pair_weight = segment_pair.to(ca.dtype)
            segment_pair_count = segment_pair_weight.sum(dim=(-1, -2)).clamp(min=1.0)
            segment_contact = (
                soft_contact * segment_pair_weight
            ).sum(dim=(-1, -2)) / segment_pair_count
            segment_values.extend([segment_rg, segment_contact])
            segment_centers.append(segment_center)
            segment_valids.append(segment_mask.sum(dim=2) > 0)

        adjacent_distances: list[torch.Tensor] = []
        for left in range(self.num_segments - 1):
            distance = torch.linalg.vector_norm(
                segment_centers[left + 1] - segment_centers[left], dim=-1
            ) / self.distance_scale_angstrom
            both_valid = segment_valids[left] & segment_valids[left + 1]
            adjacent_distances.append(torch.where(both_valid, distance, torch.zeros_like(distance)))

        n, ca_atom, c = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
        phi = _dihedral(c[:, :, :-1], n[:, :, 1:], ca_atom[:, :, 1:], c[:, :, 1:])
        psi = _dihedral(n[:, :, :-1], ca_atom[:, :, :-1], c[:, :, :-1], n[:, :, 1:])
        neighboring_chain = chain_id[:, :-1] == chain_id[:, 1:]
        neighboring_index = (residue_index[:, 1:] - residue_index[:, :-1]) == 1
        phi_mask = (
            atom_mask[:, :, :-1, 2]
            & atom_mask[:, :, 1:, 0]
            & atom_mask[:, :, 1:, 1]
            & atom_mask[:, :, 1:, 2]
            & residue_mask[:, :, :-1]
            & residue_mask[:, :, 1:]
            & neighboring_chain[:, None]
            & neighboring_index[:, None]
        )
        psi_mask = (
            atom_mask[:, :, :-1, 0]
            & atom_mask[:, :, :-1, 1]
            & atom_mask[:, :, :-1, 2]
            & atom_mask[:, :, 1:, 0]
            & residue_mask[:, :, :-1]
            & residue_mask[:, :, 1:]
            & neighboring_chain[:, None]
            & neighboring_index[:, None]
        )
        torsion_values = [
            _masked_mean(torch.sin(phi), phi_mask, dim=2),
            _masked_mean(torch.cos(phi), phi_mask, dim=2),
            _masked_mean(torch.sin(psi), psi_mask, dim=2),
            _masked_mean(torch.cos(psi), psi_mask, dim=2),
        ]
        physical_result = torch.stack(
            [rg, end_to_end, global_contact, contact_order]
            + segment_values
            + adjacent_distances
            + torsion_values,
            dim=-1,
        )
        if physical_result.shape != (B, T, self.output_dim):
            raise RuntimeError("TM-JEPA collective-variable dimension mismatch")
        return torch.nan_to_num(self.to_unconstrained(physical_result))


class ProteinConditionEncoder(nn.Module):
    """Sequence-only condition, intentionally independent of horizon and temperature."""

    def __init__(self, condition_dim: int, dataset_vocab_size: int = 4):
        super().__init__()
        if condition_dim < 8:
            raise ValueError("TM-JEPA condition_dim must be at least 8")
        self.aatype_embedding = nn.Embedding(21, condition_dim)
        self.position_projection = nn.Linear(3, condition_dim)
        self.sequence_conv = nn.Conv1d(condition_dim, condition_dim, kernel_size=3, padding=1)
        self.dataset_embedding = nn.Embedding(dataset_vocab_size, condition_dim)
        self.output = nn.Sequential(
            nn.LayerNorm(condition_dim * 3),
            nn.Linear(condition_dim * 3, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, condition_dim),
        )

    def forward(
        self,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        residue_mask: torch.Tensor,
        dataset_index: torch.Tensor | None,
    ) -> torch.Tensor:
        mask = residue_mask.bool()
        index = torch.nan_to_num(residue_index.to(torch.float32), nan=0.0)
        minimum = torch.where(mask, index, torch.full_like(index, float("inf"))).min(dim=1).values
        maximum = torch.where(mask, index, torch.full_like(index, float("-inf"))).max(dim=1).values
        minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
        maximum = torch.where(torch.isfinite(maximum), maximum, minimum + 1.0)
        span = (maximum - minimum).clamp(min=1.0)
        normalized = (index - minimum[:, None]) / span[:, None]
        position = torch.stack(
            [normalized, torch.sin(math.pi * normalized), torch.cos(math.pi * normalized)], dim=-1
        )
        token = self.aatype_embedding(aatype.clamp(min=0, max=20)) + self.position_projection(position)
        token = self.sequence_conv(token.transpose(1, 2)).transpose(1, 2)
        token = F.silu(token) * mask.unsqueeze(-1).to(token.dtype)
        mean = _masked_mean(token, mask, dim=1)
        maximum_value = token.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        maximum_value = torch.where(torch.isfinite(maximum_value), maximum_value, torch.zeros_like(maximum_value))
        if dataset_index is None:
            dataset_index = torch.zeros(aatype.shape[0], dtype=torch.long, device=aatype.device)
        dataset = self.dataset_embedding(dataset_index.clamp(min=0, max=self.dataset_embedding.num_embeddings - 1))
        return self.output(torch.cat([mean, maximum_value, dataset], dim=-1))


class CausalAuxiliaryFilter(nn.Module):
    """Infer dimensionless Mori bath variables from physical CV history."""

    def __init__(self, cv_dim: int, auxiliary_dim: int, base_step_ps: float, hidden_dim: int):
        super().__init__()
        if base_step_ps <= 0:
            raise ValueError("TM-JEPA observer time scale must be positive")
        self.auxiliary_dim = int(auxiliary_dim)
        self.base_step_ps = float(base_step_ps)
        self.input_projection = nn.Sequential(
            nn.Linear(cv_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.posterior_mean = nn.Linear(hidden_dim, auxiliary_dim)
        self.posterior_logvar = nn.Linear(hidden_dim, auxiliary_dim)
        nn.init.zeros_(self.posterior_mean.weight)
        nn.init.zeros_(self.posterior_mean.bias)
        nn.init.zeros_(self.posterior_logvar.weight)
        nn.init.constant_(self.posterior_logvar.bias, -1.0)

    def forward(
        self, sequence: torch.Tensor, mask: torch.Tensor, time_ps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_dt = (time_ps[:, 1:] - time_ps[:, :-1]).abs()
        dt = raw_dt.clamp(min=1.0e-6)
        pair_mask = mask[:, 1:] & mask[:, :-1]
        flux = torch.zeros_like(sequence)
        flux[:, 1:] = torch.where(
            pair_mask.unsqueeze(-1),
            (sequence[:, 1:] - sequence[:, :-1]) / dt.unsqueeze(-1),
            torch.zeros_like(sequence[:, 1:]),
        )
        standardized_flux = flux * self.base_step_ps
        # Every token may depend only on the current prefix.  Using the final
        # context frame here would leak the future into intermediate states.
        reference = _first_valid(sequence, mask)
        relative_dt = torch.zeros_like(time_ps)
        relative_dt[:, 1:] = torch.where(pair_mask, raw_dt / self.base_step_ps, 0.0)
        time_feature = torch.log1p(relative_dt).unsqueeze(-1)
        tokens = self.input_projection(
            torch.cat(
                [sequence - reference.unsqueeze(1), standardized_flux, time_feature],
                dim=-1,
            )
        )
        hidden = torch.zeros(
            sequence.shape[0], self.cell.hidden_size, device=sequence.device, dtype=sequence.dtype
        )
        hidden_sequence = []
        for step in range(sequence.shape[1]):
            candidate = self.cell(tokens[:, step], hidden)
            hidden = torch.where(mask[:, step : step + 1], candidate, hidden)
            hidden_sequence.append(hidden)
        hidden_values = torch.stack(hidden_sequence, dim=1)
        return self.posterior_mean(hidden_values), self.posterior_logvar(hidden_values).clamp(-8.0, 4.0)


class ThermodynamicMoriDynamics(nn.Module):
    """Free-energy-driven Markov embedding of a generalized Langevin process."""

    def __init__(
        self,
        cv_dim: int,
        condition_dim: int,
        auxiliary_dim: int = 4,
        memory_enabled: bool = True,
        base_step_ps: float = 1000.0,
        max_rollout_steps: int = 256,
        ensemble_size: int = 8,
        diffusion_floor: float = 1.0e-8,
        diffusion_scale: float = 1.0e-5,
        relaxation_rate_scale: float = 1.0e-3,
        coupling_scale: float = 1.0e-3,
        equilibrium_noise_std: float = 0.05,
    ):
        super().__init__()
        if cv_dim <= 0 or condition_dim <= 0 or auxiliary_dim <= 0:
            raise ValueError("TM-JEPA dimensions must be positive")
        if auxiliary_dim > cv_dim:
            raise ValueError("TM-JEPA auxiliary_dim cannot exceed cv_dim")
        if base_step_ps <= 0 or max_rollout_steps <= 0 or ensemble_size < 2:
            raise ValueError("TM-JEPA integration settings are invalid")
        if min(diffusion_floor, diffusion_scale, relaxation_rate_scale, coupling_scale) < 0:
            raise ValueError("TM-JEPA kinetic scales must be non-negative")
        if equilibrium_noise_std <= 0:
            raise ValueError("TM-JEPA equilibrium noise must be positive")
        self.cv_dim = int(cv_dim)
        self.condition_dim = int(condition_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.memory_enabled = bool(memory_enabled)
        self.base_step_ps = float(base_step_ps)
        self.max_rollout_steps = int(max_rollout_steps)
        self.ensemble_size = int(ensemble_size)
        self.diffusion_floor = float(diffusion_floor)
        self.diffusion_scale = float(diffusion_scale)
        self.relaxation_rate_scale = float(relaxation_rate_scale)
        self.coupling_scale = float(coupling_scale)
        self.equilibrium_noise_std = float(equilibrium_noise_std)

        energy_dim = max(64, condition_dim)
        self.energy_trunk = nn.Sequential(
            nn.Linear(cv_dim + condition_dim, energy_dim * 2),
            nn.SiLU(),
            nn.Linear(energy_dim * 2, energy_dim),
            nn.SiLU(),
        )
        self.enthalpy_head = nn.Linear(energy_dim, 1)
        self.entropy_over_kb_head = nn.Linear(energy_dim, 1)
        self.mode_basis = nn.Parameter(torch.randn(cv_dim, auxiliary_dim) * 0.05)
        self.coupling_basis = nn.Parameter(torch.randn(cv_dim, auxiliary_dim) * 0.05)
        self.kinetic_head = nn.Sequential(
            nn.Linear(condition_dim + 3, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, cv_dim + auxiliary_dim * 3),
        )
        nn.init.zeros_(self.enthalpy_head.weight)
        nn.init.zeros_(self.enthalpy_head.bias)
        nn.init.zeros_(self.entropy_over_kb_head.weight)
        nn.init.zeros_(self.entropy_over_kb_head.bias)
        nn.init.zeros_(self.kinetic_head[-1].weight)
        initial_positive = math.log(math.expm1(0.1))
        with torch.no_grad():
            self.kinetic_head[-1].bias[:cv_dim].fill_(initial_positive)
            self.kinetic_head[-1].bias[cv_dim : cv_dim + auxiliary_dim].fill_(initial_positive)
            self.kinetic_head[-1].bias[
                cv_dim + auxiliary_dim : cv_dim + 2 * auxiliary_dim
            ].zero_()
            self.kinetic_head[-1].bias[cv_dim + 2 * auxiliary_dim :].fill_(initial_positive)

    @staticmethod
    def _temperature(temperature_k: torch.Tensor | None, batch: int, reference: torch.Tensor) -> torch.Tensor:
        if temperature_k is None:
            return reference.new_full((batch,), 300.0)
        return torch.nan_to_num(temperature_k.to(reference), nan=300.0).clamp(250.0, 600.0)

    def free_energy(
        self, q: torch.Tensor, condition: torch.Tensor, temperature_k: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = self._temperature(temperature_k, q.shape[0], q)
        hidden = self.energy_trunk(torch.cat([q, condition], dim=-1))
        enthalpy = self.enthalpy_head(hidden).squeeze(-1)
        entropy_over_kb = self.entropy_over_kb_head(hidden).squeeze(-1)
        free_energy = enthalpy - K_B_KCAL_PER_MOL_K * temperature * entropy_over_kb
        beta_free_energy = free_energy / (K_B_KCAL_PER_MOL_K * temperature)
        return free_energy, beta_free_energy, entropy_over_kb

    def _energy_gradient(
        self,
        q: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor | None,
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            if not q.requires_grad:
                q = q.detach().requires_grad_(True)
            free_energy, beta_free_energy, entropy = self.free_energy(q, condition, temperature_k)
            gradient = torch.autograd.grad(
                beta_free_energy.sum(), q, create_graph=create_graph, retain_graph=create_graph
            )[0]
        return q, gradient, free_energy, entropy

    def kinetic_parameters(
        self, condition: torch.Tensor, temperature_k: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        temperature = self._temperature(temperature_k, condition.shape[0], condition)
        ratio = temperature / 300.0
        temperature_features = torch.stack([ratio, ratio.reciprocal(), ratio.log()], dim=-1)
        raw = self.kinetic_head(torch.cat([condition, temperature_features], dim=-1))
        raw_diagonal, raw_diffusion, raw_coupling, raw_relaxation = torch.split(
            raw,
            [self.cv_dim, self.auxiliary_dim, self.auxiliary_dim, self.auxiliary_dim],
            dim=-1,
        )
        diagonal_diffusion_rates = self.diffusion_floor + F.softplus(raw_diagonal) * self.diffusion_scale
        diffusion_rates = F.softplus(raw_diffusion) * self.diffusion_scale
        coupling_rates = torch.tanh(raw_coupling) * self.coupling_scale
        if not self.memory_enabled:
            coupling_rates = torch.zeros_like(coupling_rates)
        relaxation_rates = F.softplus(raw_relaxation) * self.relaxation_rate_scale
        basis = torch.linalg.qr(self.mode_basis, mode="reduced")[0]
        coupling_basis = torch.linalg.qr(self.coupling_basis, mode="reduced")[0]
        diffusion = torch.diag_embed(diagonal_diffusion_rates)
        diffusion = diffusion + torch.einsum("qr,br,sr->bqs", basis, diffusion_rates, basis)
        coupling = coupling_basis.unsqueeze(0) * coupling_rates.unsqueeze(1)
        return {
            "basis": basis,
            "coupling_basis": coupling_basis,
            "diagonal_diffusion_rates": diagonal_diffusion_rates,
            "diffusion_rates": diffusion_rates,
            "coupling_rates": coupling_rates,
            "relaxation_rates": relaxation_rates,
            "diffusion": diffusion,
            "coupling": coupling,
        }

    def _drift(
        self,
        q: torch.Tensor,
        a: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor,
        kinetic: dict[str, torch.Tensor],
        *,
        create_graph: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, gradient, free_energy, _ = self._energy_gradient(
            q, condition, temperature_k, create_graph=create_graph
        )
        q_drift = -torch.einsum("bij,bj->bi", kinetic["diffusion"], gradient)
        q_drift = q_drift + torch.einsum("bqr,br->bq", kinetic["coupling"], a)
        a_drift = -torch.einsum("bqr,bq->br", kinetic["coupling"], gradient)
        a_drift = a_drift - kinetic["relaxation_rates"] * a
        return q_drift, a_drift, free_energy

    def rollout(
        self,
        q: torch.Tensor,
        auxiliary_mean: torch.Tensor,
        auxiliary_logvar: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor | None,
        horizon_time_ps: torch.Tensor,
        *,
        ensemble_size: int | None = None,
        stochastic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ensemble = int(ensemble_size or self.ensemble_size)
        B = q.shape[0]
        temperature = self._temperature(temperature_k, B, q)
        kinetic = self.kinetic_parameters(condition, temperature)
        q_state = q[:, None].expand(-1, ensemble, -1).reshape(B * ensemble, self.cv_dim).clone()
        auxiliary_std = torch.exp(0.5 * auxiliary_logvar)
        auxiliary_noise = torch.randn(
            B, ensemble, self.auxiliary_dim, device=q.device, dtype=q.dtype
        ) if stochastic else torch.zeros(B, ensemble, self.auxiliary_dim, device=q.device, dtype=q.dtype)
        a_state = (
            auxiliary_mean[:, None] + auxiliary_std[:, None] * auxiliary_noise
        ).reshape(B * ensemble, self.auxiliary_dim)
        condition_e = condition[:, None].expand(-1, ensemble, -1).reshape(B * ensemble, -1)
        temperature_e = temperature[:, None].expand(-1, ensemble).reshape(-1)
        kinetic_e = {
            key: value[:, None].expand(-1, ensemble, *value.shape[1:]).reshape(
                B * ensemble, *value.shape[1:]
            )
            for key, value in kinetic.items()
            if key not in {"basis", "coupling_basis"}
        }
        basis = kinetic["basis"]
        total_time = horizon_time_ps.to(q).clamp(min=0.0)
        steps = torch.ceil(total_time / self.base_step_ps).long()
        max_steps = int(steps.max().detach().cpu())
        if max_steps > self.max_rollout_steps:
            raise ValueError(
                f"TM-JEPA horizon needs {max_steps} steps, above max_rollout_steps={self.max_rollout_steps}"
            )
        elapsed = torch.zeros_like(total_time)
        for _ in range(max_steps):
            dt = torch.minimum(
                (total_time - elapsed).clamp(min=0.0), total_time.new_full(total_time.shape, self.base_step_ps)
            )
            dt_e = dt[:, None].expand(-1, ensemble).reshape(-1, 1)
            q_drift, a_drift, _ = self._drift(
                q_state,
                a_state,
                condition_e,
                temperature_e,
                kinetic_e,
                create_graph=self.training,
            )
            if stochastic:
                diagonal_noise = torch.randn_like(q_state) * torch.sqrt(
                    2.0 * kinetic_e["diagonal_diffusion_rates"]
                )
                mode_noise = torch.randn(
                    B * ensemble, self.auxiliary_dim, device=q.device, dtype=q.dtype
                )
                mode_noise = mode_noise * torch.sqrt(2.0 * kinetic_e["diffusion_rates"])
                q_noise = diagonal_noise + mode_noise @ basis.T
                a_noise = torch.randn_like(a_state) * torch.sqrt(
                    2.0 * kinetic_e["relaxation_rates"]
                )
            else:
                q_noise = torch.zeros_like(q_state)
                a_noise = torch.zeros_like(a_state)
            sqrt_dt = dt_e.sqrt()
            q_increment = q_noise * sqrt_dt
            a_increment = a_noise * sqrt_dt
            q_predictor = q_state + q_drift * dt_e + q_increment
            a_predictor = a_state + a_drift * dt_e + a_increment
            predictor_q_drift, predictor_a_drift, _ = self._drift(
                q_predictor,
                a_predictor,
                condition_e,
                temperature_e,
                kinetic_e,
                create_graph=self.training,
            )
            # Stochastic Heun is weak second order for this additive-noise SDE
            # and sharply reduces the OU equilibrium bias of Euler-Maruyama.
            q_state = q_state + 0.5 * (q_drift + predictor_q_drift) * dt_e + q_increment
            a_state = a_state + 0.5 * (a_drift + predictor_a_drift) * dt_e + a_increment
            elapsed = elapsed + dt
        return (
            q_state.reshape(B, ensemble, self.cv_dim),
            a_state.reshape(B, ensemble, self.auxiliary_dim),
        )

    def forward(
        self,
        context_cv: torch.Tensor,
        future_cv: torch.Tensor,
        context_mask: torch.Tensor,
        future_mask: torch.Tensor,
        context_time_ps: torch.Tensor,
        condition: torch.Tensor,
        temperature_k: torch.Tensor | None,
        horizon_time_ps: torch.Tensor,
        auxiliary_filter: CausalAuxiliaryFilter,
    ) -> dict[str, torch.Tensor]:
        current = _last_valid(context_cv, context_mask)
        target = _last_valid(future_cv, future_mask).detach()
        auxiliary_mean_sequence, auxiliary_logvar_sequence = auxiliary_filter(
            context_cv, context_mask, context_time_ps
        )
        auxiliary_mean = _last_valid(auxiliary_mean_sequence, context_mask)
        auxiliary_logvar = _last_valid(auxiliary_logvar_sequence, context_mask)
        samples, auxiliary_samples = self.rollout(
            current,
            auxiliary_mean,
            auxiliary_logvar,
            condition,
            temperature_k,
            horizon_time_ps,
        )
        prediction = samples.mean(dim=1)
        sample_variance = samples.var(dim=1, unbiased=False).clamp(min=1.0e-8)
        kinetic = self.kinetic_parameters(condition, temperature_k)
        current_leaf, current_gradient, free_energy, entropy = self._energy_gradient(
            current, condition, temperature_k, create_graph=self.training
        )

        score_noise = torch.randn_like(current) * self.equilibrium_noise_std
        noisy_q = current.detach() + score_noise
        noisy_q, noisy_gradient, _, _ = self._energy_gradient(
            noisy_q, condition, temperature_k, create_graph=self.training
        )
        score_prediction = -noisy_gradient
        score_target = -score_noise / (self.equilibrium_noise_std**2)

        dt = (context_time_ps[:, 1:] - context_time_ps[:, :-1]).abs().clamp(min=1.0e-6)
        pair_mask = context_mask[:, 1:] & context_mask[:, :-1]
        q0 = context_cv[:, :-1]
        q1 = context_cv[:, 1:]
        flat_q0 = q0.reshape(-1, self.cv_dim)
        flat_condition = condition[:, None].expand(-1, q0.shape[1], -1).reshape(-1, condition.shape[-1])
        temperature = self._temperature(temperature_k, context_cv.shape[0], context_cv)
        flat_temperature = temperature[:, None].expand(-1, q0.shape[1]).reshape(-1)
        flat_a = auxiliary_mean_sequence[:, :-1].reshape(-1, self.auxiliary_dim)
        flat_kinetic = {
            key: value[:, None].expand(-1, q0.shape[1], *value.shape[1:]).reshape(
                -1, *value.shape[1:]
            )
            for key, value in kinetic.items()
            if key not in {"basis", "coupling_basis"}
        }
        q_drift, a_drift, _ = self._drift(
            flat_q0,
            flat_a,
            flat_condition,
            flat_temperature,
            flat_kinetic,
            create_graph=self.training,
        )
        q_drift = q_drift.reshape_as(q0)
        residual = q1 - q0 - q_drift * dt.unsqueeze(-1)
        posterior_variance0 = auxiliary_logvar_sequence[:, :-1].exp()
        coupling = kinetic["coupling"][:, None].expand(
            -1, q0.shape[1], -1, -1
        )
        diffusion = kinetic["diffusion"][:, None].expand(
            -1, q0.shape[1], -1, -1
        )
        transition_covariance = 2.0 * dt[..., None, None] * diffusion
        transition_covariance = transition_covariance + dt[..., None, None].square() * torch.einsum(
            "btqr,btr,btsr->btqs", coupling, posterior_variance0, coupling
        )
        covariance_jitter = torch.eye(
            self.cv_dim, device=q0.device, dtype=q0.dtype
        ).view(1, 1, self.cv_dim, self.cv_dim) * 1.0e-8
        transition_covariance = transition_covariance + covariance_jitter
        cholesky = torch.linalg.cholesky(
            transition_covariance.reshape(-1, self.cv_dim, self.cv_dim)
        )
        innovation = torch.linalg.solve_triangular(
            cholesky, residual.reshape(-1, self.cv_dim).unsqueeze(-1), upper=False
        ).squeeze(-1).reshape_as(q0)
        innovation = torch.where(pair_mask.unsqueeze(-1), innovation, torch.zeros_like(innovation))

        log_determinant = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(dim=-1).reshape_as(dt)
        path_nll = 0.5 * (
            innovation.square().sum(dim=-1)
            + log_determinant
            + self.cv_dim * math.log(2.0 * math.pi)
        ) / float(self.cv_dim)
        path_nll = torch.where(pair_mask, path_nll, torch.zeros_like(path_nll))

        flat_a_drift = a_drift.reshape_as(auxiliary_mean_sequence[:, :-1])
        relaxation = kinetic["relaxation_rates"][:, None]
        decay = torch.exp(-relaxation * dt.unsqueeze(-1))
        forcing = flat_a_drift + relaxation * auxiliary_mean_sequence[:, :-1]
        forcing_scale = -torch.expm1(-relaxation * dt.unsqueeze(-1)) / relaxation.clamp(
            min=1.0e-12
        )
        auxiliary_prior_mean = (
            decay * auxiliary_mean_sequence[:, :-1] + forcing_scale * forcing
        )
        auxiliary_prior_variance = (
            decay.square() * posterior_variance0 + (1.0 - decay.square())
        ).clamp(min=1.0e-8)
        auxiliary_posterior_mean = auxiliary_mean_sequence[:, 1:]
        auxiliary_posterior_variance = auxiliary_logvar_sequence[:, 1:].exp()
        auxiliary_transition_kl = 0.5 * (
            auxiliary_prior_variance.log()
            - auxiliary_logvar_sequence[:, 1:]
            + (
                auxiliary_posterior_variance
                + (auxiliary_posterior_mean - auxiliary_prior_mean).square()
            )
            / auxiliary_prior_variance
            - 1.0
        ).mean(dim=-1)
        auxiliary_transition_kl = torch.where(
            pair_mask, auxiliary_transition_kl, torch.zeros_like(auxiliary_transition_kl)
        )

        result = {
            "tmjepa_current_cv": current_leaf,
            "tmjepa_target_cv": target,
            "tmjepa_samples": samples,
            "tmjepa_auxiliary_samples": auxiliary_samples,
            "tmjepa_prediction": prediction,
            "tmjepa_sample_variance": sample_variance,
            "tmjepa_free_energy": free_energy,
            "tmjepa_entropy_over_kb": entropy,
            "tmjepa_beta_force_gradient": current_gradient,
            "tmjepa_score_prediction": score_prediction,
            "tmjepa_score_target": score_target.detach(),
            "tmjepa_auxiliary_mean": auxiliary_mean,
            "tmjepa_auxiliary_logvar": auxiliary_logvar,
            "tmjepa_auxiliary_mean_sequence": auxiliary_mean_sequence,
            "tmjepa_innovation": innovation,
            "tmjepa_innovation_mask": pair_mask,
            "tmjepa_path_nll": path_nll,
            "tmjepa_auxiliary_transition_kl": auxiliary_transition_kl,
            "tmjepa_diffusion": kinetic["diffusion"],
            "tmjepa_diagonal_diffusion_rates": kinetic["diagonal_diffusion_rates"],
            "tmjepa_diffusion_rates": kinetic["diffusion_rates"],
            "tmjepa_coupling_rates": kinetic["coupling_rates"],
            "tmjepa_relaxation_rates": kinetic["relaxation_rates"],
            "tmjepa_mode_basis": kinetic["basis"],
            "tmjepa_coupling_basis": kinetic["coupling_basis"],
            "tmjepa_temperature_k": temperature,
        }
        if not self.training:
            result = {key: value.detach() for key, value in result.items()}
        return result


class ThermodynamicMoriJEPA(nn.Module):
    """End-to-end TM-JEPA physical state, filter, and stochastic generator."""

    def __init__(
        self,
        latent_dim: int,
        num_segments: int = 3,
        learned_cv_dim: int = 0,
        condition_dim: int = 64,
        auxiliary_dim: int = 4,
        memory_enabled: bool = True,
        observation_time_scale_ps: float = 1000.0,
        base_step_ps: float = 1000.0,
        max_rollout_steps: int = 256,
        ensemble_size: int = 8,
        diffusion_floor: float = 1.0e-8,
        diffusion_scale: float = 1.0e-5,
        relaxation_rate_scale: float = 1.0e-3,
        coupling_scale: float = 1.0e-3,
        equilibrium_noise_std: float = 0.05,
        contact_cutoff_angstrom: float = 8.0,
        contact_width_angstrom: float = 0.75,
    ):
        super().__init__()
        if learned_cv_dim < 0:
            raise ValueError("TM-JEPA learned_cv_dim must be non-negative")
        self.learned_cv_dim = int(learned_cv_dim)
        self.collective_variables = FixedBackboneCollectiveVariables(
            num_segments=num_segments,
            contact_cutoff_angstrom=contact_cutoff_angstrom,
            contact_width_angstrom=contact_width_angstrom,
        )
        fixed_dim = self.collective_variables.output_dim
        self.cv_dim = fixed_dim + self.learned_cv_dim
        self.condition_encoder = ProteinConditionEncoder(condition_dim)
        self.learned_projection = (
            nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, learned_cv_dim), nn.Tanh())
            if self.learned_cv_dim else None
        )
        self.target_learned_projection = copy.deepcopy(self.learned_projection)
        if self.target_learned_projection is not None:
            for parameter in self.target_learned_projection.parameters():
                parameter.requires_grad = False
        self.auxiliary_filter = CausalAuxiliaryFilter(
            self.cv_dim,
            auxiliary_dim,
            observation_time_scale_ps,
            hidden_dim=condition_dim,
        )
        self.dynamics = ThermodynamicMoriDynamics(
            self.cv_dim,
            condition_dim,
            auxiliary_dim=auxiliary_dim,
            memory_enabled=memory_enabled,
            base_step_ps=base_step_ps,
            max_rollout_steps=max_rollout_steps,
            ensemble_size=ensemble_size,
            diffusion_floor=diffusion_floor,
            diffusion_scale=diffusion_scale,
            relaxation_rate_scale=relaxation_rate_scale,
            coupling_scale=coupling_scale,
            equilibrium_noise_std=equilibrium_noise_std,
        )

    def train(self, mode: bool = True) -> ThermodynamicMoriJEPA:
        super().train(mode)
        if self.target_learned_projection is not None:
            self.target_learned_projection.eval()
        return self

    @torch.no_grad()
    def sync_target_projection(self) -> None:
        if self.learned_projection is not None and self.target_learned_projection is not None:
            self.target_learned_projection.load_state_dict(self.learned_projection.state_dict())

    @torch.no_grad()
    def update_target_projection(self, momentum: float) -> None:
        if self.learned_projection is None or self.target_learned_projection is None:
            return
        for target, online in zip(
            self.target_learned_projection.parameters(), self.learned_projection.parameters()
        ):
            target.data.mul_(momentum).add_(online.detach().data, alpha=1.0 - momentum)

    def forward(
        self,
        *,
        context_coords: torch.Tensor,
        future_coords: torch.Tensor,
        context_atom_mask: torch.Tensor,
        future_atom_mask: torch.Tensor,
        context_residue_mask: torch.Tensor,
        future_residue_mask: torch.Tensor,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        context_time_ps: torch.Tensor,
        horizon_time_ps: torch.Tensor,
        temperature_k: torch.Tensor | None,
        dataset_index: torch.Tensor | None,
        context_frame_states: torch.Tensor | None = None,
        future_frame_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_cv = self.collective_variables(
            context_coords,
            context_atom_mask,
            context_residue_mask,
            chain_id,
            residue_index,
        )
        future_cv = self.collective_variables(
            future_coords,
            future_atom_mask,
            future_residue_mask,
            chain_id,
            residue_index,
        )
        if self.learned_projection is not None:
            if context_frame_states is None or future_frame_states is None:
                raise ValueError("TM-JEPA learned CVs require context and future frame states")
            context_learned = self.learned_projection(context_frame_states)
            with torch.no_grad():
                future_learned = self.target_learned_projection(future_frame_states)
            context_cv = torch.cat([context_cv, context_learned], dim=-1)
            future_cv = torch.cat([future_cv, future_learned], dim=-1)
        context_mask = context_residue_mask.any(dim=2)
        future_mask = future_residue_mask.any(dim=2)
        condition = self.condition_encoder(
            aatype,
            residue_index,
            context_residue_mask.any(dim=1),
            dataset_index,
        )
        result = self.dynamics(
            context_cv,
            future_cv,
            context_mask,
            future_mask,
            context_time_ps,
            condition,
            temperature_k,
            horizon_time_ps,
            self.auxiliary_filter,
        )
        fixed_dim = self.collective_variables.output_dim
        result["tmjepa_current_physical_cv"] = self.collective_variables.to_physical(
            result["tmjepa_current_cv"][..., :fixed_dim]
        )
        result["tmjepa_target_physical_cv"] = self.collective_variables.to_physical(
            result["tmjepa_target_cv"][..., :fixed_dim]
        )
        result["tmjepa_sample_physical_cv"] = self.collective_variables.to_physical(
            result["tmjepa_samples"][..., :fixed_dim]
        )
        log_jacobian = self.collective_variables.log_abs_det_to_unconstrained(
            result["tmjepa_current_physical_cv"]
        )
        result["tmjepa_physical_free_energy"] = result["tmjepa_free_energy"] - (
            K_B_KCAL_PER_MOL_K * result["tmjepa_temperature_k"] * log_jacobian
        )
        result["tmjepa_context_cv_sequence"] = context_cv
        result["tmjepa_future_cv_sequence"] = future_cv
        result["tmjepa_context_mask"] = context_mask
        result["tmjepa_condition"] = condition
        if self.learned_cv_dim:
            result["tmjepa_learned_cv"] = context_cv[..., -self.learned_cv_dim :]
        return result
