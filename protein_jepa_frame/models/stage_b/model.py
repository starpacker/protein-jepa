"""Stage B: Frozen Stage A encoders + canonical per-residue LaSP world model.

Gate 2 canonical implementation per audit plan Section 2 / Gate 2:
- Representation: Q:[B,L,8] per-residue dynamics, G=masked_mean(Q):[B,8]
- S:[B,L,16] per-residue finite bath (bath_dim=16, independent of state_dim)
- Single energy: A(Q, static_condition, T), G=P(Q) computed inside energy
- Forces: -dA/dQ via autograd on total energy (includes chain rule through G)
- Per-residue diffusion: D_i(q_i, G), positive diagonal
- Per-residue bath: S_i:[L,16], initialized once from history
- Integrator: Euler-Maruyama with C^T F bath forcing, exact-OU bath update
- Transition NLL: same coefficients as integrator

G = masked_mean(Q) is computed inside energy forward; caller never passes G independently.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.config import ModelConfig
from protein_jepa_frame.models.encoder import ProteinGeometryEncoder
from protein_jepa_frame.models.temporal import TemporalStateEncoder
from protein_jepa_frame.models.lasp import (
    ResidueMessagePassingPotential,
)
from protein_jepa_frame.models.stage_b.state import LatentState
from protein_jepa_frame.models.stage_b.transition import ClosedLoopTransition


class StageBLASP(nn.Module):
    """Canonical Stage B per-residue LaSP world model.

    Representation (matches audit plan P0-1):
      Q: [B, L, 8] per-residue dynamics
      S: [B, L, 16] per-residue finite bath
      G: [B, 8] = masked_mean(Q), recomputed inside energy
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        freeze_spatial_encoder: bool = True,
        freeze_temporal_encoder: bool = True,
        state_dim: int = 8,
        bath_dim: int = 16,
        condition_dim: int = 0,
        reference_temperature_k: float = 348.0,
        base_step_ps: float | None = None,
        use_bath: bool = True,
        history_aware: bool = True,
        energy_temperature_feature: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.bath_dim = bath_dim
        self.base_step_ps = float(
            getattr(config, "lspwm_base_step_ps", 100.0)
            if base_step_ps is None else base_step_ps
        )
        self.use_bath = use_bath
        self.history_aware = history_aware
        self.energy_temperature_feature = bool(energy_temperature_feature)

        # ── Frozen spatial encoder ──
        self.spatial_encoder = ProteinGeometryEncoder(
            latent_dim=config.latent_dim,
            depth=config.encoder_depth,
            num_heads=config.num_heads,
            num_rbf=config.num_rbf,
            dropout=config.dropout,
            max_residues=config.max_residues,
            orientation_features=config.orientation_features,
        )

        # ── Frozen temporal encoder ──
        temporal_depth = config.temporal_depth
        self.temporal_encoder = (
            TemporalStateEncoder(
                latent_dim=config.latent_dim,
                depth=temporal_depth,
                num_heads=config.num_heads,
                dropout=config.dropout,
                max_frames=config.max_frames,
                frame_pooling="mean",
                frame_pool_queries=4,
                time_conditioning=config.time_conditioning or "none",
                thermodynamic_conditioning=True,
                causal=False, causal_window=0,
                position_encoding="learned",
                delta_features=False, motion_depth=0, motion_stopgrad_input=False,
            )
            if temporal_depth > 0
            else None
        )

        # ── Per-residue projection: H → Q ──
        self.projection = nn.Sequential(
            nn.Linear(config.latent_dim, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, state_dim),
        )

        # ── Energy graph: per-residue GNN potential ──
        # Zero-width condition channels are supported by the official tensor
        # path.  Do not force a dummy channel here: older Stage-B checkpoints
        # were trained with condition_dim=0 and their state dict must load
        # exactly during evaluation.
        self.energy_graph = ResidueMessagePassingPotential(
            cv_dim=state_dim, condition_dim=condition_dim,
            hidden_dim=128, depth=3, num_rbf=8, rbf_max_dist=30.0,
            reference_temperature_k=reference_temperature_k,
            temperature_feature=self.energy_temperature_feature,
        )

        # ── Per-residue diffusion: D_i(q_i, G) → [B,L] ──
        self.diffusion_net = nn.Sequential(
            nn.Linear(state_dim + state_dim, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Use a scalar diagonal mobility so the overdamped SDE has no omitted
        # Ito divergence term.  Temperature scales D and therefore the noise
        # through the same FDT parameterization.
        # Honour the configured physical mobility scale.  The previous code
        # treated this learnable log-amplitude as an absolute D and silently
        # ignored ``lspwm_diffusion_scale``; with the default settings that
        # made D ~= 2.5e-3 instead of 1e-6 and let FDT noise dominate rollouts.
        self.diffusion_scale = float(getattr(config, "lspwm_diffusion_scale", 1.0e-6))
        if self.diffusion_scale <= 0.0:
            raise ValueError("lspwm_diffusion_scale must be positive")
        # Initialise softplus(log_diffusion) to 1.0, so a fresh model starts
        # exactly at the configured scale while retaining a trainable mobility
        # multiplier.
        self.log_diffusion = nn.Parameter(
            torch.tensor(math.log(math.expm1(1.0)))
        )

        # ── Per-residue bath initializer: H_last → S_0 [B,L,16] ──
        self.bath_initializer = nn.Sequential(
            nn.Linear(config.latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, bath_dim),
        )

        # ── Bath dynamics from Q ──
        self.bath_gamma_net = nn.Sequential(
            nn.Linear(state_dim, 32), nn.SiLU(),
            nn.Linear(32, bath_dim),
        )
        self.bath_coupling_net = nn.Sequential(
            nn.Linear(state_dim, 32), nn.SiLU(),
            nn.Linear(32, state_dim * bath_dim),
        )
        self.coupling_scale = float(
            getattr(config, "lspwm_coupling_scale", 1.0e-3)
        )
        if self.coupling_scale <= 0.0:
            raise ValueError("lspwm_coupling_scale must be positive")
        # Lower bound on the bath relaxation rate.  Keeps the exact-OU inverse
        # gain (1/gamma) bounded so long autonomous rollouts cannot diverge via
        # the bath back-action feedback loop (see compute_bath_params / step).
        self._bath_gamma_floor = float(
            getattr(config, "lspwm_bath_gamma_floor", 0.05)
        )

        # Physical-only transition wrapper (learned residual/noise removed
        # pending Gate 2.6 re-entry — see transition.py for rationale).
        self.closed_loop_transition = ClosedLoopTransition(
            state_dim=state_dim, env_dim=8,
        )

        self._freeze_flag_spatial = freeze_spatial_encoder
        self._freeze_flag_temporal = freeze_temporal_encoder
        self._reference_temperature_k = reference_temperature_k
        self._apply_freeze()

    def _apply_freeze(self):
        if self._freeze_flag_spatial:
            for p in self.spatial_encoder.parameters():
                p.requires_grad = False
            self.spatial_encoder.eval()
        if self._freeze_flag_temporal and self.temporal_encoder is not None:
            for p in self.temporal_encoder.parameters():
                p.requires_grad = False
            self.temporal_encoder.eval()

    # ── Spatial encoding ──

    def forward_spatial(
        self, coords, residue_mask, aatype, residue_index, chain_id,
    ) -> torch.Tensor:
        """coords [B,T,L,4,3] → H [B,T,L,D_H]"""
        with torch.set_grad_enabled(not self._freeze_flag_spatial):
            return self.spatial_encoder(
                coords=coords, residue_mask=residue_mask,
                aatype=aatype, residue_index=residue_index,
                chain_id=chain_id, token_mask=None,
            )

    # ── Projection and pooling ──

    def project_to_q(self, h_per_residue: torch.Tensor) -> torch.Tensor:
        """H [B,L,D_H] → Q [B,L,state_dim]"""
        return self.projection(h_per_residue)

    def compute_g(self, q: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        """G = masked_mean(Q): [B, state_dim]"""
        mask_f = residue_mask.unsqueeze(-1).float()
        return (q * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    # ── Energy and forces ──

    def compute_energy_and_forces(
        self, q, ca_distances=None, residue_mask=None, temperature_k=None,
        create_graph: bool | None = None,
    ) -> dict:
        """Per-residue energy via GNN, forces = -dA/dQ."""
        if create_graph is None:
            # Training needs higher-order derivatives so the force loss reaches
            # the energy parameters.  Evaluation only needs the force value.
            create_graph = self.training
        # Keep the force probe connected to the caller's Q graph.  Detaching Q
        # here makes the energy value and force path invisible to the H->Q
        # projection (and therefore to Stage-B training).  Only create a leaf
        # when this function is called with a tensor that does not require
        # gradients, as can happen in standalone diagnostics.
        q_for_energy = q if q.requires_grad else q.detach().requires_grad_(True)
        # Temperature is passed through the canonical API, but the learned
        # dimensionless potential itself is not rescaled by T.  Temperature
        # belongs in the mobility/noise (FDT) path; scaling U here would change
        # the drift without the matching diffusion correction.
        # Evaluation callers may be inside torch.no_grad(), but the force
        # probe itself still needs first-order autograd with respect to Q.
        with torch.enable_grad():
            energy = self.energy_graph(
                q_for_energy, ca_distances, residue_mask,
                condition=None,
                temperature_k=temperature_k,
            )
            mask_f = residue_mask.float() if residue_mask is not None else torch.ones_like(energy)
            energy_total = (energy * mask_f).sum()
            forces = -torch.autograd.grad(
                energy_total, q_for_energy,
                grad_outputs=torch.ones_like(energy_total),
                create_graph=create_graph, retain_graph=create_graph,
            )[0]
            if residue_mask is not None:
                forces = forces * residue_mask.unsqueeze(-1).float()
        if not create_graph:
            forces = forces.detach()
        return {"forces": forces, "energy_total": energy_total}

    # ── Diffusion ──

    def compute_per_residue_diffusion(
        self, q: torch.Tensor, g: torch.Tensor, temperature_k=None,
    ) -> torch.Tensor:
        """Dimensionless free-energy mobility D(T), broadcast per residue.

        A state-dependent D would require an Ito divergence correction.  The
        canonical path therefore uses a shared positive scalar and scales it
        with temperature, keeping drift and noise in one FDT parameterization.
        """
        B, L = q.shape[:2]
        # Keep the legacy diffusion head trainable for checkpoint/gradient
        # compatibility, but stop its dependence on Q in the SDE coefficient.
        # This makes it a per-protein mobility and avoids an omitted Ito
        # divergence term while still allowing likelihood calibration.
        g_exp = g.detach().unsqueeze(1).expand(B, L, -1)
        q_det = q.detach()
        raw_head = self.diffusion_net(
            torch.cat([q_det, g_exp], dim=-1)
        ).squeeze(-1).mean(dim=1)
        diffusion = self.diffusion_scale * (
            F.softplus(self.log_diffusion + 0.01 * raw_head) + 1e-6
        )
        if temperature_k is None:
            temp_scale = q.new_ones(B)
        else:
            t = torch.as_tensor(temperature_k, device=q.device, dtype=q.dtype)
            if t.ndim == 0:
                temp_scale = t / self._reference_temperature_k
            else:
                if t.numel() == 1:
                    temp_scale = t.reshape(1).expand(B) / self._reference_temperature_k
                else:
                    temp_scale = t.reshape(B, -1)[:, -1] / self._reference_temperature_k
        return (diffusion * temp_scale.reshape(B)).unsqueeze(1).expand(B, L)

    # ── Bath ──

    def initialize_bath(self, h_last, residue_mask) -> torch.Tensor:
        """H_last [B,L,D_H] → S_0 [B,L,bath_dim]. Called once."""
        s0 = self.bath_initializer(h_last)
        if residue_mask is not None:
            s0 = s0 * residue_mask.unsqueeze(-1).float()
        return s0

    def compute_bath_params(self, q, residue_mask) -> tuple:
        """Q → gamma:[B,L,bath_dim], coupling:[B,L,D,bath_dim]"""
        B, L, D = q.shape
        # Gamma is the bath relaxation rate.  The exact-OU back-action term in
        # ``step`` scales with ``1/gamma``, so a gamma that collapses toward 0
        # creates a divergent positive-feedback loop (bath state -> system
        # forcing -> larger Q -> smaller gamma -> larger bath state) that blows
        # the autonomous rollout up to NaN after a few hundred substeps.  The
        # floor is raised from 0.01 to ``bath_gamma_floor`` so that the inverse
        # gain stays bounded (<= 1/floor) for every residue at every step.  This
        # is a property of the learned dynamics and applies identically during
        # training and evaluation; it uses no future information.
        gamma = F.softplus(self.bath_gamma_net(q)) + self._bath_gamma_floor
        coupling = self.bath_coupling_net(q).reshape(B, L, D, self.bath_dim)
        coupling = coupling * self.coupling_scale
        if residue_mask is not None:
            m3 = residue_mask.unsqueeze(-1).float()
            gamma = gamma * m3
            coupling = coupling * m3.unsqueeze(-1)
        return gamma, coupling

    def compute_bath_forcing(self, coupling, s) -> torch.Tensor:
        """C^T·S: coupling:[B,L,D,M] @ s:[B,L,M] → [B,L,D]"""
        return torch.einsum("blij,blj->bli", coupling, s)

    # ── Batch parsing ──

    def _parse_batch(self, batch: dict) -> tuple:
        if "context" in batch and isinstance(batch["context"], dict):
            ctx_r = batch["context"]
            ctx = {k: ctx_r.get(k) for k in
                   ["coords","residue_mask","aatype","residue_index","chain_id","temperature_k","time_ps"]}
            ctx_coords = ctx_r["coords"]
            ctx_res_mask = ctx_r["residue_mask"]
            fut_r = batch.get("future", {})
            fut = {"coords": fut_r.get("coords")} if fut_r.get("coords") is not None else {}
        elif "context_coords" in batch:
            ctx_coords = batch["context_coords"]
            ctx_res_mask = batch["context_residue_mask"]
            ctx = {
                "coords": ctx_coords, "residue_mask": ctx_res_mask,
                "aatype": batch.get("context_aatype", batch.get("aatype")),
                "residue_index": batch.get("context_residue_index", batch.get("residue_index")),
                "chain_id": batch.get("context_chain_id", batch.get("chain_id")),
                "temperature_k": batch.get("context_temperature_k", batch.get("temperature_k")),
                "time_ps": batch.get("context_time_ps"),
            }
            fut = {"coords": batch.get("future_coords")} if "future_coords" in batch else {}
        else:
            raise KeyError(f"batch missing context coords; keys: {list(batch.keys())}")
        return ctx, fut, ctx_coords, ctx_res_mask

    def _get_topology(self, ctx, B):
        aatype = ctx["aatype"]
        res_idx = ctx["residue_index"]
        chain_id = ctx.get("chain_id", torch.zeros_like(aatype))
        if aatype.dim() == 1:
            aatype = aatype.unsqueeze(0).expand(B, -1)
            res_idx = res_idx.unsqueeze(0).expand(B, -1)
            chain_id = chain_id.unsqueeze(0).expand(B, -1)
        return aatype, res_idx, chain_id

    # ── Canonical state encoding (Call BEFORE encode_state) ──
    # ============================================================
    # encode_state() is the SINGLE entry point for obtaining
    # (q, h_last, residue_mask, bath_s0, ca_distances) from raw coords.
    # Training (forward()) and ALL evaluation scripts MUST use it so
    # temporal-history agreement is guaranteed.

    def encode_state(
        self,
        coords: torch.Tensor,        # [B, T, L, 4, 3]
        residue_mask: torch.Tensor,  # [B, T, L] or [B, L]
        aatype: torch.Tensor,        # [B, L]
        residue_index: torch.Tensor, # [B, L]
        chain_id: torch.Tensor,      # [B, L]
        time_ps: torch.Tensor = None,       # [B, T] optional
        temperature_k=None,
        *,
        use_temporal: bool = True,
        return_bath: bool = True,
        return_ca_distances: bool = True,
        return_forces: bool = False,
    ) -> dict:
        """Canonical state encoding: coords → (q, h_last, s0, ca_d, ...).

        ALWAYS applies the same temporal encoder path (when history_aware
        and a temporal sequence is available) so the resulting q matches
        exactly what forward() produces during training.

        Returns dict with keys:
            q:              [B, L, 8]    per-residue latent state
            h_last:         [B, L, D_H]  last-frame spatial embedding
            residue_mask:   [B, L]       2D residue mask
            bath_s0:        [B, L, 16]   initial bath state (if return_bath)
            ca_distances:   [B, L, L]    CA pairwise distances (if return_ca_distances)
            forces:         [B, L, 8]    conservative forces (if return_forces)
        """
        B, T, L = coords.shape[:3]
        device = coords.device

        # Reshape mask if needed
        if residue_mask.dim() == 3:
            residue_mask_2d = residue_mask[:, -1]
        else:
            residue_mask_2d = residue_mask

        # Reshape topology
        if aatype.dim() == 1:
            aatype = aatype.unsqueeze(0).expand(B, -1)
        if residue_index is None:
            residue_index = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        elif residue_index.dim() == 1:
            residue_index = residue_index.unsqueeze(0).expand(B, -1)
        if chain_id is None:
            chain_id = torch.zeros(B, L, dtype=torch.long, device=device)
        elif chain_id.dim() == 1:
            chain_id = chain_id.unsqueeze(0).expand(B, -1)

        # ── Spatial encoding ──
        context_latents = self.forward_spatial(
            coords, residue_mask, aatype, residue_index, chain_id,
        )

        # ── Q projection with temporal history ──
        if use_temporal and self.history_aware and self.temporal_encoder is not None and T > 1:
            with torch.set_grad_enabled(not self._freeze_flag_temporal):
                temporal_out = self.temporal_encoder(
                    context_latents, residue_mask,
                    time_ps=time_ps,
                    temperature_k=temperature_k,
                )
                frame_state = temporal_out["frame_states"][:, -1]
                h_last = context_latents[:, -1] + frame_state.unsqueeze(1)
        else:
            h_last = context_latents[:, -1]

        q = self.project_to_q(h_last)

        result = {"q": q, "h_last": h_last, "residue_mask": residue_mask_2d}

        # ── CA distances ──
        if return_ca_distances:
            anchor_backbone = coords[:, -1]
            ca_positions = anchor_backbone[..., 1, :]
            result["ca_distances"] = torch.cdist(ca_positions, ca_positions)

        # ── Bath initialization ──
        if return_bath and self.use_bath:
            result["bath_s0"] = self.initialize_bath(h_last, residue_mask_2d)

        # ── Forces (optional, for NLL paths) ──
        if return_forces and return_ca_distances:
            force_out = self.compute_energy_and_forces(
                q, result["ca_distances"], residue_mask_2d, temperature_k,
            )
            result["forces"] = force_out["forces"]
            result["energy_total"] = force_out.get("energy_total")

        return result

    # ── Main forward ──

    def forward(
        self, batch: dict, *,
        compute_energy: bool = True,
        compute_dynamics: bool = True,
        compute_bath: bool = True,
    ) -> dict:
        """Returns per-residue Q:[B,L,8], S:[B,L,16], forces:[B,L,8], diffusion:[B,L].

        Uses encode_state() internally — Q encoding is identical to eval scripts.
        """
        ctx, fut, ctx_coords, ctx_res_mask = self._parse_batch(batch)
        device = ctx_coords.device
        B, T, L = ctx_coords.shape[:3]

        aatype, res_idx, chain_id = self._get_topology(ctx, B)

        temperature_k = ctx.get("temperature_k")
        if temperature_k is not None and temperature_k.dim() > 1:
            temperature_k = temperature_k[:, -1]
        residue_mask = ctx_res_mask[:, -1] if ctx_res_mask.dim() == 3 else ctx_res_mask

        # ── Canonical state encoding (matches eval scripts exactly) ──
        state = self.encode_state(
            ctx_coords, ctx_res_mask, aatype, res_idx, chain_id,
            time_ps=ctx.get("time_ps"),
            temperature_k=temperature_k,
            use_temporal=self.history_aware,
            return_bath=compute_bath,
            return_ca_distances=compute_energy,
            return_forces=False,
        )
        q = state["q"]

        g = self.compute_g(q, residue_mask)

        outputs = {"q": q, "g": g, "residue_mask": residue_mask}

        ca_distances = state.get("ca_distances")
        if ca_distances is None:
            anchor_backbone = ctx_coords[:, -1]
            ca_positions = anchor_backbone[..., 1, :]
            ca_distances = torch.cdist(ca_positions, ca_positions)

        if compute_energy:
            force_outputs = self.compute_energy_and_forces(
                q, ca_distances, residue_mask, temperature_k,
            )
            outputs.update(force_outputs)

            diffusion_diag = self.compute_per_residue_diffusion(q, g, temperature_k)
            outputs["diffusion"] = diffusion_diag

        if compute_bath and self.use_bath:
            s0 = state.get("bath_s0", self.initialize_bath(state["h_last"], residue_mask))
            gamma, coupling = self.compute_bath_params(q, residue_mask)
            outputs.update({"bath_s0": s0, "bath_gamma": gamma, "bath_coupling": coupling})

        if compute_dynamics and ("future_coords" in batch or fut.get("coords") is not None):
            fut_coords = fut.get("coords") if fut.get("coords") is not None else batch.get("future_coords")
            if fut_coords is not None:
                fut_res_mask = batch.get("future_residue_mask")
                fut_time_ps = batch.get("future_time_ps")
                with torch.no_grad():
                    # ── CANONICAL: encode future endpoint through THE SAME
                    # encode_state path as context, with use_temporal=True.
                    # This guarantees q_next and q are in the same
                    # representation space.  We only need q_next from the
                    # last future frame (the target endpoint), so we
                    # discard intermediate frames in the returned dict.
                    future_state = self.encode_state(
                        fut_coords, fut_res_mask, aatype, res_idx, chain_id,
                        time_ps=fut_time_ps,
                        temperature_k=temperature_k,
                        use_temporal=True,
                        return_bath=False,
                        return_ca_distances=False,
                        return_forces=False,
                    )
                    q_next = future_state["q"].detach()
                    outputs["q_next"] = q_next

                    # ── CANONICAL dt: last-context-frame → last-future-frame.
                    # This matches the definition "Q_future = encode at the
                    # future endpoint" — the transition time is the gap
                    # between the two endpoints.
                    ctx_time = batch.get("context_time_ps")
                    if ctx_time is not None and fut_time_ps is not None:
                        # dt = future_endpoint - context_endpoint
                        actual_dt_ps = fut_time_ps[:, -1] - ctx_time[:, -1]
                        actual_dt_ps = actual_dt_ps.clamp(min=1e-6)
                    else:
                        actual_dt_ps = torch.full((B,), self.base_step_ps, device=device)
                    outputs["transition_dt_ps"] = actual_dt_ps.detach()
                if "forces" in outputs and "diffusion" in outputs:
                    bath_forcing = None
                    if self.use_bath and "bath_coupling" in outputs:
                        s_current = outputs.get("bath_s0")
                        if s_current is not None:
                            bath_forcing = self.compute_bath_forcing(
                                outputs["bath_coupling"], s_current
                            )

                    nll = self._transition_nll(
                        q, q_next, outputs["forces"], outputs["diffusion"],
                        actual_dt_ps, residue_mask, bath_forcing=bath_forcing,
                    )
                    outputs["transition_nll"] = nll

        return outputs

    def _transition_nll(
        self, q, q_next, forces, diffusion, dt_ps, residue_mask,
        bath_forcing=None,
    ) -> torch.Tensor:
        """Gate 2.2: NLL includes bath forcing in the mean, matching integrator.

        Mean: q + dt·D·F + dt·(C^T·S) if bath_forcing provided
        Cov: 2·dt·D
        """
        B, L, D = q.shape
        dt = dt_ps.to(dtype=q.dtype, device=q.device)
        if dt.dim() == 0:
            dt = dt.expand(B)
        dt_s = dt.clamp(min=1e-6).unsqueeze(1).unsqueeze(1)
        D = diffusion.unsqueeze(-1)
        mean = q + dt_s * D * forces
        if bath_forcing is not None:
            mean = mean + dt_s * bath_forcing
        cov = (2.0 * dt_s * D).clamp(min=1e-8)
        log_var = torch.log(cov)
        diff = q_next - mean
        # A raw continuous-density NLL can be negative when the variance is
        # below one.  Use a smooth positive transform rather than an element-
        # wise hard floor: the latter silently saturates common small-variance
        # batches and cuts gradients to diffusion, bath, and the Q projection.
        raw_nll = 0.5 * (log_var + diff.pow(2) / cov)
        nll = F.softplus(raw_nll) + 1.0e-6
        mask_f = residue_mask.unsqueeze(-1).float()
        return (nll * mask_f).sum() / mask_f.sum().clamp(min=1)

    # ── Dynamics (per-residue) ──

    def step(
        self, q, s=None, ca_distances=None, residue_mask=None,
        temperature_k=None, dt_ps=None, stochastic=True,
        static_condition=None,
        max_substep_dt_ps: float = 20.0,
        return_diagnostics: bool = False,
        noise_scale: float = 1.0,
    ) -> tuple:
        """One integration step with exact sub-stepping for stability.

        q_next = q + dt*D*F + dt*C^T*s + noise_scale * sqrt(2*dt*D)*xi
        s_next = exp(-Gamma*dt)*s + sqrt(1-exp(-2*Gamma*dt))*xi_bath

        noise_scale ∈ [0, 1] linearly scales the noise amplitude.  During
        training rollouts this is ramped from 0→1 as |F| grows, preventing
        the rollout gradient from being swamped by noise when forces are
        still small.

        Sub-stepping splits dt into steps of at most max_substep_dt_ps.
        There is NO cap on the number of substeps — callers must ensure
        dt_ps is reasonable (≤ 100 ps is safe for all practical datasets).

        Args:
            max_substep_dt_ps: max dt per substep (20 ps = ~1.5% Euler error)
            return_diagnostics: if True, return (q, s, diagnostics) tuple
        """
        B, L, D = q.shape
        device = q.device
        dt_val = dt_ps if dt_ps is not None else self.base_step_ps

        if ca_distances is None:
            ca_distances = torch.zeros(B, L, L, device=device, dtype=q.dtype)

        # ── Sub-stepping: use EXACT number of substeps (no cap) ──
        # Gate 2.6: the old code capped n_sub at 10, turning 4000 ps into
        # 10 × 400 ps steps instead of 200 × 20 ps.  The cap is removed so
        # that accuracy is always bounded.  Callers MUST use a reasonable
        # dt_ps (≤ 100 ps) to keep per-step cost manageable; if a single
        # call requires >1000 substeps, that is a configuration error.
        n_sub = max(1, int(math.ceil(float(dt_val) / float(max_substep_dt_ps))))
        sub_dt = dt_val / n_sub

        if n_sub > 500:
            import warnings
            warnings.warn(
                f"step() requires {n_sub} substeps for dt={dt_val:.1f}ps "
                f"(substep dt={sub_dt:.1f}ps). This many substeps may be "
                f"impractical; reduce dt_ps or increase max_substep_dt_ps.",
                RuntimeWarning,
            )

        for sub_i in range(n_sub):
            force_out = self.compute_energy_and_forces(
                q, ca_distances, residue_mask, temperature_k,
            )
            forces = force_out["forces"]
            g = self.compute_g(q, residue_mask)
            diff_diag = self.compute_per_residue_diffusion(q, g, temperature_k)

            D = diff_diag.unsqueeze(-1)
            drift = sub_dt * D * forces

            if s is not None:
                gamma, coupling = self.compute_bath_params(q, residue_mask)
                # System forcing from bath:  C @ S  (maps M → D)
                sys_forcing = self.compute_bath_forcing(coupling, s)
                drift = drift + sub_dt * sys_forcing

                # Exact-OU bath update with back-action: C^T @ F
                exp_decay = torch.exp(-gamma * sub_dt)
                s_next = exp_decay * s
                back_action = torch.einsum("blij,bli->blj", coupling, forces)
                inv_gamma = 1.0 / gamma.clamp(min=1e-8)
                s_next = s_next + (1.0 - exp_decay) * inv_gamma * back_action
                if stochastic:
                    s_std = torch.sqrt(1.0 - exp_decay**2 + 1e-12)
                    s_next = s_next + s_std * torch.randn_like(s)
                if residue_mask is not None:
                    s_next = s_next * residue_mask.unsqueeze(-1).float()
                s = s_next

            # Q update
            q_next = q + drift
            if stochastic:
                noise_std = noise_scale * torch.sqrt(2.0 * sub_dt * diff_diag.clamp(min=1e-12)).unsqueeze(-1)
                q_next = q_next + noise_std * torch.randn_like(q)

            if residue_mask is not None:
                m = residue_mask.unsqueeze(-1).float()
                q_next = q_next * m + q * (1.0 - m)

            q = q_next

        if return_diagnostics:
            diagnostics = {
                "n_substeps": n_sub,
            }
            return q, s, diagnostics

        return q, s

    def environment_state(self, temperature_k, batch_size: int, device, dtype):
        """Return the explicit environment component of the canonical state."""
        e = torch.zeros(batch_size, 8, device=device, dtype=dtype)
        if temperature_k is not None:
            t = torch.as_tensor(temperature_k, device=device, dtype=dtype)
            if t.ndim == 0:
                t = t.expand(batch_size)
            elif t.ndim > 1:
                t = t.reshape(batch_size, -1)[:, -1]
            e[:, 0] = t / self._reference_temperature_k - 1.0
        return e

    def transition(
        self, q, s, *, ca_distances, residue_mask,
        temperature_k=None, dt_ps=None, stochastic=False,
        max_substep_dt_ps: float = 20.0,
        noise_scale: float = 1.0,
    ):
        """One actual closed-loop transition on ``(q, s, e)``."""
        e = self.environment_state(
            temperature_k, q.shape[0], q.device, q.dtype,
        )
        state = LatentState(q=q, m=s, e=e)
        result = self.closed_loop_transition(
            state, dt_ps if dt_ps is not None else self.base_step_ps,
            self, ca_distances, residue_mask, temperature_k, stochastic,
            max_substep_dt_ps=max_substep_dt_ps,
            noise_scale=noise_scale,
        )
        return result.next_state.q, result.next_state.m, result.diagnostics

    def rollout(
        self, q_init, s_init=None, n_steps=10,
        ca_distances=None, residue_mask=None,
        temperature_k=None, dt_ps=None, stochastic=True,
    ) -> tuple:
        """q_init:[B,L,D], s_init:[B,L,M] → q_traj:[B,n+1,L,D], s_traj:[B,n+1,L,M]"""
        B, L, D = q_init.shape
        device = q_init.device
        q_traj = torch.zeros(B, n_steps+1, L, D, device=device, dtype=q_init.dtype)
        q_traj[:, 0] = q_init
        s_traj = None
        if s_init is not None:
            s_traj = torch.zeros(B, n_steps+1, L, self.bath_dim, device=device, dtype=s_init.dtype)
            s_traj[:, 0] = s_init

        q, s = q_init, s_init
        for i in range(n_steps):
            q, s = self.step(
                q, s, ca_distances=ca_distances, residue_mask=residue_mask,
                temperature_k=temperature_k, dt_ps=dt_ps, stochastic=stochastic,
            )
            q_traj[:, i+1] = q
            if s_traj is not None and s is not None:
                s_traj[:, i+1] = s

        return q_traj, s_traj

    def forward_encode_only(
        self, coords, residue_mask, aatype, residue_index, chain_id,
    ) -> tuple:
        """Encode single frame → Q:[B,L,8], ca_distances:[B,L,L]."""
        B, T, L = coords.shape[:3]
        h = self.forward_spatial(coords, residue_mask, aatype, residue_index, chain_id)
        h_last = h[:, -1]
        q = self.project_to_q(h_last)
        ca = coords[:, -1, :, 1, :]
        ca_distances = torch.cdist(ca, ca)
        return q, ca_distances
