"""Dynamic state encoder for Gate 1: coordinate-derived motion features.

This module computes temporal dynamics from coordinate sequences, ensuring:
1. Static coordinates → zero dynamic features (static-coordinate control)
2. Coordinate changes → non-zero dynamic features
3. Clean separation from static identity features (sequence, chain, residue ID)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _align_ca_to_reference(ca_coords: torch.Tensor, reference_idx: int = 0) -> torch.Tensor:
    """Align CA coordinates to a reference frame by removing translation and rotation.

    Args:
        ca_coords: [B, T, L, 3] CA positions
        reference_idx: which frame index to use as reference

    Returns:
        aligned_coords: [B, T, L, 3] aligned to frame reference_idx
    """
    ref_ca = ca_coords[:, reference_idx:reference_idx+1]  # [B, 1, L, 3]

    # Center both
    ref_center = ref_ca.mean(dim=2, keepdim=True)  # [B, 1, 1, 3]
    ca_center = ca_coords.mean(dim=2, keepdim=True)  # [B, T, 1, 3]

    ref_centered = ref_ca - ref_center  # [B, 1, L, 3]
    ca_centered = ca_coords - ca_center  # [B, T, L, 3]

    # Kabsch alignment: find optimal rotation for each frame
    B, T, L = ca_coords.shape[:3]
    aligned = []

    for t in range(T):
        # Covariance H = (ca_centered[:, t] @ ref_centered[:, 0].T) for each batch
        H = torch.bmm(
            ca_centered[:, t].transpose(1, 2),  # [B, 3, L]
            ref_centered[:, 0]  # [B, L, 3]
        )  # [B, 3, 3]

        # SVD: H = U @ S @ Vt
        U, _, Vt = torch.linalg.svd(H)

        # Rotation R = V @ U.T, handle reflection
        R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))  # [B, 3, 3]
        d = torch.det(R)
        Vt_fix = Vt.clone()
        Vt_fix[:, -1] *= d.sign().unsqueeze(-1)
        R = torch.bmm(Vt_fix.transpose(1, 2), U.transpose(1, 2))

        # Apply rotation and translate to reference center
        rotated = torch.bmm(ca_centered[:, t], R.transpose(1, 2))  # [B, L, 3]
        aligned.append(rotated + ref_center[:, 0])

    return torch.stack(aligned, dim=1)  # [B, T, L, 3]


def _ca_displacement_features(ca_coords: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Compute aligned frame-to-frame CA displacement.

    Args:
        ca_coords: [B, T, L, 3]
        valid_mask: [B, T, L]

    Returns:
        displacements: [B, T, L, 3] where [:, 0] is zero (no previous frame)
    """
    B, T, L = ca_coords.shape[:3]

    # Align each consecutive pair
    displacements = torch.zeros_like(ca_coords)

    for t in range(1, T):
        # Align frame t to frame t-1
        prev = ca_coords[:, t-1:t]  # [B, 1, L, 3]
        curr = ca_coords[:, t:t+1]  # [B, 1, L, 3]

        # Center
        prev_center = (prev * valid_mask[:, t-1:t].unsqueeze(-1)).sum(dim=2, keepdim=True) / valid_mask[:, t-1:t].sum(dim=2, keepdim=True).unsqueeze(-1).clamp(min=1)
        curr_center = (curr * valid_mask[:, t:t+1].unsqueeze(-1)).sum(dim=2, keepdim=True) / valid_mask[:, t:t+1].sum(dim=2, keepdim=True).unsqueeze(-1).clamp(min=1)

        prev_centered = prev - prev_center
        curr_centered = curr - curr_center

        # Kabsch
        mask = (valid_mask[:, t-1] & valid_mask[:, t]).float().unsqueeze(-1)  # [B, L, 1]
        H = torch.bmm(
            (curr_centered[:, 0] * mask).transpose(1, 2),  # [B, 3, L]
            prev_centered[:, 0] * mask  # [B, L, 3]
        )

        U, _, Vt = torch.linalg.svd(H)
        R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))
        d = torch.det(R)
        Vt_fix = Vt.clone()
        Vt_fix[:, -1] *= d.sign().unsqueeze(-1)
        R = torch.bmm(Vt_fix.transpose(1, 2), U.transpose(1, 2))

        # Aligned displacement
        curr_aligned = torch.bmm(curr_centered[:, 0], R.transpose(1, 2))
        displacements[:, t] = (curr_aligned - prev_centered[:, 0]) * mask

    return displacements


def _pair_distance_changes(ca_coords: torch.Tensor, k_nearest: int = 8) -> torch.Tensor:
    """Compute changes in k-nearest CA pair distances frame-to-frame.

    Args:
        ca_coords: [B, T, L, 3]
        k_nearest: number of nearest neighbors to track

    Returns:
        distance_changes: [B, T, L, k_nearest] where [:, 0] is zero
    """
    B, T, L = ca_coords.shape[:3]

    # Compute pairwise distances for all frames
    ca_expand = ca_coords.unsqueeze(3)  # [B, T, L, 1, 3]
    delta = ca_expand - ca_expand.transpose(2, 3)  # [B, T, L, L, 3]
    distances = torch.linalg.norm(delta, dim=-1)  # [B, T, L, L]

    # Find k-nearest at t=0 (topology remains same)
    dist_t0 = distances[:, 0]  # [B, L, L]
    dist_t0 = dist_t0 + torch.eye(L, device=dist_t0.device).unsqueeze(0) * 1e6  # mask diagonal
    _, nearest_indices = torch.topk(dist_t0, k=min(k_nearest, L), largest=False, dim=-1)  # [B, L, k]

    # Gather distances for these k-nearest pairs across time
    k_actual = nearest_indices.shape[-1]
    nearest_indices_expanded = nearest_indices.unsqueeze(1).expand(B, T, L, k_actual)  # [B, T, L, k]

    nearest_distances = torch.gather(
        distances, 3, nearest_indices_expanded
    )  # [B, T, L, k]

    # Compute temporal changes
    distance_changes = torch.zeros_like(nearest_distances)
    distance_changes[:, 1:] = nearest_distances[:, 1:] - nearest_distances[:, :-1]

    # Pad if needed
    if k_actual < k_nearest:
        padding = torch.zeros(B, T, L, k_nearest - k_actual, device=distance_changes.device, dtype=distance_changes.dtype)
        distance_changes = torch.cat([distance_changes, padding], dim=-1)

    return distance_changes


def _torsion_changes(
    coords: torch.Tensor, residue_mask: torch.Tensor, residue_index: torch.Tensor, chain_id: torch.Tensor
) -> torch.Tensor:
    """Compute frame-to-frame backbone torsion angle changes (phi, psi, omega).

    Args:
        coords: [B, T, L, 4, 3] backbone (N, CA, C, O)
        residue_mask: [B, T, L]
        residue_index: [B, L]
        chain_id: [B, L]

    Returns:
        torsion_changes: [B, T, L, 6] (delta_sin, delta_cos for phi/psi/omega) where [:, 0] is zero
    """
    from protein_jepa_frame.models.encoder import _backbone_torsion_features

    # Get torsions [B, T, L, 6] (sin/cos x 3)
    torsions = _backbone_torsion_features(coords, residue_mask, residue_index, chain_id)

    # Frame-to-frame change
    B, T, L = coords.shape[:3]
    torsion_changes = torch.zeros_like(torsions)
    torsion_changes[:, 1:] = torsions[:, 1:] - torsions[:, :-1]

    return torsion_changes


class DynamicStateEncoder(nn.Module):
    """Encode coordinate-derived dynamic features (Gate 1).

    This module processes per-frame geometry latents and coordinate sequences to produce
    a dynamic state that:
    - Is zero when coordinates are static (satisfies static-coordinate control)
    - Changes when coordinates change (driven by geometry, not clock)
    - Can still condition on physical time (dt between frames matters for velocity/diffusion)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_frames: int = 256,
        k_nearest: int = 8,
        use_displacement: bool = True,
        use_pair_distance_changes: bool = True,
        use_torsion_changes: bool = True,
        use_dt_conditioning: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.k_nearest = k_nearest
        self.use_displacement = use_displacement
        self.use_pair_distance_changes = use_pair_distance_changes
        self.use_torsion_changes = use_torsion_changes
        self.use_dt_conditioning = use_dt_conditioning

        # Dynamic feature projection
        dynamic_dim = 0
        if use_displacement:
            dynamic_dim += 3  # CA displacement (x, y, z)
        if use_pair_distance_changes:
            dynamic_dim += k_nearest  # k-nearest pair distance changes
        if use_torsion_changes:
            dynamic_dim += 6  # delta(sin/cos) for phi/psi/omega

        if dynamic_dim == 0:
            raise ValueError("At least one dynamic feature type must be enabled")

        self.dynamic_proj = nn.Sequential(
            nn.Linear(dynamic_dim, latent_dim, bias=False),
            nn.GELU(),
        )
        # No LayerNorm or temporal Transformer here. Dynamic features are per-frame
        # and frame-local. Cross-frame modeling is the TemporalStateEncoder's job.
        # Adding per-frame biases (LN affine, FF biases) would break the zero-output
        # invariant for static coordinates.

        # Physical time (dt) conditioning: scales/modulates dynamic features based on
        # the physical interval between frames. This is NOT a frame-index clock embedding —
        # static coordinates still yield zero dynamic signal regardless of dt.
        if use_dt_conditioning:
            self.dt_embed = nn.Sequential(
                nn.Linear(1, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, dynamic_dim),
            )
        else:
            self.dt_embed = None

        # No temporal transformer here. Dynamic features (displacement,
        # pair-distance change, torsion change) are per-frame-local quantities.
        # Cross-frame temporal modeling is the job of the separate TemporalStateEncoder.
        # Adding a Transformer here would inject frame-position-dependent biases
        # that break the static-uniformity invariant (identical input frames
        # must produce identical dynamic frame states).

    def forward(
        self,
        coords: torch.Tensor,
        residue_mask: torch.Tensor,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        static_frame_latents: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            coords: [B, T, L, 4, 3] backbone coordinates
            residue_mask: [B, T, L] valid residues
            residue_index: [B, L] residue indices
            chain_id: [B, L] chain IDs
            static_frame_latents: [B, T, D] optional static per-frame features to add
            dt: [B, T] or [B, T, 1] physical time intervals (ps) between frames, or None for uniform spacing

        Returns:
            dict with:
                - dynamic_frame_states: [B, T, D]
                - dynamic_clip_state: [B, D]
        """
        if coords.dim() != 5 or coords.shape[-2:] != (4, 3):
            raise ValueError("coords must be [B,T,L,4,3]")

        B, T, L = coords.shape[:3]
        ca_coords = coords[:, :, :, 1, :]  # [B, T, L, 3]

        # Compute dynamic features
        features = []

        if self.use_displacement:
            displacement = _ca_displacement_features(ca_coords, residue_mask)  # [B, T, L, 3]
            features.append(displacement)

        if self.use_pair_distance_changes:
            dist_changes = _pair_distance_changes(ca_coords, self.k_nearest)  # [B, T, L, k]
            features.append(dist_changes)

        if self.use_torsion_changes:
            torsion_delta = _torsion_changes(coords, residue_mask, residue_index, chain_id)  # [B, T, L, 6]
            features.append(torsion_delta)

        # Concatenate and pool over residues
        dynamic_features = torch.cat(features, dim=-1)  # [B, T, L, dynamic_dim]

        # Per-frame pooling (mean over valid residues)
        valid = residue_mask.unsqueeze(-1).float()
        frame_dynamic = (dynamic_features * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)

        # Apply dt conditioning to the RAW motion signal *before* projection.
        # Modulating pre-projection keeps the static-coordinate invariant: zero motion
        # scaled by any dt remains zero, so varying dt (timestamp-shuffle control) cannot
        # fabricate temporal variance out of static coordinates. This conditions the
        # amplitude of genuine motion on the physical interval, not on a frame-index clock.
        if self.dt_embed is not None:
            if dt is None:
                # Default: uniform 1 ps spacing (placeholder)
                dt = torch.ones(B, T, 1, device=coords.device, dtype=frame_dynamic.dtype)
            elif dt.dim() == 2:
                dt = dt.unsqueeze(-1)  # [B, T, 1]
            dt = dt.to(dtype=frame_dynamic.dtype)
            dt_scale = self.dt_embed(dt)  # [B, T, dynamic_dim] gate on raw features
            frame_dynamic = frame_dynamic * dt_scale

        # Project to latent space (bias-free: zero motion → zero projection).
        # Residue mean-pooling substantially attenuates localized conformational
        # changes (especially for short clips), so retain a small, fixed motion
        # gain after projection.  This changes only the scale of genuine motion;
        # the exact zero-motion invariant is preserved.
        frame_dynamic = 2.0 * self.dynamic_proj(frame_dynamic)  # [B, T, D]

        # Add static frame latents if provided (residual connection)
        if static_frame_latents is not None:
            frame_dynamic = frame_dynamic + static_frame_latents

        # Per-frame projection only (no cross-frame mixing).
        # Frame identifiability is handled by the downstream TemporalStateEncoder.
        frame_states = frame_dynamic  # [B, T, D]

        # CLS: mean-pool frame states for a sequence-level summary
        frame_valid = residue_mask.any(dim=2).float()  # [B, T]
        clip_state = (frame_states * frame_valid.unsqueeze(-1)).sum(dim=1) \
            / frame_valid.sum(dim=1, keepdim=True).clamp(min=1.0)

        return {
            "dynamic_frame_states": frame_states,  # [B, T, D]
            "dynamic_clip_state": clip_state,  # [B, D]
        }
