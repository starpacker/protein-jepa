"""Shared backbone geometry targets and validity masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _dihedral(a: torch.Tensor, b: torch.Tensor,
              c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Return a signed dihedral angle for tensors ending in ``[..., 3]``."""
    b0 = a - b
    b1 = F.normalize(c - b, dim=-1, eps=1.0e-8)
    b2 = d - c
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def backbone_torsion_targets(
    backbone_coords: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
    atom_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute phi/psi/omega targets and a per-angle validity mask.

    The returned tensors have shape ``[B, L, 7]``.  Only slots 0..2 are
    populated; chi slots are deliberately invalid when atom14 labels are not
    available.
    """
    if backbone_coords.ndim != 4 or backbone_coords.shape[-2:] != (4, 3):
        raise ValueError("backbone_coords must have shape [B, L, 4, 3]")
    B, L = backbone_coords.shape[:2]
    if residue_mask.shape != (B, L):
        raise ValueError("residue_mask must have shape [B, L]")
    if residue_index.shape != (B, L) or chain_id.shape != (B, L):
        raise ValueError("residue_index and chain_id must have shape [B, L]")

    angles = backbone_coords.new_zeros(B, L, 7)
    valid = torch.zeros(B, L, 7, dtype=torch.bool, device=backbone_coords.device)
    if L < 2:
        return angles, valid

    if atom_mask is None:
        atom_valid = residue_mask.unsqueeze(-1).expand(B, L, 4)
    else:
        if atom_mask.shape != (B, L, 4):
            raise ValueError("atom_mask must have shape [B, L, 4]")
        atom_valid = atom_mask.bool() & residue_mask.unsqueeze(-1)
    residue_valid = atom_valid.all(dim=-1)
    adjacent = (
        (chain_id[:, 1:] == chain_id[:, :-1])
        & (residue_index[:, 1:] - residue_index[:, :-1] == 1)
        & residue_valid[:, 1:]
        & residue_valid[:, :-1]
    )

    N = backbone_coords[..., 0, :]
    CA = backbone_coords[..., 1, :]
    C = backbone_coords[..., 2, :]
    angles[:, 1:, 0] = _dihedral(C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:])
    valid[:, 1:, 0] = adjacent
    angles[:, :-1, 1] = _dihedral(N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:])
    valid[:, :-1, 1] = adjacent
    angles[:, :-1, 2] = _dihedral(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:])
    valid[:, :-1, 2] = adjacent
    return angles, valid

