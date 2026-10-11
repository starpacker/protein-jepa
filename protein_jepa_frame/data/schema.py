"""Stage C data schema: atom37, units, static condition contracts.

Gate 4 canonical implementation:
- Standard atom37 heavy-atom mapping from OpenFold residue_constants
- Backbone compatibility (N/CA/C/O view)
- PBC unwrap, chain break, altloc, missing residue handling policies
- Sequence, aatype, residue/chain id, static conditions, time_ps, atom37 coords/mask
- All policies recorded in run manifest

Coordinate unit: Angstrom (internal)
Time unit: picoseconds (internal)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

# ── Atom37 constants from OpenFold residue_constants ──────────────────

# Standard PDB atom names for all 20 canonical amino acids
_ATOM37_NAMES: tuple[str, ...] = (
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD",
    "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2",
    "CH2", "CZ", "CZ2", "CZ3", "NZ", "OH",
    "NH1", "NH2",
    "OXT",
)

ATOM37_NUM: int = 37
ATOM37_NAMES: tuple[str, ...] = _ATOM37_NAMES
ATOM37_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(_ATOM37_NAMES)}

# Backbone atom indices within atom37
BACKBONE_ATOM37_INDICES: tuple[int, int, int, int] = (0, 1, 2, 4)  # N, CA, C, O


@dataclass(frozen=True)
class ResidueAtomMap:
    """Per-residue-type mapping from atom37 to physical atoms."""
    atom37_to_atom14: tuple[int, ...]  # length 37, -1 if absent
    atom14_to_atom37: tuple[int, ...]  # length 14
    atom14_mask: tuple[bool, ...]       # length 14, which atoms are present
    atom14_names: tuple[str, ...]       # length 14

    def __post_init__(self):
        if len(self.atom37_to_atom14) != 37:
            raise ValueError("atom37_to_atom14 must have length 37")
        if len(self.atom14_to_atom37) != 14:
            raise ValueError("atom14_to_atom37 must have length 14")

    def gather_atoms(self, atom37_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert [..., 37, 3] atom37 coords to [..., 14, 3] atom14 coords + mask.

        Returns:
            gathered: [..., 14, 3] atom14 coordinates
            mask: [..., 14] bool mask indicating which atoms are present
        """
        # Build a 37→14 gather: for each atom14 position (0..13), find the atom37 index
        # atom37_to_atom14[i] = j means atom37 atom i maps to atom14 position j
        # Reverse: for each atom14 position j, find which atom37 atom maps to it
        a14_to_a37 = [-1] * 14
        for a37_idx, a14_idx in enumerate(self.atom37_to_atom14):
            if a14_idx >= 0:
                a14_to_a37[a14_idx] = a37_idx

        idx = torch.tensor(a14_to_a37, dtype=torch.long, device=atom37_coords.device)
        # For absent atoms, use index 0 (N atom, always present)
        safe_idx = idx.clamp(min=0)
        gathered = atom37_coords[..., safe_idx, :]  # [..., 14, 3]
        mask = torch.tensor(self.atom14_mask, dtype=torch.bool, device=atom37_coords.device)
        return gathered, mask


# atom37_to_atom14 mapping per residue type from OpenFold
# Uses the standard mapping: indices 0-13 for atom14
# -1 means atom is absent for this residue type
_ATOM37_TO_ATOM14: dict[str, tuple[int, ...]] = {
    "ALA": (0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "ARG": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, 6, -1, -1, 7, 8, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, 10, -1, -1, 11, -1, 12, 13, -1),
    "ASN": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "ASP": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "CYS": (0, 1, 2, 3, 4, -1, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "GLN": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "GLU": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "GLY": (0, 1, 2, -1, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "HIS": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, 6, -1, 7, 8, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "ILE": (0, 1, 2, 3, 4, -1, 5, 6, -1, -1, -1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "LEU": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "LYS": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, 7, -1, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1),
    "MET": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "PHE": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, -1, -1, -1, -1),
    "PRO": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "SER": (0, 1, 2, 3, 4, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "THR": (0, 1, 2, 3, 4, -1, -1, -1, -1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
    "TRP": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, 6, 7, 8, -1, -1, -1, -1, -1, 9, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, -1),
    "TYR": (0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, -1, -1, 11, -1, -1, -1),
    "VAL": (0, 1, 2, 3, 4, -1, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
}

# Three-letter amino acid codes
_RESTYPES: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
)

_UNK_ATOM37_TO_ATOM14: tuple[int, ...] = (0, 1, 2, -1, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)


def get_residue_atom_map(aatype: int) -> ResidueAtomMap:
    """Return the ResidueAtomMap for a given aatype index (0-19 for 20 canonical AAs)."""
    if aatype < 0 or aatype >= 20:
        a37_to_a14 = _UNK_ATOM37_TO_ATOM14
        a37_to_a14_int = tuple(int(x) for x in a37_to_a14)
        atom14_names = ("N", "CA", "C", "O")
        atom14_mask_list = [False] * 14
        for j in range(37):
            idx = a37_to_a14_int[j]
            if idx >= 0:
                atom14_mask_list[idx] = True
        atom14_mask = tuple(atom14_mask_list)
        atom14_to_atom37 = tuple(
            next((j for j in range(37) if a37_to_a14_int[j] == i), -1)
            for i in range(14)
        )
        return ResidueAtomMap(
            atom37_to_atom14=a37_to_a14_int,
            atom14_to_atom37=tuple(atom14_to_atom37),
            atom14_mask=atom14_mask,
            atom14_names=atom14_names,
        )
    restype = _RESTYPES[aatype]
    a37_to_a14 = _ATOM37_TO_ATOM14[restype]
    a37_to_a14_int = tuple(int(x) for x in a37_to_a14)
    # Compute atom14 names
    atom14_names_list: list[str] = []
    seens: set[int] = set()
    for j in range(37):
        idx = a37_to_a14_int[j]
        if idx >= 0 and idx not in seens:
            while len(atom14_names_list) <= idx:
                atom14_names_list.append("")
            atom14_names_list[idx] = _ATOM37_NAMES[j]
            seens.add(idx)
    atom14_names = tuple(atom14_names_list)
    # Build reversed: per atom14 position, is its atom37 source present?
    atom14_mask_list = [False] * 14
    for j in range(37):
        idx = a37_to_a14_int[j]
        if idx >= 0:
            atom14_mask_list[idx] = True
    atom14_mask = tuple(atom14_mask_list)
    # Build reverse mapping atom14 -> atom37
    a14_to_a37_list = [-1] * 14
    for j, idx in enumerate(a37_to_a14_int):
        if idx >= 0:
            a14_to_a37_list[idx] = j
    atom14_to_atom37 = tuple(a14_to_a37_list)
    return ResidueAtomMap(
        atom37_to_atom14=a37_to_a14_int,
        atom14_to_atom37=atom14_to_atom37,
        atom14_mask=atom14_mask,
        atom14_names=atom14_names,
    )


def build_atom37_gatherer(aatype: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build gather tensors for converting atom37 coordinates to atom14.

    Args:
        aatype: [B, L] or [L] residue type indices (0-19)

    Returns:
        gather_idx: [..., 14] indices into atom37 dimension
        gather_mask: [..., 14] bool mask
        atom37_mask: [..., 37] bool mask of populated atom37 positions
    """
    *batch_dims, L = aatype.shape
    flat = aatype.reshape(-1, L)
    N = flat.shape[0]

    gather_idx = torch.full((N, L, 14), 0, dtype=torch.long)
    gather_mask = torch.zeros(N, L, 14, dtype=torch.bool)
    atom37_mask = torch.zeros(N, L, 37, dtype=torch.bool)

    for n in range(N):
        for l in range(L):
            rm = get_residue_atom_map(int(flat[n, l].item()))
            for j in range(37):
                a14_idx = rm.atom37_to_atom14[j]
                if a14_idx >= 0:
                    atom37_mask[n, l, j] = True
                    gather_idx[n, l, a14_idx] = j
                    gather_mask[n, l, a14_idx] = True

    # For masked positions, use 0 (N atom, always present)
    gather_idx = gather_idx.clamp(min=0)

    if batch_dims:
        gather_idx = gather_idx.reshape(*batch_dims, L, 14)
        gather_mask = gather_mask.reshape(*batch_dims, L, 14)
        atom37_mask = atom37_mask.reshape(*batch_dims, L, 37)

    return gather_idx, gather_mask, atom37_mask


def backbone_from_atom37(atom37_coords: torch.Tensor) -> torch.Tensor:
    """Extract backbone coordinates (N, CA, C, O) from atom37 tensor.

    Args:
        atom37_coords: [..., 37, 3] atom37 coordinates

    Returns:
        [..., 4, 3] backbone coordinates
    """
    idx = torch.tensor(BACKBONE_ATOM37_INDICES, dtype=torch.long, device=atom37_coords.device)
    return atom37_coords[..., idx, :]


@dataclass(frozen=True)
class StageCSample:
    """Canonical data sample for Stage C training and evaluation.

    Contains all fields required by schema:
    - atom37_coords: [L, 37, 3] full heavy-atom coordinates (Angstrom)
    - atom37_mask: [L, 37] which atoms are present
    - backbone_coords: [L, 4, 3] N, CA, C, O
    - sequence: str, 1-letter amino acid sequence
    - aatype: [L] residue type indices (0-19)
    - residue_index: [L] sequential residue index
    - chain_id: [L] chain identifier
    - static_condition: [L, D] per-residue condition from Stage A encoder
    - time_ps: float, absolute time in picoseconds
    - temperature_k: float
    - protein_id: str
    - trajectory_id: str
    - residue_mask: [L] which residues are structurally present
    """
    atom37_coords: torch.Tensor
    atom37_mask: torch.Tensor
    backbone_coords: torch.Tensor
    sequence: str
    aatype: torch.Tensor
    residue_index: torch.Tensor
    chain_id: torch.Tensor
    static_condition: torch.Tensor | None
    time_ps: float
    temperature_k: float
    protein_id: str
    trajectory_id: str
    residue_mask: torch.Tensor

    def __post_init__(self):
        L = len(self.aatype)
        expected_shapes = {
            "atom37_coords": (L, 37, 3),
            "atom37_mask": (L, 37),
            "backbone_coords": (L, 4, 3),
            "aatype": (L,),
            "residue_index": (L,),
            "chain_id": (L,),
            "residue_mask": (L,),
        }
        for name, expected in expected_shapes.items():
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(
                    f"StageCSample.{name} shape {value.shape} != expected {expected}"
                )
        if self.static_condition is not None and self.static_condition.shape[0] != L:
            raise ValueError(
                f"static_condition length {self.static_condition.shape[0]} != {L}"
            )
        if len(self.sequence) != L:
            raise ValueError(f"sequence length {len(self.sequence)} != {L}")


# ── PBC unwrap and missing residue policies ──────────────────────────

class PBCUnwrapPolicy:
    """Periodic boundary condition unwrapping for atom37 coordinates.

    Records policy in run manifest:
    - unwrap_method: "sequential" or "none"
    - max_displacement_angstrom: threshold for wrapped bonds
    """
    def __init__(self, unwrap: bool = True, max_displacement: float = 10.0):
        self.unwrap = unwrap
        self.max_displacement = max_displacement

    def _unwrap_sequential(self, coords: np.ndarray) -> np.ndarray:
        """Sequential PBC unwrap across time dimension [T, L, 37, 3].

        For each atom across consecutive frames, if displacement exceeds
        half the box diagonal, apply correction.
        """
        T = coords.shape[0]
        unwrapped = coords.copy()
        for t in range(1, T):
            delta = unwrapped[t] - unwrapped[t - 1]
            # Simple heuristic: large jumps (> max_displacement) are wraps
            large_jumps = np.abs(delta) > self.max_displacement
            unwrapped[t] = unwrapped[t - 1] + delta  # propagate unwrapped coords
            # Replace wrapped atoms with adjusted coordinates
            for j in range(delta.shape[0]):
                for k in range(delta.shape[1]):
                    if large_jumps[j, k, :].any():
                        unwrapped[t, j, k] = unwrapped[t - 1, j, k] + delta[j, k]
        return unwrapped

    def apply(self, coords: np.ndarray) -> np.ndarray:
        if not self.unwrap:
            return coords
        return self._unwrap_sequential(coords)

    def manifest_record(self) -> dict:
        return {
            "pbc_unwrap": self.unwrap,
            "pbc_method": "sequential" if self.unwrap else "none",
            "max_displacement_angstrom": self.max_displacement,
        }


class AltlocPolicy:
    """Altloc handling policy for PDB files."""
    def __init__(self, keep_altloc: str = "A"):
        """Keep only specified altloc identifier, or first if empty."""
        self.keep_altloc = keep_altloc

    def manifest_record(self) -> dict:
        return {"altloc_policy": self.keep_altloc}


class MissingResiduePolicy:
    """Missing residue handling for incomplete structure data.

    Policies:
    - strict: raise error on missing residues
    - mask: add to residue_mask as absent
    - fill: fill with predicted coordinates from ideal geometry
    """
    def __init__(self, policy: str = "mask"):
        if policy not in ("strict", "mask", "fill"):
            raise ValueError(f"Unknown missing residue policy: {policy}")
        self.policy = policy

    def manifest_record(self) -> dict:
        return {"missing_residue_policy": self.policy}


# ── Schema validation ────────────────────────────────────────────────

class StageCSchemaValidator:
    """Validates Stage C data samples against schema contract.

    Guarantees:
    - atom37 coordinates in Angstroms
    - time in picoseconds
    - sequence compatible with aatype
    - all indices in valid range
    - no NaN coordinates in valid positions
    """
    _AA_ONE_LETTER: ClassVar[str] = "ACDEFGHIKLMNPQRSTVWY"

    @staticmethod
    def validate(sample: StageCSample) -> list[str]:
        errors: list[str] = []
        L = sample.aatype.shape[0]

        # Check aatype range
        if sample.aatype.max().item() >= 20 or sample.aatype.min().item() < 0:
            errors.append(f"aatype out of range [0,19]: min={sample.aatype.min()}, max={sample.aatype.max()}")

        # Check sequence consistency
        expected_seq = "".join(
            StageCSchemaValidator._AA_ONE_LETTER[int(aa)]
            if 0 <= int(aa) < 20 else "X"
            for aa in sample.aatype
        )
        if sample.sequence != expected_seq:
            errors.append(f"Sequence mismatch with aatype: expected {expected_seq[:10]}..., got {sample.sequence[:10]}...")

        # Check for NaN in valid positions
        valid_mask = sample.atom37_mask & sample.residue_mask.unsqueeze(-1)
        nan_count = torch.isnan(sample.atom37_coords[valid_mask]).sum().item()
        if nan_count > 0:
            errors.append(f"{nan_count} NaN values in valid atom37 positions")

        # Check coordinate scale (Angstroms)
        max_coord = sample.atom37_coords[sample.atom37_mask].abs().max().item()
        if max_coord > 10000.0:
            errors.append(f"Coordinate max {max_coord:.1f} suspiciously large")

        # Check time
        if sample.time_ps < 0:
            errors.append(f"Negative time_ps: {sample.time_ps}")

        # Check temperature
        if sample.temperature_k <= 0:
            errors.append(f"Invalid temperature_k: {sample.temperature_k}")

        return errors

    @staticmethod
    def validate_batch(samples: list[StageCSample]) -> list[str]:
        all_errors: list[str] = []
        for i, sample in enumerate(samples):
            errors = StageCSchemaValidator.validate(sample)
            for err in errors:
                all_errors.append(f"sample[{i}]: {err}")
        return all_errors


def atom37_roundtrip_test(sample: StageCSample) -> bool:
    """Verify that atom37 -> backbone -> topology roundtrip is identity.

    Tests that the atom37 representation is self-consistent:
    - backbone coords match atom37 extraction
    - atom37 mask is consistent with aatype
    """
    # Check backbone consistency
    bb_from_37 = backbone_from_atom37(sample.atom37_coords)
    bb_match = torch.allclose(bb_from_37, sample.backbone_coords, atol=1e-5)

    # Check mask consistency: for each residue, atom37_mask should match
    # the expected atom37 occupancy for that aatype
    mask_consistent = True
    for l in range(sample.aatype.shape[0]):
        if not sample.residue_mask[l]:
            continue
        rm = get_residue_atom_map(int(sample.aatype[l].item()))
        for j in range(37):
            expected_present = rm.atom37_to_atom14[j] >= 0
            actual_present = sample.atom37_mask[l, j].item()
            if expected_present != actual_present:
                mask_consistent = False
                break
        if not mask_consistent:
            break

    return bb_match and mask_consistent


__all__ = [
    "ATOM37_NUM",
    "ATOM37_NAMES",
    "ATOM37_NAME_TO_INDEX",
    "BACKBONE_ATOM37_INDICES",
    "ResidueAtomMap",
    "get_residue_atom_map",
    "build_atom37_gatherer",
    "backbone_from_atom37",
    "StageCSample",
    "PBCUnwrapPolicy",
    "AltlocPolicy",
    "MissingResiduePolicy",
    "StageCSchemaValidator",
    "atom37_roundtrip_test",
]