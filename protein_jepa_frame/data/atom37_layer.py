"""Gate 3: Unified atom37 data layer across mdCATH / ATLAS / FastFolding.

A single, dataset-agnostic entry point that converts raw per-dataset structure
data (backbone-only or full all-atom) into the canonical atom37 representation
used by Stage C, with explicit and recorded policies for:

- PBC unwrap (periodic boundary conditions)
- altloc selection
- missing atom / missing residue handling
- units (Angstrom coordinates, picosecond time)
- atom naming (standard atom37 ordering)
- residue / chain identifiers

All conversions are validated by round-trip, unit, atom-name, and residue/chain
ID tests (tests/test_gate3_atom37_layer.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from protein_jepa_frame.data.schema import (
    ATOM37_NAMES,
    ATOM37_NAME_TO_INDEX,
    ATOM37_NUM,
    BACKBONE_ATOM37_INDICES,
    AltlocPolicy,
    MissingResiduePolicy,
    PBCUnwrapPolicy,
    _RESTYPES,
    backbone_from_atom37,
    build_atom37_gatherer,
    get_residue_atom_map,
)

# One-letter codes in the SAME order as schema._RESTYPES (3-letter alphabetical,
# the OpenFold/atom37 canonical ordering). aatype index i ↔ _RESTYPES[i].
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA_ONE_LETTER = "".join(_THREE_TO_ONE[r] for r in _RESTYPES)
AA_ONE_TO_INDEX = {aa: i for i, aa in enumerate(AA_ONE_LETTER)}

# Backbone ordering used by the frame-based loaders: (N, CA, C, O)
BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")


@dataclass
class Atom37Policies:
    """Bundle of conversion policies, recorded in the run manifest."""
    pbc: PBCUnwrapPolicy = field(default_factory=lambda: PBCUnwrapPolicy(unwrap=False))
    altloc: AltlocPolicy = field(default_factory=lambda: AltlocPolicy(keep_altloc="A"))
    missing_residue: MissingResiduePolicy = field(
        default_factory=lambda: MissingResiduePolicy(policy="mask")
    )
    coordinate_unit: str = "Angstrom"
    time_unit: str = "picosecond"

    def manifest_record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "coordinate_unit": self.coordinate_unit,
            "time_unit": self.time_unit,
        }
        rec.update(self.pbc.manifest_record())
        rec.update(self.altloc.manifest_record())
        rec.update(self.missing_residue.manifest_record())
        return rec


@dataclass
class Atom37Frames:
    """Canonical atom37 representation of a (possibly multi-frame) structure.

    coords:        [T, L, 37, 3] atom37 coordinates (Angstrom)
    atom37_mask:   [T, L, 37] present-atom mask (bool)
    residue_mask:  [L] structurally-present residues (bool)
    aatype:        [L] residue type indices (0-19, 20=UNK)
    residue_index: [L] sequential residue index (int64)
    chain_id:      [L] chain identifier (int64)
    time_ps:       [T] physical time in picoseconds
    """
    coords: torch.Tensor
    atom37_mask: torch.Tensor
    residue_mask: torch.Tensor
    aatype: torch.Tensor
    residue_index: torch.Tensor
    chain_id: torch.Tensor
    time_ps: torch.Tensor

    def __post_init__(self):
        T, L = self.coords.shape[:2]
        if self.coords.shape[2:] != (ATOM37_NUM, 3):
            raise ValueError(f"coords must be [T,L,37,3], got {tuple(self.coords.shape)}")
        if self.atom37_mask.shape != (T, L, ATOM37_NUM):
            raise ValueError(f"atom37_mask must be [T,L,37], got {tuple(self.atom37_mask.shape)}")
        for name, expected in {
            "residue_mask": (L,), "aatype": (L,),
            "residue_index": (L,), "chain_id": (L,),
        }.items():
            v = getattr(self, name)
            if v.shape != expected:
                raise ValueError(f"{name} must be {expected}, got {tuple(v.shape)}")
        if self.time_ps.shape != (T,):
            raise ValueError(f"time_ps must be [T], got {tuple(self.time_ps.shape)}")

    @property
    def num_frames(self) -> int:
        return self.coords.shape[0]

    @property
    def num_residues(self) -> int:
        return self.coords.shape[1]

    def backbone(self) -> torch.Tensor:
        """Extract backbone [T, L, 4, 3] (N, CA, C, O)."""
        return backbone_from_atom37(self.coords)

    def backbone_mask(self) -> torch.Tensor:
        """Backbone atom present-mask [T, L, 4]."""
        idx = torch.tensor(BACKBONE_ATOM37_INDICES, dtype=torch.long)
        return self.atom37_mask[..., idx]


def aatype_from_sequence(sequence: str) -> torch.Tensor:
    """Convert a 1-letter sequence to aatype indices (unknown → 20)."""
    return torch.tensor(
        [AA_ONE_TO_INDEX.get(c, 20) for c in sequence], dtype=torch.long
    )


def _expected_atom37_mask(aatype: torch.Tensor) -> torch.Tensor:
    """Per-residue expected atom37 occupancy from residue type. [L, 37] bool."""
    _, _, atom37_mask = build_atom37_gatherer(aatype)
    # build_atom37_gatherer returns [N, L, 37] for 1-D aatype (N=1); squeeze to [L, 37].
    if atom37_mask.dim() == 3 and atom37_mask.shape[0] == 1 and aatype.dim() == 1:
        atom37_mask = atom37_mask.squeeze(0)
    return atom37_mask


def backbone_to_atom37(
    backbone_coords: torch.Tensor,
    aatype: torch.Tensor,
    *,
    fill_missing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed backbone [T, L, 4, 3] (N, CA, C, O) into atom37 coordinates.

    Side-chain atoms are left as zeros with mask=False (backbone-only datasets).

    Args:
        backbone_coords: [T, L, 4, 3] in N, CA, C, O order
        aatype: [L] residue type indices
        fill_missing: if True, mark all expected atoms present (not recommended)

    Returns:
        atom37_coords: [T, L, 37, 3]
        atom37_mask: [T, L, 37] bool
    """
    if backbone_coords.dim() != 4 or backbone_coords.shape[2:] != (4, 3):
        raise ValueError("backbone_coords must be [T,L,4,3]")
    T, L = backbone_coords.shape[:2]

    atom37 = torch.zeros(T, L, ATOM37_NUM, 3, dtype=backbone_coords.dtype)
    mask = torch.zeros(T, L, ATOM37_NUM, dtype=torch.bool)

    # Place N, CA, C, O into their atom37 slots
    for bb_pos, a37_idx in enumerate(BACKBONE_ATOM37_INDICES):
        atom37[:, :, a37_idx, :] = backbone_coords[:, :, bb_pos, :]
        mask[:, :, a37_idx] = True

    if fill_missing:
        expected = _expected_atom37_mask(aatype)  # [L, 37]
        mask = expected.unsqueeze(0).expand(T, L, ATOM37_NUM).clone()

    return atom37, mask


def all_atom_to_atom37(
    coords: torch.Tensor,
    atom_names: list[str],
    aatype: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map an all-atom structure to atom37 using standard atom names.

    Args:
        coords: [T, L, A, 3] all-atom coordinates for A named atoms per residue
        atom_names: length-A list of atom names (standard PDB naming)
        aatype: [L] residue type indices

    Returns:
        atom37_coords: [T, L, 37, 3]
        atom37_mask: [T, L, 37] bool
    """
    if coords.dim() != 4:
        raise ValueError("coords must be [T,L,A,3]")
    T, L, A = coords.shape[:3]
    if len(atom_names) != A:
        raise ValueError(f"atom_names length {len(atom_names)} != A={A}")
    if aatype.shape[0] != L:
        raise ValueError(f"aatype length {aatype.shape[0]} != L={L}")

    atom37 = torch.zeros(T, L, ATOM37_NUM, 3, dtype=coords.dtype, device=coords.device)
    mask = torch.zeros(T, L, ATOM37_NUM, dtype=torch.bool, device=coords.device)

    for a, name in enumerate(atom_names):
        idx = ATOM37_NAME_TO_INDEX.get(name)
        if idx is None:
            continue  # non-atom37 atom (e.g. hydrogen) skipped
        atom37[:, :, idx, :] = coords[:, :, a, :]
        mask[:, :, idx] = True

    # Intersect with expected occupancy per residue type to drop stray atoms
    expected = _expected_atom37_mask(aatype)  # [L, 37]
    mask = mask & expected.unsqueeze(0)  # [T, L, 37]

    return atom37, mask


def build_atom37_frames(
    *,
    backbone_coords: torch.Tensor | None = None,
    all_atom_coords: torch.Tensor | None = None,
    atom_names: list[str] | None = None,
    aatype: torch.Tensor,
    residue_index: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    residue_mask: torch.Tensor | None = None,
    time_ps: torch.Tensor | None = None,
    frame_interval_ps: float = 1.0,
    first_frame_time_ps: float = 0.0,
    policies: Atom37Policies | None = None,
) -> Atom37Frames:
    """Unified atom37 builder shared by all datasets.

    Exactly one of `backbone_coords` or `all_atom_coords`+`atom_names` must be given.

    Policies (PBC unwrap etc.) are applied before packing into atom37.
    """
    policies = policies or Atom37Policies()

    if (backbone_coords is None) == (all_atom_coords is None):
        raise ValueError("provide exactly one of backbone_coords or all_atom_coords")

    aatype = aatype.long()
    L = aatype.shape[0]

    if backbone_coords is not None:
        T = backbone_coords.shape[0]
        # Apply PBC unwrap on backbone CA path if requested
        if policies.pbc.unwrap:
            bb_np = backbone_coords.detach().cpu().numpy()
            # treat (L, 4) atoms as the atom axis for unwrap
            bb_np = policies.pbc.apply(bb_np.reshape(T, L, 4, 3))
            backbone_coords = torch.from_numpy(bb_np).to(backbone_coords.dtype)
        coords, atom37_mask = backbone_to_atom37(backbone_coords, aatype)
    else:
        T = all_atom_coords.shape[0]
        if atom_names is None:
            raise ValueError("all_atom_coords requires atom_names")
        coords, atom37_mask = all_atom_to_atom37(all_atom_coords, atom_names, aatype)

    if residue_index is None:
        residue_index = torch.arange(L, dtype=torch.long)
    else:
        residue_index = residue_index.long()
    if chain_id is None:
        chain_id = torch.zeros(L, dtype=torch.long)
    else:
        chain_id = chain_id.long()
    if residue_mask is None:
        # A residue is present if its backbone CA is present in every frame
        ca_idx = BACKBONE_ATOM37_INDICES[1]
        residue_mask = atom37_mask[:, :, ca_idx].all(dim=0)
    else:
        residue_mask = residue_mask.bool()

    if time_ps is None:
        time_ps = first_frame_time_ps + frame_interval_ps * torch.arange(T, dtype=torch.float32)
    else:
        time_ps = time_ps.float()

    return Atom37Frames(
        coords=coords,
        atom37_mask=atom37_mask,
        residue_mask=residue_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_id=chain_id,
        time_ps=time_ps,
    )


def atom37_roundtrip_backbone(frames: Atom37Frames) -> bool:
    """Round-trip check: atom37 → backbone → atom37 preserves backbone atoms."""
    bb = frames.backbone()  # [T, L, 4, 3]
    coords2, mask2 = backbone_to_atom37(bb, frames.aatype)
    # Backbone slots must match exactly
    for a37_idx in BACKBONE_ATOM37_INDICES:
        orig = frames.coords[:, :, a37_idx, :]
        rt = coords2[:, :, a37_idx, :]
        # Only compare where original backbone atom is present
        present = frames.atom37_mask[:, :, a37_idx]
        if present.any():
            if not torch.allclose(orig[present], rt[present], atol=1e-5):
                return False
    return True


__all__ = [
    "Atom37Policies",
    "Atom37Frames",
    "aatype_from_sequence",
    "backbone_to_atom37",
    "all_atom_to_atom37",
    "build_atom37_frames",
    "atom37_roundtrip_backbone",
    "BACKBONE_ATOM_NAMES",
    "AA_ONE_LETTER",
]
