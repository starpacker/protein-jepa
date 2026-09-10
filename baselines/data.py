"""Shared CA feature extraction from mdCATH HDF5 files.

All baselines use aligned, flattened CA coordinates as features.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from protein_jepa_frame.data.manifest import TrajectoryRecord, load_manifest

BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


def parse_pdb_ca_indices(pdb_text: str, expected_atom_count: int) -> np.ndarray:
    """Parse PDB text to find CA atom indices for all residues.

    Returns an array of shape (n_residues,) with CA atom indices (0-indexed).
    CA indices are in the order they appear in the PDB file.
    """
    ca_indices: list[int] = []
    atom_index = 0
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        altloc = line[16:17].strip()
        if atom_name == "CA" and altloc in {"", "A"}:
            ca_indices.append(atom_index)
        atom_index += 1
    if atom_index != expected_atom_count:
        raise ValueError(
            f"PDB atom count {atom_index} != expected {expected_atom_count}"
        )
    return np.array(ca_indices, dtype=np.int64)


def get_ca_topology(record: TrajectoryRecord) -> dict[str, Any]:
    """Read PDB topology from an mdCATH HDF5 file and return CA indices and metadata.

    Returns dict with:
      - ca_indices: (n_residues,) int64 array of CA atom indices
      - n_residues: int
      - n_atoms: int
    """
    with h5py.File(record.path, "r") as f:
        domain = record.domain_id or record.protein_id
        dataset_name = (record.metadata or {}).get(
            "protein_pdb_dataset", f"{domain}/pdbProteinAtoms"
        )
        pdb_dataset = f[str(dataset_name)]
        pdb_text = pdb_dataset[()]
        if isinstance(pdb_text, bytes):
            pdb_text = pdb_text.decode("utf-8", errors="replace")
        atom_count = int((record.metadata or {}).get("atom_count", 0))
        if atom_count == 0:
            # Fallback: count atoms from PDB
            atom_count = sum(
                1 for line in str(pdb_text).splitlines()
                if line.startswith(("ATOM  ", "HETATM"))
            )
    ca_indices = parse_pdb_ca_indices(str(pdb_text), atom_count)
    return {
        "ca_indices": ca_indices,
        "n_residues": len(ca_indices),
        "n_atoms": atom_count,
    }


def _read_ca_coords_from_record(
    record: TrajectoryRecord, ca_indices: np.ndarray
) -> np.ndarray:
    """Read CA coordinates for all frames from a single trajectory.

    Returns (F, n_residues, 3) float32 array.
    """
    with h5py.File(record.path, "r") as f:
        group_name = (record.metadata or {}).get("hdf5_group")
        if not group_name:
            raise ValueError(f"{record.trajectory_id} missing hdf5_group metadata")
        group = f[str(group_name)]
        all_coords = group["coords"][:]  # (F, n_atoms, 3)
        ca_coords = all_coords[:, ca_indices, :].astype(np.float64)
    return ca_coords


def kabsch_align(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    """Align mobile points to reference using Kabsch (SVD) algorithm.

    Args:
        reference: (N, 3) target coordinates
        mobile: (F, N, 3) coordinates to align

    Returns:
        (F, N, 3) aligned coordinates
    """
    ref_center = reference.mean(axis=0, keepdims=True)
    reference_centered = reference - ref_center
    mobile_centered = mobile - mobile.mean(axis=1, keepdims=True)

    aligned = np.empty_like(mobile_centered)
    for i in range(mobile.shape[0]):
        covariance = mobile_centered[i].T @ reference_centered
        u, _, vt = np.linalg.svd(covariance)
        # Ensure right-handed coordinate system
        det = np.linalg.det(u @ vt)
        correction = np.eye(3)
        # Use sign function with proper handling: if det = 0, don't flip
        if det < 0:
            correction[2, 2] = -1.0
        rotation = u @ correction @ vt
        aligned[i] = mobile_centered[i] @ rotation
    return aligned


def extract_features_for_protein(
    records: list[TrajectoryRecord],
    frame_stride: int = 1,
) -> dict[str, np.ndarray]:
    """Extract aligned, flattened CA features for all trajectories of a protein.

    Steps:
      1. Parse PDB topology from first record
      2. Read CA coords for each trajectory
      3. Find common valid CA residues across all trajectories
      4. Kabsch-align each frame to first frame of first trajectory
      5. Flatten to (F, n_common_residues * 3)

    Args:
        records: All trajectory records for one protein
        frame_stride: Subsampling stride (1 = use all frames)

    Returns:
        {trajectory_id: (F, R*3) float64 features}
    """
    if not records:
        return {}

    # Get topology from first record
    topo = get_ca_topology(records[0])
    ca_indices = topo["ca_indices"]
    n_residues = topo["n_residues"]

    # Read all CA coords
    all_ca: list[tuple[str, np.ndarray]] = []
    for r in records:
        ca = _read_ca_coords_from_record(r, ca_indices)
        # Subsample
        ca = ca[::frame_stride]
        all_ca.append((r.trajectory_id, ca))

    # Find common valid CA residues (all atoms must be finite across all trajs)
    valid_mask = np.ones(n_residues, dtype=bool)
    for _, ca in all_ca:
        # Check that all frames have finite coords for each residue
        frame_valid = np.all(np.isfinite(ca), axis=(0, 2))  # (n_residues,)
        valid_mask &= frame_valid

    if valid_mask.sum() < 3:
        raise ValueError(
            f"Protein {records[0].protein_id}: only {valid_mask.sum()} "
            "common valid CA atoms (< 3 required)"
        )

    # Reference: first frame of first trajectory, common valid CA
    reference = all_ca[0][1][0, valid_mask]  # (R, 3)

    # Align and flatten
    features: dict[str, np.ndarray] = {}
    for traj_id, ca in all_ca:
        ca_valid = ca[:, valid_mask, :]  # (F, R, 3)
        aligned = kabsch_align(reference, ca_valid)  # (F, R, 3)
        flattened = aligned.reshape(aligned.shape[0], -1)  # (F, R*3)
        features[traj_id] = flattened

    return features


def load_all_proteins(
    manifest_path: str | Path,
    frame_stride: int = 1,
    max_proteins: int | None = None,
) -> dict[str, list[TrajectoryRecord]]:
    """Load manifest and group records by protein_id.

    Returns {protein_id: [TrajectoryRecord, ...]}
    """
    records = load_manifest(str(manifest_path))
    grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for r in records:
        grouped[r.protein_id].append(r)

    if max_proteins is not None:
        # Take first N proteins
        protein_ids = sorted(grouped)[:max_proteins]
        grouped = {pid: grouped[pid] for pid in protein_ids}

    return dict(grouped)