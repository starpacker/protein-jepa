"""Bridge mdCATH evaluation windows into bioemu2 all-atom inputs.

mdCATH HDF5 trajectories store a backbone subset of atom37: each residue only
has ``[N, CA, C, O]`` coordinates.  bioemu2, by contrast, featurizes a *full*
all-atom structure and requires every modelled atom to be present.

This module therefore:

1. crops the window to the evaluated residues (``rmask``),
2. decodes the one-letter sequence from the window's ``aatype`` indices,
3. builds a bioemu2 ``System`` (single protein chain),
4. maps the true mdCATH N/CA/C/O coordinates onto the system's atom order and
   places deterministic, non-degenerate placeholders for side-chain heavy
   atoms (only CA is scored by the ensemble metrics, so side-chain accuracy is
   irrelevant here).

The anchor is returned as an ``mdtraj.Trajectory`` on the bioemu2 topology.
"""
from __future__ import annotations

import numpy as np

# one-letter alphabet used by protein_jepa_frame.data.dataset
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
# water masks mdCATH does not need: 0 for the 20 standard residues + X


def decode_sequence(window, rmask) -> str:
    """One-letter sequence of the evaluated residues, in residue order."""
    aatype = window.context["aatype"]
    if hasattr(aatype, "cpu"):
        aatype = aatype.cpu().numpy()
    aatype = np.asarray(aatype)
    idx = np.where(np.asarray(rmask.cpu().numpy() if hasattr(rmask, "cpu") else rmask))[0]
    if idx.size == 0:
        raise ValueError("empty residue mask")
    # `aatype` may be [L] (single frame) or [K, L] (per history frame)
    if aatype.ndim == 2:
        aatype = aatype[-1]
    letters = "".join(AA_ALPHABET[int(a)] if 0 <= int(a) < len(AA_ALPHABET) else "X"
                      for a in aatype[idx])
    # non-standard / unknown -> alanine placeholder (matches _clean_seq below)
    return "".join(c if c in _STANDARD else "A" for c in letters)


def make_system(window, rmask):
    """Build a single-chain bioemu2 System for the window's evaluated residues."""
    from bioemu2 import System

    seq = decode_sequence(window, rmask)
    return System.from_description(proteins=[seq])


def window_anchor_x0(system, window, rmask):
    """Anchor frame as an mdtraj Trajectory matching ``system.topology`` (in nm)."""
    import mdtraj

    idx = np.where(np.asarray(rmask.cpu().numpy() if hasattr(rmask, "cpu") else rmask))[0]
    coords = window.context["coords"][-1]
    coords = coords.cpu().numpy() if hasattr(coords, "cpu") else np.asarray(coords)
    bb = coords[idx]                                          # [Lval, 4, 3] Angstrom (N,CA,C,O)
    xyz_nm = _bb_to_allatom(system, bb) / 10.0                # mdtraj uses nanometers
    return mdtraj.Trajectory(xyz=xyz_nm[None].astype(np.float32),
                             topology=system.topology)


def ca_indices(system):
    """Atom indices of CA atoms, in residue order."""
    return system.topology.select("name CA")


def _bb_to_allatom(system, bb: np.ndarray) -> np.ndarray:
    """Map mdCATH [Lval, 4, 3] (N, CA, C, O) onto the system's atom order (Angstrom)."""
    top = system.topology
    out = np.zeros((top.n_atoms, 3), dtype=np.float32)
    res_names = [r.name for r in top.residues]
    if len(res_names) != bb.shape[0]:
        raise ValueError(f"residue count mismatch: topology {len(res_names)} "
                         f"vs anchor {bb.shape[0]}")
    for ai, atom in enumerate(top.atoms):
        r = atom.residue.index
        name = atom.name
        N, CA, C, O = bb[r, 0], bb[r, 1], bb[r, 2], bb[r, 3]
        if name == "N":
            out[ai] = N
        elif name == "CA":
            out[ai] = CA
        elif name == "C":
            out[ai] = C
        elif name == "O":
            out[ai] = O
        elif name.startswith("H"):
            # hydrogens: bioemu2 does not model them; place on CA (never read)
            out[ai] = CA
        else:
            # Side-chain heavy atom: deterministic placeholder placed off CA.
            # Metrics only read CA; the placeholder just needs to be finite and
            # non-coincident so bioemu2's featurization stays well-conditioned.
            h = (abs(hash((int(r), name))) % 997)
            ang = 2.0 * np.pi * h / 997.0
            rad = 1.0 + 0.5 * ((h % 7) / 7.0)
            out[ai] = CA + np.array(
                [rad * np.cos(ang), rad * np.sin(ang), 0.6 * ((h % 5) - 2)],
                dtype=np.float32)
    return out
