from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CAFrame:
    coords: np.ndarray
    residue_index: np.ndarray
    chain_id: np.ndarray


def read_ca_frame(path: str | Path) -> CAFrame:
    """Read CA coordinates from a PDB file without modifying it."""
    path = Path(path)
    coords: list[list[float]] = []
    residue_index: list[int] = []
    chain_names: dict[str, int] = {}
    chain_id: list[int] = []
    seen: set[tuple[str, int, str]] = set()
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16:17].strip()
            if atom_name != "CA" or altloc not in {"", "A"}:
                continue
            chain = line[21:22].strip() or "_"
            resseq = int(line[22:26])
            icode = line[26:27].strip()
            key = (chain, resseq, icode)
            if key in seen:
                continue
            seen.add(key)
            chain_names.setdefault(chain, len(chain_names))
            residue_index.append(resseq)
            chain_id.append(chain_names[chain])
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"{path} contains no CA atoms")
    return CAFrame(
        coords=np.asarray(coords, dtype=np.float64),
        residue_index=np.asarray(residue_index, dtype=np.int64),
        chain_id=np.asarray(chain_id, dtype=np.int64),
    )


def kabsch_rmsd(coords: np.ndarray, reference: np.ndarray) -> float:
    n = min(coords.shape[0], reference.shape[0])
    if n == 0:
        return float("nan")
    x = coords[:n] - coords[:n].mean(axis=0, keepdims=True)
    y = reference[:n] - reference[:n].mean(axis=0, keepdims=True)
    cov = x.T @ y
    u, _, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = u @ vt
    aligned = x @ rot
    return float(np.sqrt(np.mean(np.sum((aligned - y) ** 2, axis=1))))


def radius_of_gyration(coords: np.ndarray) -> float:
    centered = coords - coords.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def native_contact_pairs(reference: np.ndarray, cutoff: float = 8.0, min_separation: int = 3) -> np.ndarray:
    n = reference.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + min_separation, n):
            if np.linalg.norm(reference[i] - reference[j]) <= cutoff:
                pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64)


def native_contact_fraction(coords: np.ndarray, pairs: np.ndarray, cutoff: float = 9.6) -> float:
    if pairs.size == 0:
        return float("nan")
    n = coords.shape[0]
    valid = pairs[(pairs[:, 0] < n) & (pairs[:, 1] < n)]
    if valid.size == 0:
        return float("nan")
    distances = np.linalg.norm(coords[valid[:, 0]] - coords[valid[:, 1]], axis=1)
    return float(np.mean(distances <= cutoff))

