from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.manifest import TrajectoryRecord, load_manifest

BACKBONE_ATOMS = ("N", "CA", "C", "O")
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}


@dataclass(frozen=True)
class BackboneTopology:
    atom_indices: np.ndarray
    atom_mask: np.ndarray
    aatype: np.ndarray
    residue_index: np.ndarray
    chain_id: np.ndarray


@dataclass(frozen=True)
class _ClipEntry:
    record_index: int
    start: int
    count: int


@dataclass(frozen=True)
class _PairedEntry:
    record_index: int
    horizon_frames: int
    count: int


class TrajectoryClipDataset(Dataset):
    """Lazy fixed-length clip dataset over npz-backed trajectories."""

    def __init__(self, records: list[TrajectoryRecord], config: DataConfig,
                 msm_labels_path: str | None = None):
        self.records = records
        self.config = config
        self.entries: list[_ClipEntry] = []
        self.cumulative: list[int] = []
        total = 0
        for i, record in enumerate(records):
            num_frames = record.num_frames or self._read_num_frames(record)
            span = (config.num_frames - 1) * config.frame_stride + 1
            if num_frames < span:
                continue
            count = (num_frames - span) // config.window_stride + 1
            self.entries.append(_ClipEntry(i, 0, count))
            total += count
            self.cumulative.append(total)
        if not self.entries:
            raise ValueError("no trajectory has enough frames for the requested clip length")
        self._init_auxiliary_state(msm_labels_path)

    def _init_auxiliary_state(self, msm_labels_path: str | None) -> None:
        self._topology_cache: dict[str, BackboneTopology] = {}
        self.msm_state_labels: dict[str, list[int]] | None = None
        if msm_labels_path:
            import json
            with open(msm_labels_path) as f:
                lbl_data = json.load(f)
            self.msm_state_labels = {
                tid: t["state_labels"] for tid, t in lbl_data.get("trajectories", {}).items()
            }

    @staticmethod
    def _read_num_frames(record: TrajectoryRecord) -> int:
        if record.format.lower() == "npz":
            with np.load(record.path, mmap_mode="r") as data:
                return int(data["coords"].shape[0])
        import h5py

        with h5py.File(record.path, "r") as handle:
            if record.format.lower() == "protein-jepa-hdf5":
                return int(handle["coords"].shape[0])
            if record.format.lower() == "mdcath-hdf5":
                group = (record.metadata or {}).get("hdf5_group")
                if not group:
                    raise ValueError(f"{record.trajectory_id} missing metadata.hdf5_group")
                return int(handle[str(group)]["coords"].shape[0])
        raise ValueError(f"unsupported trajectory format {record.format}")

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _locate(self, index: int) -> tuple[TrajectoryRecord, np.ndarray]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry_idx = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[entry_idx - 1] if entry_idx else 0
        local = index - previous
        entry = self.entries[entry_idx]
        start = entry.start + local * self.config.window_stride
        frames = start + np.arange(self.config.num_frames, dtype=np.int64) * self.config.frame_stride
        return self.records[entry.record_index], frames

    def __getitem__(self, index: int) -> dict:
        record, frames = self._locate(index)
        fmt = record.format.lower()
        if fmt == "npz":
            sample = self._read_npz(record, frames)
        elif fmt == "protein-jepa-hdf5":
            sample = self._read_normalized_h5(record, frames)
        elif fmt == "mdcath-hdf5":
            sample = self._read_mdcath_h5(record, frames)
        else:
            raise ValueError(f"unsupported trajectory format {record.format}")
        # Add MSM state labels if available
        if self.msm_state_labels is not None:
            labels = self.msm_state_labels.get(record.trajectory_id, [])
            state = torch.full((self.config.num_frames,), -1, dtype=torch.long)
            for t, f in enumerate(frames):
                if int(f) < len(labels):
                    state[t] = int(labels[int(f)])
            sample["msm_state_labels"] = state
        return sample

    def _read_clip(self, record: TrajectoryRecord, frames: np.ndarray) -> dict:
        fmt = record.format.lower()
        if fmt == "npz":
            return self._read_npz(record, frames)
        if fmt == "protein-jepa-hdf5":
            return self._read_normalized_h5(record, frames)
        if fmt == "mdcath-hdf5":
            return self._read_mdcath_h5(record, frames)
        raise ValueError(f"unsupported trajectory format {record.format}")

    def _finalize(
        self,
        record: TrajectoryRecord,
        frames: np.ndarray,
        coords: np.ndarray,
        atom_mask: np.ndarray,
        residue_mask: np.ndarray,
        aatype: np.ndarray,
        residue_index: np.ndarray,
        chain_id: np.ndarray,
        time_ps: np.ndarray,
        forces: np.ndarray | None = None,
    ) -> dict:
        sample = {
            "coords": torch.from_numpy(coords * self.config.coordinate_scale),
            "atom_mask": torch.from_numpy(atom_mask),
            "residue_mask": torch.from_numpy(residue_mask),
            "aatype": torch.from_numpy(aatype),
            "residue_index": torch.from_numpy(residue_index),
            "chain_id": torch.from_numpy(chain_id),
            "time_ps": torch.from_numpy(time_ps.astype(np.float32, copy=False)),
            "temperature_k": torch.tensor(
                float(record.temperature_k) if record.temperature_k is not None else float("nan"), dtype=torch.float32
            ),
            "dataset": record.dataset or "unknown",
            "protein_id": record.protein_id,
            "trajectory_id": record.trajectory_id,
        }
        if forces is not None:
            sample["forces"] = torch.from_numpy(forces)
        return sample

    def _read_npz(self, record: TrajectoryRecord, frames: np.ndarray) -> dict:
        with np.load(record.path, mmap_mode="r") as data:
            coords = np.asarray(data["coords"][frames], dtype=np.float32)
            if coords.ndim != 4 or coords.shape[2:] != (4, 3):
                raise ValueError(f"{record.path} coords must be [F,L,4,3]")
            if self.config.max_residues is not None:
                coords = coords[:, : self.config.max_residues]
            T, L = coords.shape[:2]
            atom_mask = np.asarray(
                data["atom_mask"][frames, :L] if "atom_mask" in data else np.ones((T, L, 4)),
                dtype=bool,
            )
            residue_mask = np.asarray(
                data["residue_mask"][frames, :L] if "residue_mask" in data else atom_mask.all(axis=-1),
                dtype=bool,
            )
            aatype = np.asarray(data["aatype"][:L] if "aatype" in data else np.full(L, 20), dtype=np.int64)
            residue_index = np.asarray(
                data["residue_index"][:L] if "residue_index" in data else np.arange(L),
                dtype=np.int64,
            )
            chain_id = np.asarray(data["chain_id"][:L] if "chain_id" in data else np.zeros(L), dtype=np.int64)
            if "time_ps" in data:
                time_ps = np.asarray(data["time_ps"][frames], dtype=np.float32)
            elif record.frame_interval_ps is not None:
                time_ps = frames.astype(np.float32) * float(record.frame_interval_ps)
            else:
                time_ps = frames.astype(np.float32)
        return self._finalize(record, frames, coords, atom_mask, residue_mask, aatype, residue_index, chain_id, time_ps)

    def _read_normalized_h5(self, record: TrajectoryRecord, frames: np.ndarray) -> dict:
        import h5py

        with h5py.File(record.path, "r") as handle:
            coords = np.asarray(handle["coords"][frames.tolist()], dtype=np.float32)
            atom_mask = np.asarray(handle["atom_mask"][frames.tolist()], dtype=bool)
            time_ps = np.asarray(handle["time_ps"][frames.tolist()], dtype=np.float32)
            topo = handle["topology"]
            aatype = np.asarray(topo["aatype"], dtype=np.int64)
            residue_index = np.asarray(topo["residue_index"], dtype=np.int64)
            chain_id = np.asarray(topo["chain_index"], dtype=np.int64)
        if self.config.max_residues is not None:
            L = self.config.max_residues
            coords = coords[:, :L]
            atom_mask = atom_mask[:, :L]
            aatype = aatype[:L]
            residue_index = residue_index[:L]
            chain_id = chain_id[:L]
        residue_mask = atom_mask.all(axis=-1)
        return self._finalize(record, frames, coords, atom_mask, residue_mask, aatype, residue_index, chain_id, time_ps)

    def _read_mdcath_h5(self, record: TrajectoryRecord, frames: np.ndarray) -> dict:
        import h5py

        topology = self._mdcath_topology(record)
        with h5py.File(record.path, "r") as handle:
            group_name = (record.metadata or {}).get("hdf5_group")
            if not group_name:
                raise ValueError(f"{record.trajectory_id} missing metadata.hdf5_group")
            group = handle[str(group_name)]
            all_atom = np.asarray(group["coords"][frames.tolist()], dtype=np.float32)
            all_atom_forces = None
            if self.config.load_forces:
                if "forces" not in group:
                    raise ValueError(f"{record.trajectory_id} has no force dataset")
                all_atom_forces = np.asarray(group["forces"][frames.tolist()], dtype=np.float32)
        coords, atom_mask = self._gather_backbone(all_atom, topology)
        forces = None
        if all_atom_forces is not None:
            forces, force_mask = self._gather_backbone(all_atom_forces, topology)
            forces[~force_mask] = 0.0
        aatype = topology.aatype
        residue_index = topology.residue_index
        chain_id = topology.chain_id
        if self.config.max_residues is not None:
            L = self.config.max_residues
            coords = coords[:, :L]
            if forces is not None:
                forces = forces[:, :L]
            atom_mask = atom_mask[:, :L]
            aatype = aatype[:L]
            residue_index = residue_index[:L]
            chain_id = chain_id[:L]
        residue_mask = atom_mask.all(axis=-1)
        if record.frame_interval_ps is not None:
            time_ps = record.first_frame_time_ps + frames.astype(np.float32) * float(record.frame_interval_ps)
        else:
            time_ps = frames.astype(np.float32)
        return self._finalize(
            record, frames, coords, atom_mask, residue_mask, aatype,
            residue_index, chain_id, time_ps, forces=forces,
        )

    def _mdcath_topology(self, record: TrajectoryRecord) -> BackboneTopology:
        stat = record.path.stat()
        key = f"{record.path}:{stat.st_size}:{stat.st_mtime_ns}:{record.domain_id or record.protein_id}"
        cached = self._topology_cache.get(key)
        if cached is not None:
            return cached
        import h5py

        with h5py.File(record.path, "r") as handle:
            domain = handle[record.domain_id or record.protein_id]
            dataset_name = (record.metadata or {}).get("protein_pdb_dataset")
            pdb_dataset = handle[str(dataset_name)] if dataset_name else domain["pdbProteinAtoms"]
            pdb_text = pdb_dataset[()]
            if isinstance(pdb_text, bytes):
                pdb_text = pdb_text.decode("utf-8", errors="replace")
            atom_count = int((record.metadata or {}).get("atom_count", len(domain["resid"])))
        topology = self._parse_pdb_backbone(str(pdb_text), atom_count)
        self._topology_cache[key] = topology
        return topology

    @staticmethod
    def _parse_pdb_backbone(pdb_text: str, expected_atom_count: int) -> BackboneTopology:
        residues: dict[tuple[str, int, str], dict] = {}
        atom_index = 0
        for line in pdb_text.splitlines():
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16:17].strip()
            residue_name = line[17:20].strip().upper()
            chain_name = line[21:22].strip() or "_"
            residue_number = int(line[22:26])
            insertion_code = line[26:27].strip()
            key = (chain_name, residue_number, insertion_code)
            residue = residues.setdefault(
                key,
                {"name": residue_name, "atoms": {}, "chain": chain_name, "number": residue_number},
            )
            if atom_name in BACKBONE_ATOMS and altloc in {"", "A"}:
                residue["atoms"].setdefault(atom_name, atom_index)
            atom_index += 1
        if atom_index != expected_atom_count:
            raise ValueError(f"PDB atom count {atom_index} != expected {expected_atom_count}")
        chain_names = tuple(dict.fromkeys(str(residue["chain"]) for residue in residues.values()))
        chain_to_index = {chain: i for i, chain in enumerate(chain_names)}
        atom_indices = np.full((len(residues), len(BACKBONE_ATOMS)), -1, dtype=np.int64)
        atom_mask = np.zeros_like(atom_indices, dtype=bool)
        aatype = np.zeros((len(residues),), dtype=np.int64)
        residue_index = np.zeros((len(residues),), dtype=np.int64)
        chain_id = np.zeros((len(residues),), dtype=np.int64)
        for row, residue in enumerate(residues.values()):
            one = THREE_TO_ONE.get(residue["name"], "X")
            aatype[row] = AA_TO_INDEX[one]
            residue_index[row] = int(residue["number"])
            chain_id[row] = chain_to_index[residue["chain"]]
            for col, atom_name in enumerate(BACKBONE_ATOMS):
                idx = residue["atoms"].get(atom_name, -1)
                atom_indices[row, col] = idx
                atom_mask[row, col] = idx >= 0
        return BackboneTopology(atom_indices, atom_mask, aatype, residue_index, chain_id)

    @staticmethod
    def _gather_backbone(all_atom_coords: np.ndarray, topology: BackboneTopology) -> tuple[np.ndarray, np.ndarray]:
        frames = all_atom_coords.shape[0]
        coords = np.zeros((frames, topology.atom_indices.shape[0], 4, 3), dtype=np.float32)
        valid = topology.atom_mask
        coords[:, valid, :] = all_atom_coords[:, topology.atom_indices[valid], :]
        finite = np.isfinite(coords).all(axis=-1)
        atom_mask = np.broadcast_to(valid, (frames,) + valid.shape).copy() & finite
        coords[~atom_mask] = 0.0
        return coords, atom_mask


def _paired_context_offsets(config: DataConfig) -> np.ndarray:
    """Build chronological offsets with recent and slow endpoint history."""
    if config.paired_history_mode == "uniform":
        return np.arange(config.num_frames, dtype=np.int64) * config.frame_stride
    if config.paired_history_mode != "multiscale":
        raise ValueError(f"unknown paired_history_mode {config.paired_history_mode}")
    recent = int(config.paired_recent_frames)
    slow_stride = int(config.paired_slow_stride)
    if not 1 <= recent < config.num_frames:
        raise ValueError("paired_recent_frames must be between 1 and num_frames - 1")
    if slow_stride <= config.frame_stride:
        raise ValueError("paired_slow_stride must exceed frame_stride")
    recent_back = np.arange(recent, dtype=np.int64) * config.frame_stride
    slow_back = recent * config.frame_stride + np.arange(
        config.num_frames - recent, dtype=np.int64
    ) * slow_stride
    span = int(max(recent_back.max(initial=0), slow_back.max(initial=0)) + 1)
    return np.sort(span - 1 - np.concatenate([recent_back, slow_back]))


class PairedTrajectoryDataset(TrajectoryClipDataset):
    """Two same-trajectory windows separated by a raw-frame horizon."""

    def __init__(self, records: list[TrajectoryRecord], config: DataConfig, msm_labels_path: str | None = None):
        self.records = records
        self.config = config
        self.context_offsets = _paired_context_offsets(config)
        self.horizons_frames = tuple(sorted({int(h) for h in config.paired_horizons_frames}))
        if not self.horizons_frames or any(h <= 0 for h in self.horizons_frames):
            raise ValueError("paired horizons must be positive")
        if any(h % config.frame_stride != 0 for h in self.horizons_frames):
            raise ValueError("paired horizons must be divisible by frame_stride")
        self.entries: list[_PairedEntry] = []
        self.cumulative: list[int] = []
        total = 0
        span = max(self.context_offsets) + 1
        if config.paired_nonoverlap_only and any(horizon < span for horizon in self.horizons_frames):
            raise ValueError(
                "paired_nonoverlap_only requires every horizon to be at least "
                f"the raw clip span ({span})"
            )
        for record_index, record in enumerate(records):
            num_frames = record.num_frames or self._read_num_frames(record)
            entry_horizons = (max(self.horizons_frames),) if config.paired_all_horizons else self.horizons_frames
            for horizon in entry_horizons:
                count = (num_frames - horizon - span) // config.window_stride + 1
                if count <= 0:
                    continue
                self.entries.append(_PairedEntry(record_index, horizon, count))
                total += count
                self.cumulative.append(total)
        if not self.entries:
            raise ValueError("no trajectory has enough frames for paired windows")
        self._init_auxiliary_state(msm_labels_path)

    def _locate(self, index: int) -> tuple[TrajectoryRecord, np.ndarray, np.ndarray, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        entry_idx = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[entry_idx - 1] if entry_idx else 0
        local = index - previous
        entry = self.entries[entry_idx]
        start = local * self.config.window_stride
        offsets = self.context_offsets
        return self.records[entry.record_index], start + offsets, start + entry.horizon_frames + offsets, entry.horizon_frames

    def __getitem__(self, index: int) -> dict:
        record, context_frames, future_frames, horizon = self._locate(index)
        context = self._read_clip(record, context_frames)
        if self.config.paired_all_horizons:
            start = int(context_frames[0])
            offsets = self.context_offsets
            pairs = []
            for grouped_horizon in self.horizons_frames:
                grouped_future = self._read_clip(record, start + grouped_horizon + offsets)
                pairs.append(
                    {
                        "context": context,
                        "future": grouped_future,
                        "horizon_frames": torch.tensor(grouped_horizon, dtype=torch.long),
                        "horizon_time_ps": grouped_future["time_ps"][0] - context["time_ps"][0],
                    }
                )
            return {"pairs": pairs}
        future = self._read_clip(record, future_frames)
        return {
            "context": context,
            "future": future,
            "horizon_frames": torch.tensor(horizon, dtype=torch.long),
            "horizon_time_ps": future["time_ps"][0] - context["time_ps"][0],
        }


class SyntheticProteinDynamicsDataset(Dataset):
    """Smooth toy trajectories for smoke tests and CI."""

    def __init__(self, config: DataConfig):
        self.config = config

    def __len__(self) -> int:
        return self.config.synthetic_size

    def __getitem__(self, index: int) -> dict:
        T = self.config.num_frames
        L = self.config.synthetic_residues
        residue = torch.arange(L, dtype=torch.float32)
        base = torch.stack(
            [
                1.5 * residue,
                4.0 * torch.sin(residue / 5.0),
                4.0 * torch.cos(residue / 5.0),
            ],
            dim=-1,
        )
        offsets = torch.tensor(
            [[-1.2, 0.1, 0.0], [0.0, 0.0, 0.0], [1.3, 0.1, 0.0], [1.8, 1.0, 0.0]],
            dtype=torch.float32,
        )
        coords = []
        phase = float(index % 17) / 17.0 * 2.0 * math.pi
        for t in range(T):
            time = phase + 0.4 * t
            bend = torch.zeros_like(base)
            bend[:, 1] = torch.sin(residue / 7.0 + time) * 0.8
            bend[:, 2] = torch.cos(residue / 11.0 - time) * 0.6
            coords.append(base[:, None, :] + offsets[None, :, :] + bend[:, None, :])
        coords_t = torch.stack(coords, dim=0)
        sample = {
            "coords": coords_t,
            "atom_mask": torch.ones(T, L, 4, dtype=torch.bool),
            "residue_mask": torch.ones(T, L, dtype=torch.bool),
            "aatype": torch.full((L,), 20, dtype=torch.long),
            "residue_index": torch.arange(L, dtype=torch.long),
            "chain_id": torch.zeros(L, dtype=torch.long),
            "time_ps": torch.arange(T, dtype=torch.float32),
            "temperature_k": torch.tensor(300.0, dtype=torch.float32),
            "dataset": "synthetic",
            "protein_id": f"synthetic_{index % 8}",
            "trajectory_id": f"synthetic_traj_{index}",
        }
        if self.config.load_forces:
            center = coords_t.mean(dim=(1, 2), keepdim=True)
            sample["forces"] = -(coords_t - center)
        return sample


class SyntheticPairedProteinDynamicsDataset(Dataset):
    """Synthetic paired windows used by V2.1 smoke tests."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.context_offsets = _paired_context_offsets(config)
        self.horizons_frames = tuple(sorted({int(h) for h in config.paired_horizons_frames}))
        if not self.horizons_frames or any(h <= 0 for h in self.horizons_frames):
            raise ValueError("paired horizons must be positive")
        if any(h % config.frame_stride != 0 for h in self.horizons_frames):
            raise ValueError("paired horizons must be divisible by frame_stride")
        span = max(self.context_offsets) + 1
        if config.paired_nonoverlap_only and any(horizon < span for horizon in self.horizons_frames):
            raise ValueError(
                "paired_nonoverlap_only requires every horizon to be at least "
                f"the raw clip span ({span})"
            )
        total_frames = span + max(self.horizons_frames) + span - 1
        full_config = replace(config, num_frames=total_frames, training_mode="clip")
        self.base = SyntheticProteinDynamicsDataset(full_config)

    def __len__(self) -> int:
        multiplier = 1 if self.config.paired_all_horizons else len(self.horizons_frames)
        return self.config.synthetic_size * multiplier

    @staticmethod
    def _slice(sample: dict, indices: torch.Tensor) -> dict:
        frame_keys = {"coords", "forces", "atom_mask", "residue_mask", "time_ps", "msm_state_labels"}
        return {
            key: value[indices] if key in frame_keys and torch.is_tensor(value) else value
            for key, value in sample.items()
        }

    def __getitem__(self, index: int) -> dict:
        if self.config.paired_all_horizons:
            sample = self.base[index]
            offsets = torch.as_tensor(self.context_offsets, dtype=torch.long)
            context = self._slice(sample, offsets)
            return {
                "pairs": [
                    {
                        "context": context,
                        "future": self._slice(sample, offsets + horizon),
                        "horizon_frames": torch.tensor(horizon, dtype=torch.long),
                        "horizon_time_ps": sample["time_ps"][horizon] - sample["time_ps"][0],
                    }
                    for horizon in self.horizons_frames
                ]
            }
        horizon_index = index % len(self.horizons_frames)
        base_index = index // len(self.horizons_frames)
        horizon = self.horizons_frames[horizon_index]
        sample = self.base[base_index]
        offsets = torch.as_tensor(self.context_offsets, dtype=torch.long)
        context = self._slice(sample, offsets)
        future = self._slice(sample, offsets + horizon)
        return {
            "context": context,
            "future": future,
            "horizon_frames": torch.tensor(horizon, dtype=torch.long),
            "horizon_time_ps": future["time_ps"][0] - context["time_ps"][0],
        }


def build_dataset(config: DataConfig, msm_labels_path: str | None = None) -> Dataset:
    if config.training_mode == "paired_window":
        if config.manifest_path:
            return PairedTrajectoryDataset(load_manifest(config.manifest_path), config, msm_labels_path=msm_labels_path)
        return SyntheticPairedProteinDynamicsDataset(config)
    if config.manifest_path:
        return TrajectoryClipDataset(load_manifest(config.manifest_path), config, msm_labels_path=msm_labels_path)
    return SyntheticProteinDynamicsDataset(config)
