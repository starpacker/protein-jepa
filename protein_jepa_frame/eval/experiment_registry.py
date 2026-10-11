"""E00: Experiment registry and data manifest infrastructure.

Following Section 5.1 of the CVPR plan:
  - experiment_registry.csv: data version, code version, checkpoint, seed, hyperparams
  - metrics.jsonl: per-protein, per-seed, per-horizon raw metrics
  - generate_tables.py: from unique result source → tables and diffs
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ── Experiment registry ──────────────────────────────────────────

@dataclass
class ExperimentRecord:
    """Single experiment run record."""
    experiment_id: str
    task: str                        # T1a, T1b, E01, etc.
    model: str                       # LaSP, B0, B1, B2, B4, B5
    dataset: str                     # mdCATH, ATLAS, FastFolding
    representation: str              # Stage A checkpoint hash
    seed: int
    horizon_ps: float | None = None  # None = multi-horizon
    n_params: int = 0
    n_steps_trained: int = 0
    train_loss_final: float = 0.0
    git_sha: str = ""
    timestamp: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class ExperimentRegistry:
    """Append-only registry of all experiments."""

    FIELD_NAMES = [
        "experiment_id", "task", "model", "dataset", "representation",
        "seed", "horizon_ps", "n_params", "n_steps_trained",
        "train_loss_final", "git_sha", "timestamp",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._ensure_header()

    def _ensure_header(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.FIELD_NAMES)
                w.writeheader()

    def register(self, record: ExperimentRecord):
        record.timestamp = record.timestamp or datetime.datetime.now().isoformat()
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            record.git_sha = record.git_sha or sha
        except Exception:
            pass
        row = {k: record.__dict__.get(k, "") for k in self.FIELD_NAMES}
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELD_NAMES)
            w.writerow(row)

    def query(self, **filters) -> list[dict]:
        rows = []
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                if all(str(row.get(k, "")) == str(v) for k, v in filters.items()):
                    rows.append(row)
        return rows


# ── Metrics logger ──────────────────────────────────────────────

class MetricsLogger:
    """Append-only JSONL logger for per-sample metrics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: dict):
        entry["timestamp"] = datetime.datetime.now().isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries


# ── Data manifest ───────────────────────────────────────────────

@dataclass
class DataManifestEntry:
    """One row in data_manifest.csv."""
    protein_id: str
    dataset: str                   # mdCATH, ATLAS, FastFolding
    n_residues: int
    n_frames: int
    time_interval_ps: float        # real time between frames
    total_time_ps: float           # total trajectory time
    temperature_k: float | None = None
    chain: str = "A"
    sequence: str = ""


class DataManifest:
    """Build and audit data_manifest.csv."""

    FIELD_NAMES = [
        "protein_id", "dataset", "n_residues", "n_frames",
        "time_interval_ps", "total_time_ps", "temperature_k",
        "chain", "sequence",
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def scan_dataset(self, dataset_name: str, data_dir: Path) -> list[DataManifestEntry]:
        """Scan a dataset directory and build entries."""
        entries = []
        if not data_dir.exists():
            return entries

        for npz_path in sorted(data_dir.glob("*.npz")):
            try:
                data = np.load(npz_path, allow_pickle=True)
                protein_id = npz_path.stem
                n_residues = int(data.get("n_residues", 0))
                n_frames = int(data.get("n_frames", 0))
                time_interval = float(data.get("time_interval_ps", 100.0))
                total_time = n_frames * time_interval
                temperature = float(data.get("temperature_k", 348.0)) if "temperature_k" in data else None
                entries.append(DataManifestEntry(
                    protein_id=protein_id, dataset=dataset_name,
                    n_residues=n_residues, n_frames=n_frames,
                    time_interval_ps=time_interval, total_time_ps=total_time,
                    temperature_k=temperature,
                ))
            except Exception as e:
                print(f"Warning: could not scan {npz_path}: {e}")
        return entries

    def write(self, entries: list[DataManifestEntry]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELD_NAMES)
            w.writeheader()
            for e in entries:
                w.writerow({k: e.__dict__.get(k, "") for k in self.FIELD_NAMES})
        return len(entries)


# ── Split manifest ──────────────────────────────────────────────

@dataclass
class SplitEntry:
    protein_id: str
    stage: str          # "A" or "B"
    split: str           # "train", "val", "test"
    dataset: str
    fold: int = 0
    notes: str = ""


class SplitManifest:
    FIELD_NAMES = ["protein_id", "stage", "split", "dataset", "fold", "notes"]

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def write_random_split(
        self, protein_ids: list[str], dataset: str,
        stage: str, val_frac: float = 0.1, test_frac: float = 0.2,
        seed: int = 42,
    ) -> list[SplitEntry]:
        """Create random stratified split."""
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(protein_ids))
        n_test = max(1, int(len(protein_ids) * test_frac))
        n_val = max(1, int(len(protein_ids) * val_frac))
        n_train = len(protein_ids) - n_test - n_val

        entries = []
        for i, pos in enumerate(idx):
            pid = protein_ids[pos]
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            entries.append(SplitEntry(pid, stage, split, dataset))

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELD_NAMES)
            w.writeheader()
            for e in entries:
                w.writerow({k: e.__dict__.get(k, "") for k in self.FIELD_NAMES})

        return entries


__all__ = [
    "ExperimentRecord", "ExperimentRegistry",
    "MetricsLogger",
    "DataManifestEntry", "DataManifest",
    "SplitEntry", "SplitManifest",
]
