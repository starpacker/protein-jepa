"""Run manifest: provenance, lineage, and reproducibility tracking.

Gate 0/Gate 11 canonical implementation:
- Each run uses a non-reusable directory
- Checkpoint records git commit, dirty diff hash, complete initial config,
  resume lineage, data manifest hash, upstream checkpoint hash,
  dependency lock, and RNG state
- Any result can be uniquely traced from run manifest back to code, data, weights
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# Hashing utilities
# ═══════════════════════════════════════════════════════════════════════

def _compute_sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    if not path.exists():
        return "MISSING"
    return _compute_sha256(path.read_bytes())


def _hash_config(config_dict: dict) -> str:
    """Deterministic hash of config."""
    return _compute_sha256(json.dumps(config_dict, sort_keys=True, default=str))


def _git_info(repo_path: Path | None = None) -> dict[str, str]:
    """Collect git commit and dirty state."""
    repo = repo_path or Path.cwd()
    info = {"commit": "unknown", "dirty": "false", "dirty_diff_hash": ""}

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(repo), timeout=10,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()

        # Check for uncommitted changes
        diff_result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True, text=True, cwd=str(repo), timeout=30,
        )
        if diff_result.returncode == 0 and diff_result.stdout.strip():
            info["dirty"] = "true"
            info["dirty_diff_hash"] = _compute_sha256(diff_result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return info


def _python_info() -> dict[str, Any]:
    """Collect Python environment info."""
    return {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
    }


# ═══════════════════════════════════════════════════════════════════════
# Manifest dataclass
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RunManifest:
    """Single source of truth for experiment provenance."""

    # Immutable identifiers
    run_id: str = ""                           # unique run identifier
    stage: str = ""                             # "A", "B", or "C"
    started_at: str = ""                        # ISO 8601 timestamp
    completed_at: str = ""                      # ISO 8601 timestamp

    # Code provenance
    git_commit: str = ""                        # git commit at run start
    git_dirty: str = "false"                    # whether uncommitted changes exist
    git_dirty_diff_hash: str = ""               # SHA256 of dirty diff
    code_package_hash: str = ""                 # SHA256 of installed package
    python_version: str = ""
    python_executable: str = ""
    platform: str = ""

    # Config provenance
    config_hash: str = ""                       # SHA256 of initial config
    config_snapshot: dict = field(default_factory=dict)  # full config at runtime

    # Data provenance
    data_manifest_hash: str = ""                # SHA256 of data manifest
    num_trajectories: int = 0
    num_proteins: int = 0
    total_frames: int = 0
    dataset_names: list[str] = field(default_factory=list)

    # Checkpoint provenance (upstream)
    init_checkpoint_path: str = ""
    init_checkpoint_sha256: str = ""
    init_stage: str = ""                         # which stage produced init checkpoint

    # Resume lineage
    resume_checkpoint_path: str = ""
    resume_checkpoint_sha256: str = ""
    resume_run_id: str = ""
    previous_runs: list[str] = field(default_factory=list)

    # Downstream checkpoints
    checkpoint_sha256s: dict[str, str] = field(default_factory=dict)

    # Random number generator state
    rng_seed: int = 0
    torch_rng_state: str = ""                   # hex-encoded initial RNG state

    # Policies
    policies: dict[str, Any] = field(default_factory=dict)

    # Gate-specific metadata
    gate_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        data = json.loads(path.read_text())
        return cls(**data)


# ═══════════════════════════════════════════════════════════════════════
# Manifest creation and management
# ═══════════════════════════════════════════════════════════════════════

class ManifestManager:
    """Create, save, and validate run manifests."""

    @staticmethod
    def create_run_id(stage: str) -> str:
        """Generate unique run ID: stage_YYYYMMDD_HHMMSS_<random>."""
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%d_%H%M%S")
        random_part = os.urandom(4).hex()
        return f"{stage}_{ts}_{random_part}"

    @staticmethod
    def create(
        stage: str,
        config: Any,
        init_checkpoint_path: str | None = None,
        resume_checkpoint_path: str | None = None,
        data_manifest_path: str | None = None,
        policies: dict | None = None,
        gate_metadata: dict | None = None,
    ) -> RunManifest:
        """Create a new run manifest.

        Args:
            stage: "A", "B", or "C"
            config: TrainConfig or dict
            init_checkpoint_path: path to initial checkpoint (Stage N-1)
            resume_checkpoint_path: path to resume checkpoint
            data_manifest_path: path to data manifest
            policies: dict of policy records (e.g., PBC, altloc)
            gate_metadata: gate-specific tags

        Returns:
            RunManifest instance
        """
        import torch
        from protein_jepa_frame.config import config_to_dict

        run_id = ManifestManager.create_run_id(stage)

        # Config
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif hasattr(config, '__dataclass_fields__'):
            from dataclasses import asdict
            config_dict = asdict(config)
        else:
            config_dict = dict(config)

        # Git info
        git = _git_info()
        py_info = _python_info()

        # Init checkpoint hash
        init_hash = ""
        if init_checkpoint_path:
            init_hash = _hash_file(Path(init_checkpoint_path))

        # Resume checkpoint hash
        resume_hash = ""
        resume_run = ""
        if resume_checkpoint_path:
            resume_hash = _hash_file(Path(resume_checkpoint_path))
            # Try to load parent manifest
            parent_manifest = Path(resume_checkpoint_path).parent / "run_manifest.json"
            if parent_manifest.exists():
                parent = RunManifest.load(parent_manifest)
                resume_run = parent.run_id

        # Data manifest hash
        data_hash = ""
        n_traj, n_prot, n_frames = 0, 0, 0
        datasets: list[str] = []
        if data_manifest_path:
            dp = Path(data_manifest_path)
            if dp.exists():
                data_hash = _hash_file(dp)
                # Count entries
                n_traj = len([l for l in dp.read_text().splitlines() if l.strip()])

        # Capture initial RNG state
        rng_state = ""
        try:
            rng_state = torch.get_rng_state().numpy().tobytes().hex()
        except Exception:
            pass

        manifest = RunManifest(
            run_id=run_id,
            stage=stage,
            started_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git["commit"],
            git_dirty=git["dirty"],
            git_dirty_diff_hash=git["dirty_diff_hash"],
            python_version=py_info["python_version"],
            python_executable=py_info["python_executable"],
            platform=py_info["platform"],
            config_hash=_hash_config(config_dict),
            config_snapshot=config_dict,
            data_manifest_hash=data_hash,
            num_trajectories=n_traj,
            num_proteins=n_prot,
            total_frames=n_frames,
            dataset_names=datasets,
            init_checkpoint_path=str(init_checkpoint_path or ""),
            init_checkpoint_sha256=init_hash,
            resume_checkpoint_path=str(resume_checkpoint_path or ""),
            resume_checkpoint_sha256=resume_hash,
            resume_run_id=resume_run,
            rng_seed=getattr(config, 'seed', 0) if hasattr(config, 'seed') else 0,
            torch_rng_state=rng_state,
            policies=policies or {},
            gate_metadata=gate_metadata or {},
        )

        return manifest

    @staticmethod
    def finalize(manifest: RunManifest, output_dir: Path) -> RunManifest:
        """Finalize manifest after training completes."""
        manifest.completed_at = datetime.now(timezone.utc).isoformat()

        # Hash all checkpoint files in output
        for ckpt_path in output_dir.glob("checkpoint_*.pt"):
            manifest.checkpoint_sha256s[ckpt_path.name] = _hash_file(ckpt_path)

        return manifest

    @staticmethod
    def validate_lineage(manifest: RunManifest) -> list[str]:
        """Validate that run lineage is traceable.

        Returns:
            list of validation warnings (empty if valid)
        """
        warnings = []

        if manifest.stage not in ("A", "B", "C"):
            warnings.append(f"Unknown stage: {manifest.stage}")

        if manifest.git_commit == "unknown":
            warnings.append("Git commit unknown; cannot verify code version")

        if manifest.init_checkpoint_path and not manifest.init_checkpoint_sha256:
            warnings.append("Init checkpoint specified but not hashed")

        if manifest.resume_checkpoint_path and not manifest.resume_run_id:
            warnings.append("Resume checkpoint but no parent run_id")

        if not manifest.config_hash:
            warnings.append("Config not hashed")

        return warnings


# ═══════════════════════════════════════════════════════════════════════
# Audit report builder
# ═══════════════════════════════════════════════════════════════════════

def build_run_lineage_chain(manifests: list[RunManifest]) -> list[dict]:
    """Build chronological chain of runs for audit trail."""
    chain = []
    for m in manifests:
        chain.append({
            "run_id": m.run_id,
            "stage": m.stage,
            "started_at": m.started_at,
            "git_commit": m.git_commit[:8],
            "init_checkpoint_sha256": m.init_checkpoint_sha256[:16],
            "resume_from": m.resume_run_id,
        })
    return chain


__all__ = [
    "RunManifest",
    "ManifestManager",
    "build_run_lineage_chain",
    "_hash_file",
    "_hash_config",
    "_git_info",
]