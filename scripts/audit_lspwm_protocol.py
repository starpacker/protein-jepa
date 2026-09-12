#!/usr/bin/env python3
"""Audit the frozen LSP-WM v4.4 evaluation protocol.

This command performs metadata-only checks. It does not load checkpoints,
modify model files, or start training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = {
    "train": ROOT / "manifests/tmjepa_v3.train.jsonl",
    "validation": ROOT / "manifests/tmjepa_v3.validation.jsonl",
    "test": ROOT / "manifests/tmjepa_v3.test.jsonl",
    "temperature_test": ROOT / "manifests/tmjepa_v3.temperature_test.jsonl",
}
DEFAULT_CONFIGS = {
    "stage_b_v3": ROOT / "configs/lspwm_v44_stage_b_v3_improved_thermodynamics.yaml",
    "detskill_v1": ROOT / "configs/lspwm_v44_stage_b_detskill_fix.yaml",
    "detskill_v2": ROOT / "artifacts/lspwm_v44_stage_b_detskill_fix_v2/config.json",
    "detskill_v3": ROOT / "artifacts/lspwm_v44_stage_b_detskill_fix_v3/config.json",
}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            records.append(record)
    return records


def _manifest_summary(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    records = _load_jsonl(path)
    proteins = {str(record.get("protein_id", "")) for record in records}
    trajectories = {str(record.get("trajectory_id", "")) for record in records}
    temperatures = sorted(
        {
            float(record["temperature_k"])
            for record in records
            if record.get("temperature_k") is not None
        }
    )
    intervals = sorted(
        {
            float(record["frame_interval_ps"])
            for record in records
            if record.get("frame_interval_ps") is not None
        }
    )
    return {
        "path": str(path),
        "exists": True,
        "records": len(records),
        "proteins": len(proteins - {""}),
        "trajectories": len(trajectories - {""}),
        "temperatures_k": temperatures,
        "frame_intervals_ps": intervals,
        "protein_ids": sorted(proteins - {""}),
    }


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle) if path.suffix == ".json" else yaml.safe_load(handle)
    model = payload.get("model", {})
    optimization = payload.get("optimization", {})
    data = payload.get("data", {})
    return {
        "path": str(path),
        "exists": True,
        "seed": payload.get("seed"),
        "manifest_path": data.get("manifest_path"),
        "batch_size": data.get("batch_size"),
        "paired_horizons_frames": data.get("paired_horizons_frames"),
        "mean_force_bandwidth": model.get("lspwm_mean_force_bandwidth"),
        "rollout_enabled": model.get("lspwm_rollout_enabled"),
        "rollout_trajectory_nll_weight": optimization.get(
            "lspwm_rollout_trajectory_nll_weight"
        ),
        "paired_delta_weight": optimization.get("paired_delta_weight"),
        "paired_state_weight": optimization.get("paired_state_weight"),
        "freeze_context_encoder": model.get("freeze_context_encoder"),
        "freeze_temporal_encoder": model.get("freeze_temporal_encoder"),
        "freeze_target_encoders": model.get("freeze_target_encoders"),
        "encoder_lr_scale": optimization.get("encoder_lr_scale"),
    }


def _render_markdown(payload: dict) -> str:
    lines = [
        "# LSP-WM Phase 0 Protocol Audit",
        "",
        f"Generated: `{payload['generated_at']}`",
        f"Git commit: `{payload.get('git_commit') or 'unknown'}`",
        "",
        "## Split Summary",
        "",
        "| Split | Exists | Records | Proteins | Trajectories | Temperatures (K) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, summary in payload["manifests"].items():
        lines.append(
            f"| `{name}` | {summary.get('exists', False)} | "
            f"{summary.get('records', 0)} | {summary.get('proteins', 0)} | "
            f"{summary.get('trajectories', 0)} | "
            f"`{summary.get('temperatures_k', [])}` |"
        )
    lines.extend(["", "## Protein Overlap Checks", ""])
    lines.append("| Pair | Shared proteins | Status |")
    lines.append("|---|---:|---|")
    for item in payload["overlaps"]:
        expected = {item["left"], item["right"]} == {"test", "temperature_test"}
        status = "EXPECTED" if expected else ("PASS" if item["count"] == 0 else "FAIL")
        lines.append(
            f"| `{item['left']} vs {item['right']}` | {item['count']} | "
            f"{status} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Configs",
            "",
            "| Candidate | Paired delta weight | KDE bandwidth | Rollout | Encoder frozen |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, config in payload["configs"].items():
        frozen = all(
            config.get(key) is True
            for key in (
                "freeze_context_encoder",
                "freeze_temporal_encoder",
                "freeze_target_encoders",
            )
        )
        lines.append(
            f"| `{name}` | `{config.get('paired_delta_weight')}` | "
            f"`{config.get('mean_force_bandwidth')}` | "
            f"`{config.get('rollout_enabled')}` | `{frozen}` |"
        )
    lines.extend(["", "## Warnings", ""])
    if payload["warnings"]:
        lines.extend(f"- {warning}" for warning in payload["warnings"])
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifests = {
        name: _manifest_summary(path) for name, path in DEFAULT_SPLITS.items()
    }
    warnings: list[str] = []
    for name, summary in manifests.items():
        if not summary.get("exists"):
            warnings.append(f"missing manifest: {summary['path']}")

    overlaps = []
    names = list(manifests)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_ids = set(manifests[left].get("protein_ids", []))
            right_ids = set(manifests[right].get("protein_ids", []))
            shared = sorted(left_ids & right_ids)
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "count": len(shared),
                    "protein_ids": shared,
                }
            )
            if left in {"train", "validation", "test"} and right in {
                "train",
                "validation",
                "test",
            } and shared:
                warnings.append(
                    f"protein overlap between {left} and {right}: {shared[:5]}"
                )

    configs = {name: _load_config(path) for name, path in DEFAULT_CONFIGS.items()}
    for name, config in configs.items():
        if not config.get("exists"):
            warnings.append(f"missing candidate config: {config['path']}")
        if config.get("manifest_path") == str(DEFAULT_SPLITS["train"]):
            warnings.append(
                f"{name} points at the training manifest by default; every held-out "
                "evaluation must pass --manifest explicitly"
            )
        if config.get("paired_delta_weight", 0.0) == 0.0:
            warnings.append(f"{name} has paired_delta_weight=0")
        if config.get("mean_force_bandwidth") in {0.05, 0.1}:
            pass

    payload = {
        "protocol_version": "lspwm-v44-phase0-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "repository": str(ROOT),
        "manifests": manifests,
        "overlaps": overlaps,
        "configs": configs,
        "frozen_horizons_frames": [24, 48, 96],
        "fastfolding_horizons": [1, 4, 8, 16],
        "force_degeneracy_threshold": 1.0e-6,
        "force_bandwidths_to_check": [0.05, 0.1, 0.0],
        "warnings": warnings,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "protocol_audit.md").write_text(
        _render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
