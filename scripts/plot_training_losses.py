from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_FIELDS = (
    "loss",
    "state_jepa_loss",
    "lagged_state_jepa_loss",
    "state_transition_jepa_loss",
    "semigroup_loss",
    "residue_jepa_loss",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("{"):
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"no JSON records found in {path}")
    return records


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    half = window // 2
    out = []
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        out.append(sum(values[lo:hi]) / float(hi - lo))
    return out


def _mean_tail(records: list[dict[str, Any]], field: str, count: int) -> float:
    tail = records[-min(count, len(records)) :]
    return sum(float(record[field]) for record in tail) / float(len(tail))


def _nested_keys(records: list[dict[str, Any]], field: str) -> list[str]:
    keys = {str(key) for record in records for key in record.get(field, {})}
    try:
        return sorted(keys, key=lambda value: int(value))
    except ValueError:
        return sorted(keys)


def _nested_series(records: list[dict[str, Any]], field: str, key: str) -> tuple[list[float], list[float]]:
    points = [
        (float(record["step"]), float(record[field][key]))
        for record in records
        if key in record.get(field, {})
    ]
    return [point[0] for point in points], [point[1] for point in points]


def _plot_dynamics_details(
    records: list[dict[str, Any]], prefix: Path, steps_per_epoch: int, smooth_window: int
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    lag_keys = _nested_keys(records, "lagged_state_loss_by_lag")
    persistence_keys = _nested_keys(records, "lagged_state_persistence_by_lag")
    intrinsic_keys = _nested_keys(records, "lagged_state_intrinsic_persistence_by_lag")
    skill_keys = _nested_keys(records, "lagged_state_skill_by_lag")
    composition_keys = _nested_keys(records, "semigroup_loss_by_composition")

    for key in lag_keys:
        steps, values = _nested_series(records, "lagged_state_loss_by_lag", key)
        epochs = [step / float(steps_per_epoch) for step in steps]
        axes[0].plot(epochs, _moving_average(values, smooth_window), label=f"lag {key}", linewidth=1.8)
    for key in persistence_keys:
        steps, values = _nested_series(records, "lagged_state_persistence_by_lag", key)
        epochs = [step / float(steps_per_epoch) for step in steps]
        axes[0].plot(
            epochs,
            _moving_average(values, smooth_window),
            label=f"persistence {key}",
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
        )
    for key in intrinsic_keys:
        steps, values = _nested_series(records, "lagged_state_intrinsic_persistence_by_lag", key)
        epochs = [step / float(steps_per_epoch) for step in steps]
        axes[0].plot(
            epochs,
            _moving_average(values, smooth_window),
            label=f"intrinsic {key}",
            linewidth=0.8,
            linestyle=":",
            alpha=0.65,
        )
    axes[0].set_ylabel("Smooth L1")
    axes[0].set_title("Lagged-state prediction by horizon")
    axes[0].grid(True, alpha=0.25)
    if lag_keys or persistence_keys:
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    for key in skill_keys:
        steps, values = _nested_series(records, "lagged_state_skill_by_lag", key)
        epochs = [step / float(steps_per_epoch) for step in steps]
        axes[1].plot(epochs, _moving_average(values, smooth_window), label=f"lag {key}", linewidth=1.8)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1].set_ylabel("skill vs persistence")
    axes[1].grid(True, alpha=0.25)
    if skill_keys:
        axes[1].legend(loc="lower right", fontsize=8, ncol=2)
    else:
        axes[1].text(0.5, 0.5, "Skill logging begins with A3", ha="center", va="center", transform=axes[1].transAxes)

    if composition_keys:
        for key in composition_keys:
            steps, values = _nested_series(records, "semigroup_loss_by_composition", key)
            epochs = [step / float(steps_per_epoch) for step in steps]
            axes[2].plot(epochs, _moving_average(values, smooth_window), label=key, linewidth=1.8)
    else:
        values = [float(record["semigroup_loss"]) for record in records if "semigroup_loss" in record]
        steps = [float(record["step"]) for record in records if "semigroup_loss" in record]
        epochs = [step / float(steps_per_epoch) for step in steps]
        axes[2].plot(epochs, _moving_average(values, smooth_window), label="aggregate", linewidth=2.0)
    axes[2].set_xlabel(f"epoch ({steps_per_epoch} steps / epoch)")
    axes[2].set_ylabel("consistency loss")
    axes[2].set_title("Semigroup consistency by composition")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()

    dynamics_path = Path(f"{prefix}.dynamics.png")
    fig.savefig(dynamics_path, dpi=180)
    plt.close(fig)
    return dynamics_path


def plot_losses(
    log_path: Path,
    output_prefix: Path | None,
    fields: tuple[str, ...],
    steps_per_epoch: int,
    smooth_window: int,
) -> dict[str, Any]:
    records = _read_jsonl(log_path)
    available_fields = tuple(field for field in fields if field in records[-1])
    if not available_fields:
        raise ValueError("none of the requested fields are present in the log")

    steps = [int(record["step"]) for record in records]
    epochs = [step / float(steps_per_epoch) for step in steps]
    prefix = output_prefix or log_path.with_suffix("")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    split = min(3, len(available_fields))
    for field in available_fields[:split]:
        values = [float(record[field]) for record in records]
        axes[0].plot(epochs, values, alpha=0.25, linewidth=1.0)
        axes[0].plot(epochs, _moving_average(values, smooth_window), label=f"{field} smoothed", linewidth=2.0)
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training losses")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=9)

    for field in available_fields[split:]:
        values = [float(record[field]) for record in records]
        axes[1].plot(epochs, _moving_average(values, smooth_window), label=f"{field} smoothed", linewidth=2.0)
    axes[1].set_xlabel(f"epoch ({steps_per_epoch} steps / epoch)")
    axes[1].set_ylabel("component value")
    axes[1].grid(True, alpha=0.25)
    if available_fields[split:]:
        axes[1].legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    png_path = prefix.with_suffix(".png")
    csv_path = prefix.with_suffix(".csv")
    summary_path = prefix.with_suffix(".summary.json")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    dynamics_path = _plot_dynamics_details(records, prefix, steps_per_epoch, smooth_window)

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch"] + list(available_fields))
        writer.writeheader()
        for record in records:
            step = int(record["step"])
            row = {"step": step, "epoch": step / float(steps_per_epoch)}
            for field in available_fields:
                row[field] = float(record[field])
            writer.writerow(row)

    summary = {
        "log_path": str(log_path),
        "num_records": len(records),
        "first_step": steps[0],
        "last_step": steps[-1],
        "last_epoch": steps[-1] / float(steps_per_epoch),
        "fields": list(available_fields),
        "latest": {field: float(records[-1][field]) for field in available_fields},
        "tail20_mean": {field: _mean_tail(records, field, 20) for field in available_fields},
        "tail50_mean": {field: _mean_tail(records, field, 50) for field in available_fields},
        "plot": str(png_path),
        "dynamics_plot": str(dynamics_path),
        "csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training losses from train.jsonl")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=5000)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--field", action="append", default=None)
    args = parser.parse_args()

    fields = tuple(args.field) if args.field else DEFAULT_FIELDS
    summary = plot_losses(args.log_path, args.output_prefix, fields, args.steps_per_epoch, args.smooth_window)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
