#!/usr/bin/env python3
"""Embed the analytic short-time closure gate in an EVA5 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate-steps", type=float, required=True)
    args = parser.parse_args()
    if args.gate_steps <= 0.0:
        raise ValueError("gate-steps must be positive")

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("checkpoint does not contain a serialized model config")
    config["model"]["tmjepa_projected_force_horizon_gate_steps"] = args.gate_steps

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output.parent)
    torch.save(checkpoint, output)
    (output.parent / "config.json").write_text(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
