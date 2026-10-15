from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protein_jepa_frame.config import model_config_from_dict
from protein_jepa_frame.models.jepa import ProteinJEPA, upgrade_tmjepa_v3_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bias", type=float, default=0.031)
    parser.add_argument("--log-horizon-slope", type=float, default=-0.074)
    parser.add_argument("--short-horizon-slope", type=float, default=0.10)
    parser.add_argument("--temperature-slope", type=float, default=0.0)
    parser.add_argument("--short-temperature-slope", type=float, default=-0.13)
    parser.add_argument("--calibration-scale", type=float, default=0.5)
    parser.add_argument("--temperature-scale", type=float, default=6.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint)
    config = checkpoint["config"]
    model_values = dict(config["model"])
    model_values.update(
        {
            "tmjepa_endpoint_calibration_enabled": True,
            "tmjepa_endpoint_calibration_scale": args.calibration_scale,
            "tmjepa_endpoint_calibration_temperature_scale": args.temperature_scale,
        }
    )
    model = ProteinJEPA(model_config_from_dict(model_values, state))
    model.load_state_dict(upgrade_tmjepa_v3_state_dict(model, state), strict=True)
    calibrator = model.tmjepa.endpoint_calibration
    with torch.no_grad():
        calibrator.bias.fill_(args.bias)
        calibrator.log_horizon_slope.fill_(args.log_horizon_slope)
        calibrator.short_horizon_slope.fill_(args.short_horizon_slope)
        calibrator.temperature_slope.fill_(args.temperature_slope)
        calibrator.short_temperature_slope.fill_(args.short_temperature_slope)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "config": {
            **config,
            "model": model_values,
            "output_dir": str(output.parent),
            "init_checkpoint": args.base_checkpoint,
        },
        "analytic_calibration": {
            "bias": args.bias,
            "log_horizon_slope": args.log_horizon_slope,
            "short_horizon_slope": args.short_horizon_slope,
            "temperature_slope": args.temperature_slope,
            "short_temperature_slope": args.short_temperature_slope,
            "calibration_scale": args.calibration_scale,
            "temperature_scale": args.temperature_scale,
        },
    }
    torch.save(payload, output)
    print(output)


if __name__ == "__main__":
    main()
