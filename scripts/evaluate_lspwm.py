"""Run the deterministic LSP-WM contract evaluation on synthetic trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from protein_jepa_frame.models.lspwm import (
    LSPWMConfig,
    LSPWMWorldModel,
    koopman_diagnostics,
)


def _synthetic_pairs(
    samples: int,
    state_dim: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(samples, state_dim, generator=generator)
    diagonal = torch.linspace(0.92, 0.65, state_dim)
    target = source * diagonal
    target = target + 0.01 * torch.randn(target.shape, generator=generator)
    return source, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="run the built-in contract data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6303)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()
    if not args.synthetic:
        parser.error("only --synthetic is implemented; protein evaluation uses the matched paper protocol")

    torch.manual_seed(args.seed)
    config = LSPWMConfig(
        state_dim=8,
        condition_dim=0,
        auxiliary_dim=3,
        relaxation_times_ps=(1000.0, 2000.0, 4000.0),
        diffusion_scale=1.0e-4,
    )
    model = LSPWMWorldModel(config).eval()
    source, target = _synthetic_pairs(args.samples, config.state_dim, seed=args.seed)
    diagnostics = koopman_diagnostics(source, target, lag_time=1.0)
    rollout = model.rollout(
        source[:8],
        None,
        dt_ps=1.0,
        steps=4,
        temperature_k=348.0,
        stochastic=False,
    )
    coefficients = model.coefficients(source[:8], torch.full((8,), 348.0))
    payload = {
        "framework": "LSP-WM v4.1",
        "protocol": "synthetic_contract",
        "seed": args.seed,
        "samples": args.samples,
        "koopman_residual_mse": float(diagnostics["residual_mse"]),
        "koopman_vamp2": float(diagnostics["vamp2_score"]),
        "koopman_max_implied_timescale": float(
            diagnostics["implied_timescales"].real.max()
        ),
        "diffusion_min_eigenvalue": float(
            torch.linalg.eigvalsh(coefficients["diffusion"]).min().detach()
        ),
        "rollout_finite": bool(torch.isfinite(rollout["q"]).all()),
        "rollout_shape": list(rollout["q"].shape),
        "claim_boundary": "synthetic contract only; not a protein benchmark result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
