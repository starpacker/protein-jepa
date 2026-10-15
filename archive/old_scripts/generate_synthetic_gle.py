#!/usr/bin/env python3
"""S1: Generate synthetic double-well GLE trajectories with known A*, D*, K*."""

from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np


def double_well_potential(z: np.ndarray, a: float = 4.0, b: float = 2.0) -> np.ndarray:
    return a * (z ** 2 - b) ** 2


def double_well_force(z: np.ndarray, a: float = 4.0, b: float = 2.0) -> np.ndarray:
    return -4.0 * a * z * (z ** 2 - b)


def generate_gle_trajectory(n_steps: int, dt: float, temperature: float,
                             a: float = 4.0, b: float = 2.0,
                             diffusion: float = 1.0,
                             kernel_decay: float = 0.5,
                             kernel_strength: float = 0.3,
                             seed: int = 42, auxiliary_dim: int = 4):
    rng = np.random.default_rng(seed)
    gamma, kappa = kernel_decay, kernel_strength
    z = np.zeros(n_steps, dtype=np.float64)
    eta = np.zeros((n_steps, auxiliary_dim), dtype=np.float64)
    z[0] = rng.normal(0, 1.0)
    noise_scale = np.sqrt(2.0 * diffusion * temperature / dt)
    aux_noise_scale = np.sqrt(2.0 * gamma * temperature / dt)
    for t in range(1, n_steps):
        force = double_well_force(z[t - 1], a, b)
        aux_coupling = kappa * eta[t - 1].sum()
        z[t] = z[t - 1] + dt * (diffusion * (force + aux_coupling) / temperature
                                 + noise_scale * rng.normal())
        eta[t] = (eta[t - 1] - dt * gamma * eta[t - 1]
                  + aux_noise_scale * rng.normal(size=auxiliary_dim))
    t_vals = np.arange(n_steps) * dt
    kernel_analytic = (kappa ** 2 * auxiliary_dim / gamma) * np.exp(-gamma * t_vals)
    z_grid = np.linspace(-3.0, 3.0, 200)
    free_energy_analytic = double_well_potential(z_grid, a, b)
    return z, free_energy_analytic, kernel_analytic, z_grid, t_vals


def save_ground_truth(output_dir: Path, free_energy, kernel, z_grid, t_vals, params):
    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(gt_dir / "analytics.npz",
             free_energy=free_energy, kernel=kernel,
             z_grid=z_grid, t_vals=t_vals, params=str(params))
    with open(gt_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {"n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test,
              "traj_length": args.traj_length, "dt": args.dt,
              "temperature": args.temperature, "a": args.double_well_depth,
              "b": args.double_well_barrier, "diffusion": args.diffusion_scale,
              "kernel_decay": args.memory_kernel_decay,
              "kernel_strength": args.memory_kernel_strength,
              "base_seed": args.seed, "observation_noise": args.observation_noise}
    z, free_energy, kernel, z_grid, t_vals = generate_gle_trajectory(
        n_steps=args.traj_length, dt=args.dt, temperature=args.temperature,
        a=args.double_well_depth, b=args.double_well_barrier,
        diffusion=args.diffusion_scale,
        kernel_decay=args.memory_kernel_decay,
        kernel_strength=args.memory_kernel_strength, seed=args.seed)
    save_ground_truth(output_dir, free_energy, kernel, z_grid, t_vals, params)
    for split, count in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        trajectories = []
        for i in range(count):
            traj_seed = args.seed + 100 + hash(split) % 1000 + i
            z_i, _, _, _, _ = generate_gle_trajectory(
                n_steps=args.traj_length, dt=args.dt, temperature=args.temperature,
                a=args.double_well_depth, b=args.double_well_barrier,
                diffusion=args.diffusion_scale,
                kernel_decay=args.memory_kernel_decay,
                kernel_strength=args.memory_kernel_strength, seed=traj_seed)
            if args.observation_noise > 0:
                z_i += np.random.default_rng(traj_seed + 1000).normal(0, args.observation_noise, size=z_i.shape)
            traj_path = output_dir / "trajectories" / f"{split}_{i:03d}.npy"
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(traj_path, z_i)
            trajectories.append({"path": str(traj_path.relative_to(output_dir)),
                                 "split": split, "seed": traj_seed,
                                 "n_steps": args.traj_length, "dt": args.dt,
                                 "temperature": args.temperature, "index": i})
        manifest_path = output_dir / f"{split}.jsonl"
        with open(manifest_path, "w") as f:
            for traj in trajectories:
                f.write(json.dumps(traj) + "\n")
        print(f"  {split}: {count} trajectories -> {manifest_path}")
    with open(output_dir / "generation_params.json", "w") as f:
        json.dump(params, f, indent=2, default=str)
    print(f"Synthetic GLE data saved to {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate synthetic double-well GLE data")
    p.add_argument("--output-dir", default="/liuyutian/protein_jepa_frame/artifacts/reviewer_experiments/synthetic_gle")
    p.add_argument("--n-train", type=int, default=5)
    p.add_argument("--n-val", type=int, default=2)
    p.add_argument("--n-test", type=int, default=3)
    p.add_argument("--traj-length", type=int, default=10000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--double-well-depth", type=float, default=2.0)
    p.add_argument("--double-well-barrier", type=float, default=4.0)
    p.add_argument("--diffusion-scale", type=float, default=1.0)
    p.add_argument("--memory-kernel-decay", type=float, default=0.5)
    p.add_argument("--memory-kernel-strength", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--observation-noise", type=float, default=0.01)
    main(p.parse_args())
