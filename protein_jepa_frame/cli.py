from __future__ import annotations

import argparse
from dataclasses import replace

from protein_jepa_frame.config import DataConfig, OptimizationConfig, TrainConfig, load_config
from protein_jepa_frame.training.train import train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="protein-jepa-frame")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)

    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--output-dir", default="artifacts/smoke")
    smoke_parser.add_argument("--steps", type=int, default=5)

    args = parser.parse_args(argv)
    if args.cmd == "train":
        train(load_config(args.config))
        return
    if args.cmd == "smoke":
        cfg = TrainConfig(
            output_dir=args.output_dir,
            data=DataConfig(batch_size=2, num_workers=0, num_frames=4, synthetic_size=16, synthetic_residues=24),
            optimization=OptimizationConfig(epochs=1, steps_per_epoch=args.steps, warmup_steps=1),
            log_every=1,
        )
        train(cfg)
        return
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
