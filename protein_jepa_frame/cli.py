from __future__ import annotations

import argparse

from protein_jepa_frame.fastfolding.manifest import build_manifest
from protein_jepa_frame.fastfolding.tasks import derive_labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="protein-jepa-frame")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", required=True)

    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--output-dir", default="artifacts/smoke")
    smoke_parser.add_argument("--steps", type=int, default=5)

    ff_manifest = sub.add_parser("fastfolding-manifest")
    ff_manifest.add_argument("--root", default="/liuyutian/folddata/FastFolding")
    ff_manifest.add_argument("--output", default="artifacts/fastfolding/manifest.jsonl")
    ff_manifest.add_argument("--max-scan-per-protein", type=int, default=None)
    ff_manifest.add_argument("--protein", action="append", default=None)

    ff_labels = sub.add_parser("fastfolding-labels")
    ff_labels.add_argument("--manifest", default="artifacts/fastfolding/manifest.jsonl")
    ff_labels.add_argument("--output", default="artifacts/fastfolding/frame_labels.jsonl")
    ff_labels.add_argument("--stride", type=int, default=100)
    ff_labels.add_argument("--max-frames-per-protein", type=int, default=2000)
    ff_labels.add_argument("--max-scan-entries-per-protein", type=int, default=None)
    ff_labels.add_argument("--folded-q", type=float, default=0.8)
    ff_labels.add_argument("--unfolded-q", type=float, default=0.2)
    ff_labels.add_argument("--direction-threshold", type=float, default=0.02)
    ff_labels.add_argument("--committor-horizon", type=int, default=20)

    ff_probe = sub.add_parser("fastfolding-probe")
    ff_probe.add_argument("--checkpoint", required=True)
    ff_probe.add_argument("--labels", default="artifacts/fastfolding/frame_labels.jsonl")
    ff_probe.add_argument("--output", default="artifacts/fastfolding/downstream_probe.json")
    ff_probe.add_argument("--num-frames", type=int, default=8)
    ff_probe.add_argument("--readout", choices=["mean", "state", "clip_state"], default="state")
    ff_probe.add_argument("--batch-size", type=int, default=4)
    ff_probe.add_argument("--max-samples", type=int, default=200)
    ff_probe.add_argument("--max-residues", type=int, default=384)
    ff_probe.add_argument("--split-gap", type=int, default=1)
    ff_probe.add_argument("--device", default="auto")
    ff_probe.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])

    args = parser.parse_args(argv)
    if args.cmd == "train":
        from protein_jepa_frame.config import load_config
        from protein_jepa_frame.training.train import train

        train(load_config(args.config))
        return
    if args.cmd == "smoke":
        from protein_jepa_frame.config import DataConfig, OptimizationConfig, TrainConfig
        from protein_jepa_frame.training.train import train

        cfg = TrainConfig(
            output_dir=args.output_dir,
            data=DataConfig(batch_size=2, num_workers=0, num_frames=4, synthetic_size=16, synthetic_residues=24),
            optimization=OptimizationConfig(epochs=1, steps_per_epoch=args.steps, warmup_steps=1),
            log_every=1,
        )
        train(cfg)
        return
    if args.cmd == "fastfolding-manifest":
        records = build_manifest(
            args.root,
            args.output,
            max_scan_per_protein=args.max_scan_per_protein,
            protein_ids=set(args.protein) if args.protein else None,
        )
        print(f"wrote {len(records)} FastFolding records to {args.output}", flush=True)
        return
    if args.cmd == "fastfolding-labels":
        summary = derive_labels(
            args.manifest,
            args.output,
            stride=args.stride,
            max_frames_per_protein=args.max_frames_per_protein,
            max_scan_entries_per_protein=args.max_scan_entries_per_protein,
            folded_q=args.folded_q,
            unfolded_q=args.unfolded_q,
            direction_threshold=args.direction_threshold,
            committor_horizon=args.committor_horizon,
        )
        print(summary, flush=True)
        return
    if args.cmd == "fastfolding-probe":
        from protein_jepa_frame.fastfolding.downstream import run_probes

        out = run_probes(
            args.checkpoint,
            args.labels,
            args.output,
            num_frames=args.num_frames,
            readout=args.readout,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_residues=args.max_residues,
            split_gap=args.split_gap,
            device=args.device,
            lambdas=args.lambdas,
        )
        print({"output": args.output, "n_windows": out["n_windows"], "tasks": out["tasks"]}, flush=True)
        return
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
