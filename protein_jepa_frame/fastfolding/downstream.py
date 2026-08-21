from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import AA_TO_INDEX, BACKBONE_ATOMS, THREE_TO_ONE
from protein_jepa_frame.eval.paper_eval import _device, _load_checkpoint_spec, _model_from_spec


@dataclass(frozen=True)
class BackboneFrame:
    coords: np.ndarray
    atom_mask: np.ndarray
    aatype: np.ndarray
    residue_index: np.ndarray
    chain_id: np.ndarray


def load_label_rows(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: (row["protein_id"], row.get("trajectory_id", row["protein_id"]), int(row["frame"])))
    return rows


def read_backbone_frame(path: str | Path) -> BackboneFrame:
    residues: dict[tuple[str, int, str], dict] = {}
    chain_names: dict[str, int] = {}
    with Path(path).open("r", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16:17].strip()
            if atom_name not in BACKBONE_ATOMS or altloc not in {"", "A"}:
                continue
            residue_name = line[17:20].strip().upper()
            chain = line[21:22].strip() or "_"
            resseq = int(line[22:26])
            icode = line[26:27].strip()
            key = (chain, resseq, icode)
            chain_names.setdefault(chain, len(chain_names))
            residue = residues.setdefault(
                key,
                {"name": residue_name, "atoms": {}, "chain": chain, "number": resseq},
            )
            residue["atoms"].setdefault(
                atom_name,
                np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32),
            )
    if not residues:
        raise ValueError(f"{path} contains no backbone atoms")
    L = len(residues)
    coords = np.zeros((L, len(BACKBONE_ATOMS), 3), dtype=np.float32)
    atom_mask = np.zeros((L, len(BACKBONE_ATOMS)), dtype=bool)
    aatype = np.full((L,), 20, dtype=np.int64)
    residue_index = np.zeros((L,), dtype=np.int64)
    chain_id = np.zeros((L,), dtype=np.int64)
    for i, residue in enumerate(residues.values()):
        aatype[i] = AA_TO_INDEX.get(THREE_TO_ONE.get(residue["name"], "X"), 20)
        residue_index[i] = int(residue["number"])
        chain_id[i] = chain_names[str(residue["chain"])]
        for j, atom in enumerate(BACKBONE_ATOMS):
            value = residue["atoms"].get(atom)
            if value is not None:
                coords[i, j] = value
                atom_mask[i, j] = True
    return BackboneFrame(coords=coords, atom_mask=atom_mask, aatype=aatype, residue_index=residue_index, chain_id=chain_id)


def build_window_samples(rows: list[dict], num_frames: int, max_samples: int | None = None) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["protein_id"], row.get("trajectory_id", row["protein_id"]))].append(row)
    samples = []
    for (_, _), group in grouped.items():
        group.sort(key=lambda row: int(row["frame"]))
        for end in range(num_frames - 1, len(group)):
            window = group[end - num_frames + 1 : end + 1]
            samples.append({"target": group[end], "window": window})
    if max_samples is not None and len(samples) > max_samples:
        rng = np.random.RandomState(0)
        picked = sorted(rng.choice(len(samples), max_samples, replace=False).tolist())
        samples = [samples[i] for i in picked]
    return samples


def _frame_to_sample(frame: BackboneFrame, row: dict, time_value: float) -> dict:
    residue_mask = frame.atom_mask.all(axis=-1)
    return {
        "coords": torch.from_numpy(frame.coords[None]),
        "atom_mask": torch.from_numpy(frame.atom_mask[None]),
        "residue_mask": torch.from_numpy(residue_mask[None]),
        "aatype": torch.from_numpy(frame.aatype),
        "residue_index": torch.from_numpy(frame.residue_index),
        "chain_id": torch.from_numpy(frame.chain_id),
        "time_ps": torch.tensor([time_value], dtype=torch.float32),
        "protein_id": row["protein_id"],
        "trajectory_id": row.get("trajectory_id", row["protein_id"]),
    }


def _window_to_clip(sample: dict, cache: dict[str, BackboneFrame], max_residues: int | None) -> dict:
    frame_samples = []
    for row in sample["window"]:
        path = str(row["pdb_path"])
        frame = cache.get(path)
        if frame is None:
            frame = read_backbone_frame(path)
            if max_residues is not None:
                frame = BackboneFrame(
                    coords=frame.coords[:max_residues],
                    atom_mask=frame.atom_mask[:max_residues],
                    aatype=frame.aatype[:max_residues],
                    residue_index=frame.residue_index[:max_residues],
                    chain_id=frame.chain_id[:max_residues],
                )
            cache[path] = frame
        frame_samples.append(_frame_to_sample(frame, row, float(row["frame"])))
    coords = torch.cat([item["coords"] for item in frame_samples], dim=0)
    atom_mask = torch.cat([item["atom_mask"] for item in frame_samples], dim=0)
    residue_mask = torch.cat([item["residue_mask"] for item in frame_samples], dim=0)
    first = frame_samples[-1]
    return {
        "coords": coords,
        "atom_mask": atom_mask,
        "residue_mask": residue_mask,
        "aatype": first["aatype"],
        "residue_index": first["residue_index"],
        "chain_id": first["chain_id"],
        "time_ps": torch.tensor([float(row["frame"]) for row in sample["window"]], dtype=torch.float32),
        "protein_id": sample["target"]["protein_id"],
        "trajectory_id": sample["target"].get("trajectory_id", sample["target"]["protein_id"]),
    }


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def extract_contextual_latents(
    checkpoint: str,
    label_rows: list[dict],
    num_frames: int,
    readout: str,
    batch_size: int,
    max_samples: int | None,
    max_residues: int | None,
    device_name: str,
) -> tuple[np.ndarray, list[dict], dict]:
    device = _device(device_name)
    spec = _load_checkpoint_spec(checkpoint)
    model, cfg = _model_from_spec(spec, device)
    samples = build_window_samples(label_rows, num_frames=num_frames, max_samples=max_samples)
    cache: dict[str, BackboneFrame] = {}
    latents = []
    targets = []
    for start in range(0, len(samples), batch_size):
        clips = [_window_to_clip(sample, cache, max_residues=max_residues) for sample in samples[start : start + batch_size]]
        batch = _move_batch(collate_clips(clips), device)
        residue_latents = model.context_encoder(
            coords=batch["coords"],
            residue_mask=batch["residue_mask"],
            aatype=batch["aatype"],
            residue_index=batch["residue_index"],
            chain_id=batch["chain_id"],
            token_mask=None,
        )
        if readout == "mean":
            valid = batch["residue_mask"][:, -1].unsqueeze(-1).float()
            pooled = (residue_latents[:, -1] * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            states = model.encode_states(residue_latents, batch["residue_mask"])
            if readout == "state":
                pooled = states["frame_states"][:, -1]
            elif readout == "clip_state":
                pooled = states["clip_state"]
            else:
                raise ValueError(f"unknown readout {readout}")
        latents.append(pooled.detach().cpu().numpy().astype(np.float64))
        targets.extend(sample["target"] for sample in samples[start : start + batch_size])
    if not latents:
        raise ValueError("no FastFolding windows could be built")
    return np.concatenate(latents, axis=0), targets, cfg


def _standardize(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    keep = std.reshape(-1) > 1.0e-8
    return (train[:, keep] - mean[:, keep]) / std[:, keep], (val[:, keep] - mean[:, keep]) / std[:, keep]


def _blocked_split(n: int, train_fraction: float = 0.7, gap: int = 1) -> tuple[np.ndarray, np.ndarray]:
    split = max(1, min(n - 1, int(round(n * train_fraction))))
    train_end = max(1, split - gap)
    val_start = min(n, split + gap)
    return np.arange(train_end), np.arange(val_start, n)


def _ridge_regression(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, lambdas: list[float]) -> dict:
    train_x, val_x = _standardize(train_x, val_x)
    x_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    v_aug = np.concatenate([val_x, np.ones((val_x.shape[0], 1))], axis=1)
    denom = float(np.mean((val_y - train_y.mean(axis=0, keepdims=True)) ** 2))
    if denom <= 1.0e-12:
        return {}
    best = {"r2": -np.inf}
    for lam in lambdas:
        reg = lam * np.eye(x_aug.shape[1])
        reg[-1, -1] = 0.0
        weights = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ train_y
        pred = v_aug @ weights
        mse = float(np.mean((pred - val_y) ** 2))
        r2 = 1.0 - mse / denom
        if r2 > best["r2"]:
            best = {"r2": float(r2), "mse": mse, "lambda": lam}
    return best


def _ridge_classifier(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, lambdas: list[float]) -> dict:
    classes = np.array(sorted(set(train_y.tolist()) | set(val_y.tolist())))
    train_x, val_x = _standardize(train_x, val_x)
    x_aug = np.concatenate([train_x, np.ones((train_x.shape[0], 1))], axis=1)
    v_aug = np.concatenate([val_x, np.ones((val_x.shape[0], 1))], axis=1)
    y_onehot = (train_y[:, None] == classes[None, :]).astype(np.float64)
    best = {"accuracy": -1.0}
    for lam in lambdas:
        reg = lam * np.eye(x_aug.shape[1])
        reg[-1, -1] = 0.0
        weights = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ y_onehot
        pred = classes[np.argmax(v_aug @ weights, axis=1)]
        acc = float(np.mean(pred == val_y))
        if acc > best["accuracy"]:
            values, counts = np.unique(val_y, return_counts=True)
            baseline = float(counts.max() / max(1, counts.sum()))
            best = {"accuracy": acc, "baseline": baseline, "lambda": lam, "classes": classes.tolist()}
    return best


def run_probes(
    checkpoint: str,
    labels: str | Path,
    output: str | Path,
    num_frames: int = 8,
    readout: str = "state",
    batch_size: int = 4,
    max_samples: int | None = 200,
    max_residues: int | None = 384,
    split_gap: int = 1,
    device: str = "auto",
    lambdas: list[float] | None = None,
) -> dict:
    lambdas = lambdas or [0.01, 0.1, 1.0, 10.0, 100.0]
    rows = load_label_rows(labels)
    x, target_rows, cfg = extract_contextual_latents(
        checkpoint,
        rows,
        num_frames=num_frames,
        readout=readout,
        batch_size=batch_size,
        max_samples=max_samples,
        max_residues=max_residues,
        device_name=device,
    )
    train_idx, val_idx = _blocked_split(len(target_rows), gap=split_gap)
    if len(train_idx) < 3 or len(val_idx) < 3:
        raise ValueError("not enough windows for blocked train/validation split")
    train_x, val_x = x[train_idx], x[val_idx]
    out = {
        "protocol": {
            "checkpoint": checkpoint,
            "labels": str(labels),
            "num_frames": num_frames,
            "readout": readout,
            "max_samples": max_samples,
            "max_residues": max_residues,
            "split": "blocked chronological 70/30 with gap",
            "split_gap": split_gap,
            "lambdas": lambdas,
        },
        "checkpoint_step": cfg.get("_checkpoint_step"),
        "n_windows": len(target_rows),
        "feature_dim": int(x.shape[1]),
        "tasks": {},
    }
    y_progress = np.asarray([[row["q_native"], row["rmsd_to_ref"], row["radius_of_gyration"]] for row in target_rows], dtype=np.float64)
    out["tasks"]["progress_regression"] = _ridge_regression(train_x, y_progress[train_idx], val_x, y_progress[val_idx], lambdas)
    directions = np.asarray([row["direction"] for row in target_rows], dtype=object)
    keep_direction = directions != "stationary"
    direction_train = train_idx[keep_direction[train_idx]]
    direction_val = val_idx[keep_direction[val_idx]]
    if len(direction_train) >= 3 and len(direction_val) >= 3:
        out["tasks"]["direction_classification"] = _ridge_classifier(
            x[direction_train],
            directions[direction_train],
            x[direction_val],
            directions[direction_val],
            lambdas,
        )
    comm = np.asarray([row.get("committor_proxy") for row in target_rows], dtype=object)
    keep_comm = np.asarray([value is not None for value in comm], dtype=bool)
    comm_train = train_idx[keep_comm[train_idx]]
    comm_val = val_idx[keep_comm[val_idx]]
    if len(comm_train) >= 3 and len(comm_val) >= 3:
        out["tasks"]["committor_classification"] = _ridge_classifier(
            x[comm_train],
            comm[comm_train].astype(int),
            x[comm_val],
            comm[comm_val].astype(int),
            lambdas,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run FastFolding contextual latent downstream probes")
    parser.add_argument("--checkpoint", required=True, help="name=/path/checkpoint.pt")
    parser.add_argument("--labels", type=Path, default=Path("artifacts/fastfolding/frame_labels.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fastfolding/downstream_probe.json"))
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--readout", choices=["mean", "state", "clip_state"], default="state")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-residues", type=int, default=384)
    parser.add_argument("--split-gap", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0, 100.0])
    args = parser.parse_args(argv)
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
    print(json.dumps({"output": str(args.output), "n_windows": out["n_windows"], "tasks": out["tasks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
