from __future__ import annotations

from collections.abc import Sequence

import torch


def collate_clips(samples: Sequence[dict]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    frames = {sample["coords"].shape[0] for sample in samples}
    if len(frames) != 1:
        raise ValueError("all clips in a batch must have the same frame count")
    T = frames.pop()
    max_len = max(sample["coords"].shape[1] for sample in samples)
    B = len(samples)

    coords = torch.zeros(B, T, max_len, 4, 3, dtype=torch.float32)
    atom_mask = torch.zeros(B, T, max_len, 4, dtype=torch.bool)
    residue_mask = torch.zeros(B, T, max_len, dtype=torch.bool)
    aatype = torch.full((B, max_len), 20, dtype=torch.long)
    residue_index = torch.full((B, max_len), -1, dtype=torch.long)
    chain_id = torch.full((B, max_len), -1, dtype=torch.long)
    time_ps = torch.zeros(B, T, dtype=torch.float32)
    temperature_k = torch.full((B,), float("nan"), dtype=torch.float32)
    lengths = torch.zeros(B, dtype=torch.long)
    has_msm = "msm_state_labels" in samples[0]
    has_forces = all("forces" in sample for sample in samples)
    if has_msm:
        msm_state_labels = torch.full((B, T), -1, dtype=torch.long)
    if has_forces:
        forces = torch.zeros(B, T, max_len, 4, 3, dtype=torch.float32)

    for i, sample in enumerate(samples):
        L = sample["coords"].shape[1]
        lengths[i] = L
        coords[i, :, :L] = sample["coords"]
        atom_mask[i, :, :L] = sample["atom_mask"]
        residue_mask[i, :, :L] = sample["residue_mask"]
        aatype[i, :L] = sample["aatype"]
        residue_index[i, :L] = sample["residue_index"]
        chain_id[i, :L] = sample["chain_id"]
        time_ps[i] = sample["time_ps"]
        temperature_k[i] = sample.get("temperature_k", float("nan"))
        if has_msm:
            msm_state_labels[i] = sample["msm_state_labels"]
        if has_forces:
            forces[i, :, :L] = sample["forces"]

    out = {
        "coords": coords,
        "atom_mask": atom_mask,
        "residue_mask": residue_mask,
        "aatype": aatype,
        "residue_index": residue_index,
        "chain_id": chain_id,
        "time_ps": time_ps,
        "temperature_k": temperature_k,
        "dataset": [sample.get("dataset", "unknown") for sample in samples],
        "dataset_index": torch.tensor(
            [{"unknown": 0, "mdcath": 1, "atlas": 2, "synthetic": 3}.get(sample.get("dataset", "unknown"), 0) for sample in samples],
            dtype=torch.long,
        ),
        "lengths": lengths,
        "protein_id": [sample["protein_id"] for sample in samples],
        "trajectory_id": [sample["trajectory_id"] for sample in samples],
    }
    # Gate 1: per-frame physical dt for dynamic encoder conditioning
    out["dt_ps"] = torch.diff(time_ps, dim=1, prepend=time_ps[:, :1])
    if has_msm:
        out["msm_state_labels"] = msm_state_labels
    if has_forces:
        out["forces"] = forces
    return out


def collate_paired_windows(samples: Sequence[dict]) -> dict:
    """Collate paired windows while keeping context/future namespaces explicit."""
    if not samples:
        raise ValueError("cannot collate an empty paired batch")
    grouped = "pairs" in samples[0]
    if grouped:
        if not all("pairs" in sample for sample in samples):
            raise ValueError("cannot mix grouped and ungrouped paired samples")
        group_sizes = [len(sample["pairs"]) for sample in samples]
        if not group_sizes or min(group_sizes) <= 0:
            raise ValueError("grouped paired samples must contain at least one horizon")
        flattened = [pair for sample in samples for pair in sample["pairs"]]
        group_index = torch.repeat_interleave(
            torch.arange(len(samples), dtype=torch.long), torch.tensor(group_sizes, dtype=torch.long)
        )
        samples = flattened
    context = collate_clips([sample["context"] for sample in samples])
    future = collate_clips([sample["future"] for sample in samples])
    if context["trajectory_id"] != future["trajectory_id"]:
        raise ValueError("paired windows must come from the same trajectories")
    out = {}
    for key, value in context.items():
        out[f"context_{key}"] = value
    for key, value in future.items():
        out[f"future_{key}"] = value
    out["horizon_frames"] = torch.stack([sample["horizon_frames"] for sample in samples])
    out["horizon_time_ps"] = torch.stack([sample["horizon_time_ps"] for sample in samples]).float()
    out["temperature_k"] = context["temperature_k"]
    out["dataset_index"] = context["dataset_index"]
    out["dataset"] = context["dataset"]
    out["protein_id"] = context["protein_id"]
    out["trajectory_id"] = context["trajectory_id"]

    # Gate 1 fix: provide per-frame physical dt so dynamic encoder can condition on
    # real physical intervals rather than falling back to 1 ps placeholder.
    if "context_time_ps" in out:
        time_ps = out["context_time_ps"]  # [B, T]
        out["context_dt_ps"] = torch.diff(time_ps, dim=1, prepend=time_ps[:, :1])
    if "future_time_ps" in out:
        time_ps = out["future_time_ps"]  # [B, T]
        out["future_dt_ps"] = torch.diff(time_ps, dim=1, prepend=time_ps[:, :1])

    if grouped:
        out["context_group_index"] = group_index
        out["context_group_size"] = torch.tensor(group_sizes, dtype=torch.long)
    return out
