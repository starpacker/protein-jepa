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
    lengths = torch.zeros(B, dtype=torch.long)
    has_msm = "msm_state_labels" in samples[0]
    if has_msm:
        msm_state_labels = torch.full((B, T), -1, dtype=torch.long)

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
        if has_msm:
            msm_state_labels[i] = sample["msm_state_labels"]

    out = {
        "coords": coords,
        "atom_mask": atom_mask,
        "residue_mask": residue_mask,
        "aatype": aatype,
        "residue_index": residue_index,
        "chain_id": chain_id,
        "time_ps": time_ps,
        "lengths": lengths,
        "protein_id": [sample["protein_id"] for sample in samples],
        "trajectory_id": [sample["trajectory_id"] for sample in samples],
    }
    if has_msm:
        out["msm_state_labels"] = msm_state_labels
    return out
