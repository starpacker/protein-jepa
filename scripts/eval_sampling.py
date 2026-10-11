"""Shared protein-stratified sampling for ABC evaluation scripts."""

from __future__ import annotations

from collections import defaultdict
import random


def stratified_indices(dataset, n: int, seed: int = 42) -> list[int]:
    """Return at most ``n`` dataset indices in protein round-robin order.

    Dataset order is typically grouped by protein/temperature.  Taking the
    first N items therefore over-represents the first protein.  ``_locate`` is
    the canonical dataset lookup already used by the paired dataset, so this
    helper does not read trajectory coordinates just to construct the sample.
    """
    n = min(max(int(n), 0), len(dataset))
    if n == 0:
        return []
    groups: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        record = dataset._locate(index)[0]
        groups[str(record.protein_id)].append(index)
    rng = random.Random(seed)
    protein_ids = sorted(groups)
    for protein_id in protein_ids:
        rng.shuffle(groups[protein_id])
    rng.shuffle(protein_ids)

    selected: list[int] = []
    cursor = 0
    while len(selected) < n and protein_ids:
        next_ids = []
        for protein_id in protein_ids:
            values = groups[protein_id]
            if cursor < len(values):
                selected.append(values[cursor])
                if len(selected) >= n:
                    break
            if cursor + 1 < len(values):
                next_ids.append(protein_id)
        protein_ids = next_ids
        cursor += 1
    return selected


__all__ = ["stratified_indices"]
