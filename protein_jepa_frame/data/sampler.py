from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Iterator

from torch.utils.data import Sampler

from protein_jepa_frame.data.dataset import TrajectoryClipDataset


class HierarchicalClipSampler(Sampler[int]):
    """Sample dataset, protein, trajectory, and clip as separate decisions."""

    def __init__(
        self,
        dataset: TrajectoryClipDataset,
        num_samples: int,
        seed: int = 0,
        dataset_weights: dict[str, float] | None = None,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.iteration = 0
        self.groups: OrderedDict[str, OrderedDict[str, list[int]]] = OrderedDict()
        for entry_index, entry in enumerate(dataset.entries):
            record = dataset.records[entry.record_index]
            dataset_name = record.dataset or "unknown"
            proteins = self.groups.setdefault(dataset_name, OrderedDict())
            proteins.setdefault(record.protein_id, []).append(entry_index)
        requested = dataset_weights or {}
        weights = [max(0.0, float(requested.get(name, 1.0))) for name in self.groups]
        if sum(weights) <= 0:
            raise ValueError("at least one available dataset must have positive sampling weight")
        total = sum(weights)
        self.dataset_names = list(self.groups)
        self.dataset_weights = [weight / total for weight in weights]

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.iteration)
        self.iteration += 1
        for _ in range(self.num_samples):
            dataset_name = rng.choices(self.dataset_names, weights=self.dataset_weights, k=1)[0]
            proteins = self.groups[dataset_name]
            protein_id = rng.choice(list(proteins))
            entry_index = rng.choice(proteins[protein_id])
            entry = self.dataset.entries[entry_index]
            previous = self.dataset.cumulative[entry_index - 1] if entry_index else 0
            yield previous + rng.randrange(entry.count)
