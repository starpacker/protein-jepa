from types import SimpleNamespace
from pathlib import Path

from protein_jepa_frame.data.manifest import TrajectoryRecord
from protein_jepa_frame.data.sampler import HierarchicalClipSampler


class _Dataset:
    def __init__(self):
        self.records = [
            TrajectoryRecord(path=Path("unused"), protein_id="m1", trajectory_id="m1-r1", dataset="mdcath"),
            TrajectoryRecord(path=Path("unused"), protein_id="m2", trajectory_id="m2-r1", dataset="mdcath"),
            TrajectoryRecord(path=Path("unused"), protein_id="a1", trajectory_id="a1-r1", dataset="atlas"),
        ]
        self.entries = [
            SimpleNamespace(record_index=0, count=10),
            SimpleNamespace(record_index=1, count=10),
            SimpleNamespace(record_index=2, count=1000),
        ]
        self.cumulative = [10, 20, 1020]


def test_hierarchical_sampler_obeys_dataset_weights_not_clip_counts():
    dataset = _Dataset()
    sampler = HierarchicalClipSampler(
        dataset,
        num_samples=4000,
        seed=7,
        dataset_weights={"mdcath": 0.75, "atlas": 0.25},
    )

    indices = list(sampler)
    atlas_fraction = sum(index >= 20 for index in indices) / len(indices)
    m1 = sum(index < 10 for index in indices)
    m2 = sum(10 <= index < 20 for index in indices)

    assert 0.22 < atlas_fraction < 0.28
    assert abs(m1 - m2) < 0.08 * len(indices)
    assert all(0 <= index < 1020 for index in indices)
