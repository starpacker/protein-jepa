import json

import numpy as np
import pytest
import torch

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset


def _dataset(tmp_path, **extra):
    np.savez(tmp_path/'x.npz', coords=np.zeros((4,6,4,3),dtype=np.float32))
    row = {'npz_path':str(tmp_path/'x.npz'),'protein_id':'p','trajectory_id':'t',
           'num_frames':4,'temperature_k':320., **extra}
    (tmp_path/'m.jsonl').write_text(json.dumps(row)+'\n')
    return build_dataset(DataConfig(manifest_path=str(tmp_path/'m.jsonl'),num_frames=4))


def test_missing_time_units_are_not_silently_frame_indices(tmp_path):
    with pytest.raises(ValueError, match='picoseconds'):
        _dataset(tmp_path)[0]


def test_manifest_time_offset_is_preserved(tmp_path):
    sample = _dataset(tmp_path,frame_interval_ps=1000.,first_frame_time_ps=500.)[0]
    assert torch.equal(sample['time_ps'],torch.tensor([500.,1500.,2500.,3500.]))


def test_missing_temperature_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='temperature_k'):
        _dataset(tmp_path,frame_interval_ps=1000.,temperature_k=None)[0]
