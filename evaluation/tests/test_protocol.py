from types import SimpleNamespace
import numpy as np
import pytest
from evaluation.data import EvalProtocol, EvalWindow, _context_offsets, _select_trajectories


def test_distinct_protein_cap_keeps_selected_replicas():
    records = [SimpleNamespace(protein_id=p, temperature_k=320.) for p in ['a','a','b','b','c']]
    out = _select_trajectories(records, EvalProtocol('x', max_proteins=2))
    assert [r.protein_id for r in out] == ['a','a','b','b']


def test_stride_and_nonpositive_horizons():
    assert list(_context_offsets(EvalProtocol('x', context_frames=3, frame_stride=2))) == [-4,-2,0]
    with pytest.raises(ValueError): EvalProtocol('x', horizons_frames=(0,))


def test_deliberate_target_leak_is_rejected():
    w = EvalWindow('a','r',320.,7,1,1000.,{}, {}, np.arange(8),7)
    with pytest.raises(ValueError, match='strictly after'):
        w.validate()
