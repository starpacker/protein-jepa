import random
import numpy as np
import torch
from types import SimpleNamespace
from protein_jepa_frame.training.lasp_run_state import budget_subset,rng_state,restore_rng,step_batch


def test_full_manifest_and_large_population_sampling():
    dataset=range(10**9)
    assert budget_subset(dataset,0,True,42) is dataset
    subset=budget_subset(dataset,20,True,42)
    assert len(set(subset.indices))==20
    assert subset.indices==budget_subset(dataset,20,True,42).indices
    assert subset.indices!=budget_subset(dataset,20,True,43).indices


def test_rng_recovery_replays_training_randomness():
    state=rng_state()
    expected=(random.random(),np.random.rand(3),torch.rand(4))
    restore_rng(state)
    actual=(random.random(),np.random.rand(3),torch.rand(4))
    assert actual[0]==expected[0]
    np.testing.assert_array_equal(actual[1],expected[1])
    torch.testing.assert_close(actual[2],expected[2],atol=0,rtol=0)


def test_step_indexed_batches_do_not_depend_on_previous_reads(monkeypatch):
    import protein_jepa_frame.training.lasp_run_state as module
    monkeypatch.setattr(module,'collate_clips',lambda values:values)
    loader=SimpleNamespace(dataset=range(100000),batch_size=8)
    expected=step_batch(loader,200,42)
    for step in range(1,200):step_batch(loader,step,42)
    assert step_batch(loader,200,42)==expected
    assert step_batch(loader,200,43)!=expected
