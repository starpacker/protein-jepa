"""Deterministic step-indexed batches and restart state for long LASP runs."""
import random
import numpy as np
import torch
from torch.utils.data import Subset
from protein_jepa_frame.data.collate import collate_clips


def step_batch(loader,step,seed):
    rng=random.Random(seed+1000003*step)
    return collate_clips([loader.dataset[rng.randrange(len(loader.dataset))]
                         for _ in range(loader.batch_size)])


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def budget_subset(dataset,budget,training,seed):
    # A zero budget means access the entire manifest-backed window dataset.
    if budget==0:return dataset
    n=min(len(dataset),budget)
    if training:
        indices=random.Random(seed).sample(range(len(dataset)),n)
    else:indices=torch.linspace(0,len(dataset)-1,n).long().tolist()
    return Subset(dataset,indices)
