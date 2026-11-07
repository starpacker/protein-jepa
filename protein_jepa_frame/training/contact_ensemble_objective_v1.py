"""Contact training-estimand decomposition; existing evaluation is unchanged.

For conditionally iid ensemble members z_m, given the observed window and fixed
model, the plug-in squared mean error has expectation (E[z]-q)^2+Var(z)/M.
Subtracting unbiased sample variance / M estimates the population-mean squared
error. Members may have correlated contacts within a structure; independence is
required across members, and between their sampling randomness and target q.
Shared stochastic model randomness or interacting members can invalidate this
interpretation. Algebraic identities hold without those sampling assumptions.

The corrected finite-member estimate can be negative, including at probability
boundaries. It must not be clamped. This module proposes no evaluation change
and does not imply that empirical-ensemble Brier evaluation is erroneous.
"""
from typing import NamedTuple

import torch


class ContactEnsembleBrier(NamedTuple):
    plug_in_brier: torch.Tensor
    variance_penalty: torch.Tensor
    population_brier_u: torch.Tensor


def contact_ensemble_brier(predictions: torch.Tensor, target: torch.Tensor) -> ContactEnsembleBrier:
    """Return scalar pair-mean scores for probabilities [M,P] and target [P].

    Requires M>=2, P>=1, finite probabilities in [0,1], and matching floating
    dtype/device. All arithmetic preserves autograd, including target gradients;
    neither input is mutated or detached. Callers decide whether labels require
    gradients. Explicit finite/range validation intentionally synchronizes an
    accelerator, appropriate for this bounded diagnostic primitive.
    """
    if not isinstance(predictions, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError('predictions and target must be tensors')
    if not predictions.is_floating_point() or not target.is_floating_point():
        raise TypeError('predictions and target must have floating dtype')
    if predictions.ndim != 2 or target.ndim != 1:
        raise ValueError('expected predictions [M,P] and target [P]')
    members, pairs = predictions.shape
    if members < 2 or pairs < 1 or target.shape != (pairs,):
        raise ValueError('requires M>=2, P>=1 and matching target [P]')
    if predictions.dtype != target.dtype or predictions.device != target.device:
        raise ValueError('predictions and target must share dtype and device')
    for name, values in (('predictions', predictions), ('target', target)):
        if not bool(torch.isfinite(values).all()):
            raise ValueError(name+' must be finite')
        if not bool(((values >= 0) & (values <= 1)).all()):
            raise ValueError(name+' must lie in [0,1]')
    mean = predictions.mean(dim=0)
    plug_in = (mean-target).square().mean()
    penalty = predictions.var(dim=0, unbiased=True).mean()/members
    return ContactEnsembleBrier(plug_in, penalty, plug_in-penalty)
