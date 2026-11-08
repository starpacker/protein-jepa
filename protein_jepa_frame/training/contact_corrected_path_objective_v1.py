"""Matched marginal-path objectives differing only in contact mean estimation.

The plugin control preserves the frozen ordered objective's loss arithmetic and
graph. The population_u arm subtracts 20 times the endpoint member-variance/M
term. This changes a training estimand; existing evaluation is unchanged.
The correction estimates squared error of conditional mean soft contacts under
conditionally iid member sampling. It does not identify their full law, and
finite corrected losses may be negative. No clipping or new sampling occurs.
"""
from __future__ import annotations

import torch

from protein_jepa_frame.training.contact_ensemble_objective_v1 import contact_ensemble_brier
from protein_jepa_frame.training.ordered_path_objective_v1 import (
    _CachedSampleView, _require, _validate_batch, coordinate_path_scores, sample_paths,
)
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics, _align_to


VERSION = 'contact_corrected_marginal_path_objective_v1'
CONTACT_ESTIMATORS = ('plugin', 'population_u')


def endpoint_contact_variance(endpoint_ca, residue_mask, residue_index, chain_id):
    """Batch-mean sample variance/M on the exact endpoint trainer pair mask.

    No target is accepted. The primitive's zero placeholder target is used only
    to obtain its target-independent variance term; its other scores are ignored.
    Residue gaps/cross-chain pairs follow the frozen trainer's index convention.
    """
    _require(endpoint_ca.ndim == 4 and endpoint_ca.shape[1] == 12
             and endpoint_ca.shape[-1] == 3, 'requires endpoint CA [B,12,L,3]')
    b, _, l, _ = endpoint_ca.shape
    _require(b > 0 and residue_mask.shape == residue_index.shape == chain_id.shape == (b,l)
             and residue_mask.dtype == torch.bool, 'invalid endpoint residue metadata')
    total = endpoint_ca.new_zeros(())
    for i in range(b):
        mask = residue_mask[i]
        distances = torch.cdist(endpoint_ca[i:i+1], endpoint_ca[i:i+1])[0]
        separation = (residue_index[i,:,None]-residue_index[i,None,:]).abs()
        same = chain_id[i,:,None] == chain_id[i,None,:]
        pairs = mask[:,None] & mask[None,:] & ((separation >= 3) | ~same)
        pairs = pairs & torch.triu(torch.ones_like(pairs), diagonal=1)
        _require(pairs.any(), 'no valid nonlocal residue pairs')
        probabilities = torch.sigmoid((8-distances)/.5)[:,pairs]
        placeholder = probabilities.new_zeros(probabilities.shape[1])
        total = total + contact_ensemble_brier(probabilities, placeholder).variance_penalty
    return total / b


def contact_corrected_path_losses(model, batch, horizon, args, generator):
    """Frozen marginal-path objective plus an explicit contact-estimator arm.

    Requires args.contact_estimator in {plugin,population_u}, n_samples=12,
    coord_objective=marginal_path and the frozen .5 path/20 contact weights.
    All returned terms are scalar tensors. ``contact`` remains the original
    unweighted plug-in diagnostic, aliased by ``contact_plugin``; added
    ``contact_variance`` and ``contact_population_u`` are also unweighted.
    ``loss_plugin`` preserves the original full objective (including latent ES),
    while ``loss`` is the selected training objective. Arm metadata belongs in
    the caller's configuration, not this scalar loss dictionary.
    """
    _require(getattr(args, 'contact_estimator', None) in CONTACT_ESTIMATORS,
             'contact_estimator must be plugin or population_u')
    _require(args.coord_objective == 'marginal_path', 'v1 fixes marginal_path')
    _require(type(args.n_samples) is int and args.n_samples == 12, 'v1 fixes 12 members')
    safe, mask = _validate_batch(model, batch, horizon, args)
    # This orchestration and its accumulation order reproduce the frozen
    # ordered_path_losses control; no legacy module or model method is patched.
    paths = sample_paths(model,safe['coords'][:,:4],mask,*[safe[k] for k in ('aatype','residue_index','chain_id')],
        safe['temperature_k'],horizon,args.n_samples,generator,args.grad_steps,
        observed_time_ps=safe['time_ps'][:,:4],checkpoint_decoder=getattr(args,'checkpoint_decoder',False))
    cached = _CachedSampleView(model,paths)
    rng_before = None if generator is None else generator.get_state().clone()
    terms = CADynamics.losses(cached,safe,horizon,args,generator)
    _require(cached.calls == 1, 'endpoint losses did not consume exactly one cached sample')
    if generator is not None:
        _require(torch.equal(generator.get_state(),rng_before), 'cached endpoint scoring consumed sampling RNG')
    with torch.no_grad():
        aligned = torch.stack([_align_to(safe['coords'][i,3:,:,1],safe['coords'][i,3,:,1],mask[i])
                               for i in range(len(mask))])
    scores = coordinate_path_scores(paths['ca_paths'],aligned,mask)
    _require(torch.equal(scores['endpoint'],terms['coord_es']), 'path endpoint scoring differs from frozen endpoint objective')
    if horizon == 1:
        joint = marginal = objective = terms['coord_es']
    else:
        joint,marginal = scores['joint'],scores['marginal']
        selected = marginal
        objective = (1-args.path_weight)*terms['coord_es'] + args.path_weight*selected
        terms['loss'] = terms['loss'] + args.path_weight*(selected-terms['coord_es'])
    terms.update(coordinate_path_es=joint,coordinate_marginal_es=marginal,coordinate_objective=objective)
    variance = endpoint_contact_variance(paths['endpoint_ca'],mask,safe['residue_index'],safe['chain_id'])
    terms.update(contact_plugin=terms['contact'], contact_variance=variance,
                 contact_population_u=terms['contact']-variance, loss_plugin=terms['loss'])
    if args.contact_estimator == 'population_u':
        terms['loss'] = terms['loss_plugin'] - args.contact_weight*variance
    # The control preserves the exact old loss tensor, with no subtract-zero op.
    _require(all(value.ndim == 0 and torch.isfinite(value).all() for value in terms.values()),
             'nonfinite or nonscalar corrected path loss')
    return terms
