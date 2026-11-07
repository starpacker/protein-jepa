"""Matched absolute-distance and signed-increment additions to path training.

The frozen marginal coordinate objective, latent endpoint score, contact and
clash terms remain intact. Each arm adds one Angstrom-valued, time-mean physical
distance U-energy score with fixed weight one, using the same single rollout.
Eligibility comes from full-PSF graph distance >2 before crop/mask, supplied
separately from untouched legacy encoder IDs. This is a versioned pair-support
change from v1, not merely relaxing duplicate-ID validation. It changes no
evaluation definition or sampler.
"""
from __future__ import annotations

import torch

from protein_jepa_frame.training.ordered_path_objective_v1 import (
    _CachedSampleView, _require, _validate_batch, coordinate_path_scores, sample_paths,
)
from protein_jepa_frame.training.physical_distance_path_objective_v2 import (
    physical_distance_path_scores, validate_physical_pair_support,
)
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics, _align_to


VERSION = 'physical_distance_augmented_marginal_path_objective_v2'
DISTANCE_OBJECTIVES = ('absolute', 'increment')


def physical_distance_augmented_path_losses(model, batch, horizon, args, generator):
    """Return scalar base, auxiliary and augmented losses without extra draws.

    ``loss_base`` reproduces the frozen marginal objective exactly. Both
    ``loss_absolute`` and ``loss_increment`` are evaluated on the same paths;
    ``loss`` aliases the predeclared arm. The auxiliary uses all nonlocal
    physical pairs and an equal-time mean, never target-dependent scales.
    Existing ``distance_es`` remains the old zero-weight endpoint diagnostic.

    H1 auxiliary values and parameter gradients agree up to arithmetic rounding
    because the observed anchor is fixed. This identity does not extend to an
    independently trainable coordinate at time zero.
    """
    _require(getattr(args, 'distance_path_objective', None) in DISTANCE_OBJECTIVES,
             'distance_path_objective must be absolute or increment')
    _require(getattr(args, 'distance_path_weight', None) == 1., 'v2 fixes distance_path_weight=1')
    _require(getattr(args, 'contact_estimator', 'plugin') == 'plugin', 'v2 retains plugin contact loss')
    _require(args.coord_objective == 'marginal_path', 'v2 fixes marginal_path')
    _require(type(args.n_samples) is int and args.n_samples == 12, 'v2 fixes 12 members')
    _require(model.kernel.cfg.step_ps == 1000., 'v2 requires 1000 ps kernel steps')
    safe, mask = _validate_batch(model, batch, horizon, args)
    _require('geometry_row_ids' in safe and 'physical_pair_eligibility' in safe,
             'explicit physical pair support metadata is required before sampling')
    validate_physical_pair_support(mask,safe['geometry_row_ids'],safe['physical_pair_eligibility'])
    # Preserve the frozen ordered objective's graph and accumulation order.
    # Future coordinates are consumed only after this observation-only draw.
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
        objective = (1-args.path_weight)*terms['coord_es'] + args.path_weight*marginal
        terms['loss'] = terms['loss'] + args.path_weight*(marginal-terms['coord_es'])
    terms.update(coordinate_path_es=joint,coordinate_marginal_es=marginal,coordinate_objective=objective)
    # Distances use original physical coordinates, requiring no target alignment.
    # In particular their observed time zero remains the exact common anchor.
    auxiliary = physical_distance_path_scores(
        paths['ca_paths'],safe['coords'][:,3:,:,1],safe['coords'][:,3,:,1],mask,
        geometry_row_ids=safe['geometry_row_ids'],physical_pair_eligibility=safe['physical_pair_eligibility'],
        predicted_times_ps=safe['time_ps'][:,3:],reference_times_ps=safe['time_ps'][:,3:],
        reference_mask=safe['residue_mask'][:,3:])
    terms.update(loss_base=terms['loss'],distance_absolute_es=auxiliary['loss_absolute'],
                 distance_increment_es=auxiliary['loss_increment'])
    terms['loss_absolute'] = terms['loss_base'] + args.distance_path_weight*terms['distance_absolute_es']
    terms['loss_increment'] = terms['loss_base'] + args.distance_path_weight*terms['distance_increment_es']
    terms['loss'] = terms['loss_'+args.distance_path_objective]
    if generator is not None:
        _require(torch.equal(generator.get_state(),rng_before), 'auxiliary scoring consumed sampling RNG')
    _require(all(value.ndim == 0 and torch.isfinite(value).all() for value in terms.values()),
             'nonfinite or nonscalar augmented path loss')
    return terms
