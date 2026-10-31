"""BioEmu-style static/equilibrium ensemble metrics (T2A) for any sample source.

Reference states are clustered from the *true* MD ensemble in internal-distance
space; model samples are assigned with the same centroids (BioEmu's "shared
feature + shared state" rule). Implements the definitions in
EVALUATION_METHODS_AND_COMMANDS_20261021.md 2.6 via evaluation.states_metrics.

This module only measures; it never decodes or trains anything.
"""
from __future__ import annotations

import torch

from evaluation.states_metrics import (
    _kmeans,
    assign_to_centroids,
    free_energy_mae,
    internal_distance_features,
    reference_state_coverage,
    state_populations,
)


def static_ensemble_terms(model_ca, ref_ca, mask, temperature_k, *, k_states=8,
                          eps_angstrom=2.0, min_sep=3, max_pairs=512, seed=0):
    """Coverage / free-energy / population terms for one reference ensemble.

    Args:
        model_ca: [N, L, 3] generated conformations (same residue convention as ref).
        ref_ca:   [M, L, 3] true MD frames used as the reference ensemble.
        mask:     [L] bool residue mask, or None when the arrays are ALREADY
                  restricted to the evaluated residues (both callers in this
                  repository pre-restrict them, so a full-length mask here would
                  index a compacted array out of range).
        temperature_k: temperature of the reference ensemble (for F = -kT ln p).
    """
    model_ca = torch.as_tensor(model_ca, dtype=torch.float32)
    ref_ca = torch.as_tensor(ref_ca, dtype=torch.float32)
    if mask is None:
        mask_t = torch.ones(model_ca.shape[1], dtype=torch.bool)
    else:
        mask_t = torch.as_tensor(mask).bool()
        if mask_t.numel() != model_ca.shape[1]:
            raise ValueError(
                f'mask has {mask_t.numel()} residues but samples have '
                f'{model_ca.shape[1]}; pass mask=None for pre-restricted arrays')
    if not mask_t.any():
        raise ValueError('empty residue mask for static ensemble metrics')
    coverage = reference_state_coverage(model_ca, ref_ca, mask_t, k_states=k_states,
                                        eps_angstrom=eps_angstrom, min_sep=min_sep,
                                        max_pairs=max_pairs, seed=seed)
    # One clustering of the reference ensemble, reused for both populations so
    # the label space stays model-independent.
    ref_feat, _ = internal_distance_features(ref_ca, mask_t, min_sep, max_pairs, seed)
    model_feat, _ = internal_distance_features(model_ca, mask_t, min_sep, max_pairs, seed)
    centroids, ref_labels = _kmeans(ref_feat, k_states, seed=seed)
    model_labels = assign_to_centroids(model_feat, centroids)
    pop_ref = state_populations(ref_labels, centroids.shape[0])
    pop_model = state_populations(model_labels, centroids.shape[0])
    terms = free_energy_mae(pop_model, pop_ref, float(temperature_k))
    return dict(coverage=float(coverage['coverage']),
                coverage_threshold_curve=coverage['coverage_threshold_curve'],
                n_states=int(coverage['n_states']),
                n_model_samples=int(model_ca.shape[0]),
                n_ref_frames=int(ref_ca.shape[0]),
                **{k: float(v) for k, v in terms.items() if isinstance(v, (int, float))})


__all__ = ['static_ensemble_terms']