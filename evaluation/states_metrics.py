"""Metastable-state ensemble metrics (TASK 2A, NEW 2026-10-21).

Implements the *static/ensemble* half of the BioEmu-style benchmark described in
``BioEmu_指标梳理_蛋白质动力学基准.md`` §2, i.e. what BioEmu actually reports
(NOT next-state prediction):

    * reference-state coverage (fraction of reference states reached),
    * metastable-state free-energy MAE  F_k = -k_B T ln p_k,
    * state-population MAE / JSD / TV.

State definition
----------------
Reference states are defined by clustering the **true** MD ensemble of a protein
in an internal-coordinate feature space (pairwise Cα distances, maskable and
rotation invariant), so the assignment is a property of the *reference* only.
Model samples are assigned with the SAME centroids (nearest-centroid), so no
model-specific freedom enters the state definition.  This mirrors the "shared
feature + shared state model, project both onto it" fairness rule (§2.2).

All distances are in Å; free energies use k_B in kcal/(mol·K) so F is in
kcal/mol (comparable to BioEmu's ~0.7–0.9 kcal/mol).

Units: T is Kelvin; k_B = 0.0019872041 kcal/(mol·K) = R/1000.
"""
from __future__ import annotations

from typing import Optional

import torch

KB_KCAL = 1.9872041e-3  # kcal/(mol*K)


def internal_distance_features(ca: torch.Tensor, mask: torch.Tensor | None = None,
                               min_sep: int = 3,
                               max_pairs: int = 512,
                               seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotation-invariant features: masked pairwise Cα distances.

    Args:
        ca: [N, L, 3] (N frames, single protein).
        mask: [L] bool residue mask.

    Returns:
        feats: [N, P] pairwise distances (P <= max_pairs).
        pair_idx: [P, 2] the selected residue pairs.
    """
    N, L, _ = ca.shape
    dev = ca.device
    i, j = torch.triu_indices(L, L, offset=max(min_sep, 1), device=dev)
    if mask is not None:
        m = mask.bool()
        keep = m[i] & m[j]
        i, j = i[keep], j[keep]
    if i.numel() > max_pairs:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(i.numel(), generator=g)[:max_pairs]
        sel = sel.sort().values
        i, j = i[sel], j[sel]
    pair_idx = torch.stack([i, j], dim=-1)
    d = (ca[:, i, :] - ca[:, j, :]).norm(dim=-1)   # [N, P]
    return d, pair_idx


def _kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 0):
    """Minimal deterministic KMeans (LLoyd).  x: [N, D] -> centroids [k, D], labels [N]."""
    g = torch.Generator().manual_seed(seed)
    N = x.shape[0]
    k = min(k, N)
    # k-means++ style init: pick first at random, rest farthest-point.
    first = torch.randint(0, N, (1,), generator=g).item()
    cents = [x[first]]
    d2 = ((x - cents[0]) ** 2).sum(-1)
    for _ in range(1, k):
        idx = torch.argmax(d2).item()
        cents.append(x[idx])
        d2 = torch.minimum(d2, ((x - cents[-1]) ** 2).sum(-1))
    C = torch.stack(cents)                                   # [k, D]
    labels = torch.zeros(N, dtype=torch.long, device=x.device)
    for _ in range(iters):
        dist = torch.cdist(x, C)                             # [N, k]
        new_labels = dist.argmin(dim=-1)
        if torch.equal(new_labels, labels) and _:
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            sel = labels == c
            if sel.any():
                C[c] = x[sel].mean(dim=0)
    return C, labels


def assign_to_centroids(x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid assignment [N, D] -> labels [N] given centroids [k, D]."""
    return torch.cdist(x, C).argmin(dim=-1)


def state_populations(labels: torch.Tensor, k: int) -> torch.Tensor:
    """Empirical populations [k] (fractions) from integer labels."""
    counts = torch.zeros(k, dtype=torch.float64, device=labels.device)
    counts.index_add_(0, labels.long(), torch.ones_like(labels, dtype=torch.float64))
    return counts / counts.sum().clamp_min(1.0)


def free_energy(pop: torch.Tensor, temperature_k: float) -> torch.Tensor:
    """F_k = -k_B T ln p_k, relative to the most populated state (kcal/mol).

    States with p_k <= 0 are returned as +inf (unreachable).
    """
    p = pop.clamp_min(0.0).double()
    with torch.no_grad():
        F = torch.where(p > 0, -KB_KCAL * temperature_k * torch.log(p.clamp_min(1e-300)),
                        torch.full_like(p, float("inf")))
    finite = torch.isfinite(F)
    if finite.any():
        F = F - F[finite].min()
    return F


def free_energy_mae(pop_model: torch.Tensor, pop_ref: torch.Tensor,
                    temperature_k: float) -> dict:
    """State free-energy MAE (kcal/mol) + population MAE / JSD / TV."""
    Fm = free_energy(pop_model, temperature_k)
    Fr = free_energy(pop_ref, temperature_k)
    both = torch.isfinite(Fm) & torch.isfinite(Fr)
    # States present in ref but absent in model get the +inf floor: report them as
    # a large-but-finite penalty equal to the ref's dynamic range.
    rng = (Fr[torch.isfinite(Fr)].max() - Fr[torch.isfinite(Fr)].min()) if torch.isfinite(Fr).any() else torch.tensor(1.0)
    Fr_f = torch.where(torch.isfinite(Fr), Fr, torch.full_like(Fr, float(rng)))
    Fm_f = torch.where(torch.isfinite(Fm), Fm, torch.full_like(Fm, float(rng)))
    mae = float((Fm_f - Fr_f).abs().mean())
    pm, pr = pop_model.double().clamp_min(0), pop_ref.double().clamp_min(0)
    pop_mae = float((pm - pr).abs().mean())
    tv = float(0.5 * (pm - pr).abs().sum())
    m = 0.5 * (pm + pr)
    def kl(a, b):
        good = (a > 0) & (b > 0)
        return float((a[good] * torch.log(a[good] / b[good])).sum()) if good.any() else 0.0
    jsd = 0.5 * kl(pm, m) + 0.5 * kl(pr, m)
    return {"free_energy_mae_kcal": mae, "population_mae": pop_mae,
            "population_tv": tv, "population_jsd": jsd,
            "n_states_in_model": int((pm > 0).sum()), "n_states_in_ref": int((pr > 0).sum())}


def reference_state_coverage(model_ca: torch.Tensor, ref_ca: torch.Tensor,
                             mask: torch.Tensor | None = None,
                             *, k_states: int = 8, eps_angstrom: float = 2.0,
                             min_frac: float = 0.001, align: bool = True,
                             kabsch_ref: torch.Tensor | None = None,
                             min_sep: int = 3, max_pairs: int = 512,
                             seed: int = 0) -> dict:
    """Reference-state coverage (BioEmu §2.1).

    Reference states = KMeans clusters of the true ensemble in internal-distance
    space.  A state is *covered* if at least ``min_frac`` of model samples fall
    within ``eps_angstrom`` (RMS in feature space) of its centroid.

    Args:
        model_ca: [N, L, 3] model samples.
        ref_ca:   [M, L, 3] true/reference frames.
        mask:     [L] residue mask.

    Returns dict with coverage, per-state hits, hits/sample, and the coverage
    at a small threshold curve.
    """
    ref_feat, pair_idx = internal_distance_features(ref_ca, mask, min_sep, max_pairs, seed)
    model_feat, _ = internal_distance_features(model_ca, mask, min_sep, max_pairs, seed)
    C, ref_labels = _kmeans(ref_feat, k_states, seed=seed)
    ref_pop = state_populations(ref_labels, C.shape[0])
    # model distance to each ref centroid, RMS over features
    d = torch.cdist(model_feat, C) / (C.shape[1] ** 0.5)     # [N_model, k]
    nearest = d.argmin(dim=-1)
    covered, hits = [], []
    threshold = eps_angstrom
    for s in range(C.shape[0]):
        frac = float((d[:, s] < threshold).double().mean())
        covered.append(1.0 if frac >= min_frac else 0.0)
        hits.append(frac)
    coverage = float(sum(covered) / len(covered))
    # threshold curve
    curve = {}
    for eps in (1.0, 1.5, 2.0, 3.0, 4.0):
        cov = sum(1.0 if float((d[:, s] < eps).double().mean()) >= min_frac else 0.0
                  for s in range(C.shape[0])) / C.shape[0]
        curve[str(eps)] = cov
    return {"coverage": coverage, "n_states": int(C.shape[0]),
            "per_state_hit_frac": hits, "ref_populations": ref_pop.tolist(),
            "coverage_threshold_curve": curve,
            "n_model_samples": int(model_ca.shape[0]),
            "n_ref_frames": int(ref_ca.shape[0])}


__all__ = [
    "internal_distance_features", "state_populations", "free_energy",
    "free_energy_mae", "reference_state_coverage", "assign_to_centroids",
]
