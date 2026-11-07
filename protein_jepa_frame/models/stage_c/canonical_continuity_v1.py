"""Source-faithful PCA-feature continuity diagnostics, with no model execution.

The source frame is computed in float32/B1, including its arbitrary raw eigenvector
signs and third-moment sign convention. Analysis promotes those recorded arrays,
not a newly computed float64 eigensystem. A proper, translation-optimal physical
Kabsch fit is computed separately from native CA coordinates in float64. Its
rotation also transports the recorded float32-centered arrays; their shape term
therefore retains float32 centroid roundoff.

Signed PCA frames live in O(3), not necessarily SO(3). All 48 signed permutations
are retained. Their best match is a descriptive relabeling, not a replacement
feature or a diagnosis of causality. Equal eigenvalues remain degenerate after
isotropic jitter. Rank and cost-tie tolerances are diagnostics, never exclusions.
"""
from __future__ import annotations

from itertools import permutations, product
from typing import Any

import torch

COST_TIE_ATOL = 1.0e-12
RANK_ATOL = 1.0e-12
RANK_RTOL = 1.0e-10
CENTERED_H_EPS = 1.0e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mask(mask: torch.Tensor, length: int) -> None:
    _require(isinstance(mask, torch.Tensor) and mask.device.type == "cpu"
             and mask.dtype == torch.bool and mask.shape == (length,),
             "mask must be a CPU bool [L] tensor")
    _require(bool(mask.any()), "fixed mask must contain at least one residue")


@torch.no_grad()
def canonical_frame_audit(coords: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reproduce frozen encoder canonicalization on sanitized float32 B1 input.

    Masked coordinates are set to zero before the source operations; all valid
    backbone atoms must be finite. No input is modified. Standardized skew uses
    the recorded source third moment and a float64 projected second moment;
    zero projected variance is reported undefined rather than thresholded out.
    """
    _require(isinstance(coords, torch.Tensor) and coords.device.type == "cpu"
             and coords.dtype == torch.float32 and coords.ndim == 3
             and coords.shape[1:] == (4, 3), "coords must be CPU float32 [L,4,3]")
    _mask(mask, coords.shape[0])
    _require(bool(torch.isfinite(coords[mask]).all()), "nonfinite coordinates on fixed support")
    safe = torch.where(mask[:, None, None], coords, torch.zeros_like(coords))[None]
    ca = safe[:, :, 1, :]
    m = mask[None].to(ca.dtype).unsqueeze(-1)
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    centroid = (ca * m).sum(dim=1, keepdim=True) / denom
    centered = (ca - centroid) * m
    cov = torch.einsum("bli,blj->bij", centered, centered)
    cov = cov + 1.0e-6 * torch.eye(3, dtype=coords.dtype).unsqueeze(0)
    evals, evecs = torch.linalg.eigh(cov)
    raw_basis = evecs.flip(-1)
    proj = torch.einsum("blj,bjk->blk", centered, raw_basis)
    n = m.squeeze(-1).sum(dim=1, keepdim=True).clamp_min(1.0)
    skew = proj.pow(3).sum(dim=1) / n
    sign = torch.where(skew >= 0, torch.ones_like(skew), -torch.ones_like(skew))
    basis = raw_basis * sign.unsqueeze(1)
    local = proj * sign.unsqueeze(1) * m
    variance = proj[0, mask].double().square().mean(0)
    defined = variance > 0
    standardized = torch.full((3,), float("nan"), dtype=torch.float64)
    standardized[defined] = skew[0].double().abs()[defined] / variance[defined].pow(1.5)
    descending = evals[0].flip(-1)
    gaps = (descending[:-1].double() - descending[1:].double()) / descending[0].double()
    return {
        "ca": ca[0].clone(), "centroid": centroid[0, 0].clone(),
        "centered": centered[0], "canonical": local[0],
        "raw_basis": raw_basis[0], "basis": basis[0],
        "eigvals_desc": descending, "skew_before_sign": skew[0], "sign": sign[0],
        "abs_standardized_skew": standardized, "skew_defined": defined,
        "normalized_eigengaps": gaps, "mask": mask.clone(),
        "basis_determinant": torch.linalg.det(basis[0].double()),
    }


def signed_permutations() -> torch.Tensor:
    """48 matrices in permutation lex order, then (-1,+1) sign lex order.

    Column j is sign[j] times coordinate vector permutation[j]. The first exact
    minimum wins; the separate near-tie diagnostic does not affect selection.
    """
    result = []
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = torch.zeros((3, 3), dtype=torch.float64)
            for j in range(3):
                matrix[order[j], j] = signs[j]
            result.append(matrix)
    return torch.stack(result)


def _mean(values: torch.Tensor, support: torch.Tensor | None = None) -> float | None:
    selected = values if support is None else values[support]
    return float(selected.mean()) if selected.numel() else None


def _squared_rows(values: torch.Tensor) -> torch.Tensor:
    return values.square().sum(-1)


def _rms(values: torch.Tensor) -> float | None:
    return float(_squared_rows(values).mean().sqrt()) if values.shape[0] else None


@torch.no_grad()
def transition_audit(
    frame0: dict[str, torch.Tensor], frame1: dict[str, torch.Tensor],
    H0: torch.Tensor, H1: torch.Tensor, mask: torch.Tensor,
) -> dict[str, Any]:
    """Compare all fixed-support rows without a jump/event selection threshold.

    H tangent/radial directions require centered H0 norm > 1e-12. Undefined
    directions are NaN with an explicit mask; raw/centered increments retain
    every observed residue. Mean squared quantities are per-residue means;
    RMS quantities take the square root after averaging squared vector norms.
    """
    _require(isinstance(H0, torch.Tensor) and H0.ndim == 2 and H0.shape[1] > 0,
             "H must have shape [L,d] with d positive")
    _mask(mask, H0.shape[0])
    for H in (H0, H1):
        _require(isinstance(H, torch.Tensor) and H.shape == H0.shape
                 and H.device.type == "cpu" and H.dtype == torch.float32,
                 "H must be CPU float32 tensors with identical [L,d] shape")
        _require(bool(torch.isfinite(H[mask]).all()), "nonfinite H on fixed support")
    for frame in (frame0, frame1):
        _require(torch.equal(frame["mask"], mask), "frame mask differs from fixed mask")
        for key, shape in (("ca", (len(mask), 3)), ("centered", (len(mask), 3)),
                           ("canonical", (len(mask), 3)), ("basis", (3, 3))):
            value = frame[key]
            _require(value.shape == shape and value.dtype == torch.float32
                     and value.device.type == "cpu" and bool(torch.isfinite(value).all()),
                     f"invalid source frame {key}")
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    x, y = frame0["ca"][mask].double(), frame1["ca"][mask].double()
    cx, cy = x.mean(0), y.mean(0)
    xc, yc = x - cx, y - cy
    U, singular, Vh = torch.linalg.svd(yc.T @ xc)
    correction = torch.eye(3, dtype=torch.float64)
    reflection_correction = bool(torch.linalg.det(U @ Vh) < 0)
    correction[-1, -1] = -1.0 if reflection_correction else 1.0
    R = U @ correction @ Vh
    translation = cx - cy @ R
    physical_residual = y @ R + translation - x
    xs, ys = frame0["centered"][mask].double(), frame1["centered"][mask].double()
    B0, B1 = frame0["basis"].double(), frame1["basis"].double()
    Z0, Z1 = xs @ B0, ys @ B1
    jump = Z1 - Z0
    source_jump = frame1["canonical"][mask].double() - frame0["canonical"][mask].double()
    basis_term = Z1 - ys @ R @ B0
    shape_term = (ys @ R - xs) @ B0
    overlap = B0.T @ R.T @ B1
    candidates = signed_permutations()
    costs = (B0[None] - R.T[None] @ B1[None] @ candidates).square().sum((-2, -1))
    selected = int(torch.argmin(costs))
    P = candidates[selected]
    sorted_costs = torch.sort(costs).values
    matched_jump = Z1 @ P - Z0
    local_pairs = torch.triu_indices(len(idx), len(idx), offset=1).T
    pairs = idx[local_pairs]
    pair_change = ((y[local_pairs[:, 0]] - y[local_pairs[:, 1]]).norm(dim=-1)
                   - (x[local_pairs[:, 0]] - x[local_pairs[:, 1]]).norm(dim=-1))
    # Use a transported prior basis, not raw eigensolver skew signs, to compare
    # third moments. Double-centering here describes physical cloud shape.
    prior_moments0 = (xc @ B0).pow(3).mean(0)
    prior_moments1 = (yc @ R @ B0).pow(3).mean(0)
    h0, h1 = H0[mask].double(), H1[mask].double()
    dh = h1 - h0
    z0, z1 = h0 - h0.mean(-1, keepdim=True), h1 - h1.mean(-1, keepdim=True)
    dz = z1 - z0
    radius = z0.norm(dim=-1)
    tangent_valid = radius > CENTERED_H_EPS
    radial = torch.full_like(radius, float("nan"))
    tangent2 = torch.full_like(radius, float("nan"))
    u = z0[tangent_valid] / radius[tangent_valid, None]
    radial[tangent_valid] = (u * dz[tangent_valid]).sum(-1)
    tangent2[tangent_valid] = _squared_rows(dz[tangent_valid] - radial[tangent_valid, None] * u)
    raw2, centered2 = _squared_rows(dh), _squared_rows(dz)
    channel2 = dh.mean(-1).square() * dh.shape[-1]
    radial2 = radial.square()
    raw_norm_change = _squared_rows(h1) - _squared_rows(h0)
    centered_norm_change = _squared_rows(z1) - _squared_rows(z0)
    finite_chord_closure = centered_norm_change - (2 * radius * radial + radial2 + tangent2)
    centroid_residual0, centroid_residual1 = xs.mean(0), ys.mean(0)
    reconstruction0 = Z0 - frame0["canonical"][mask].double()
    reconstruction1 = Z1 - frame1["canonical"][mask].double()
    arrays = {
        "valid_indices": idx, "R": R, "translation": translation,
        "native_centroid0": cx, "native_centroid1": cy,
        "kabsch_singular_values": singular, "physical_residual": physical_residual,
        "source_centering_residual0": centroid_residual0, "source_centering_residual1": centroid_residual1,
        "analysis_Z0": Z0, "analysis_Z1": Z1, "source_jump": source_jump,
        "analysis_jump": jump, "basis_term": basis_term, "shape_term": shape_term,
        "full_overlap": overlap, "candidate_permutations": candidates,
        "candidate_costs": costs, "selected_permutation": P, "matched_jump": matched_jump,
        "pair_indices": pairs, "pair_distance_change": pair_change,
        "transported_prior_basis_third_moment0": prior_moments0,
        "transported_prior_basis_third_moment1": prior_moments1,
        "source_reconstruction_error0": reconstruction0, "source_reconstruction_error1": reconstruction1,
        "H_increment": dh, "H_centered_increment": dz,
        "H_raw_increment_squared": raw2, "H_centered_increment_squared": centered2,
        "H_channel_increment_squared": channel2, "H_radial_increment": radial,
        "H_radial_increment_squared": radial2, "H_tangent_increment_squared": tangent2,
        "H_raw_norm_squared_change": raw_norm_change,
        "H_centered_norm_squared_change": centered_norm_change,
        "H_centered_initial_norm": radius, "H_finite_chord_closure": finite_chord_closure,
        "tangent_valid": tangent_valid,
    }
    rank_threshold = RANK_ATOL + RANK_RTOL * float(singular.max())
    rank = int((singular > rank_threshold).sum())
    basis2, shape2 = _squared_rows(basis_term), _squared_rows(shape_term)
    cross = 2 * (basis_term * shape_term).sum(-1)
    summary = {
        "residue_count": len(idx), "pair_count": len(pairs),
        "kabsch_rmsd": _rms(physical_residual), "kabsch_determinant": float(torch.linalg.det(R)),
        "kabsch_rank": rank, "kabsch_rank_threshold": rank_threshold,
        "kabsch_rank_below_two": rank < 2,
        "kabsch_reflection_correction": reflection_correction,
        "kabsch_reflected_smallest_singular_near_tie": reflection_correction and bool(
            (singular[-2] - singular[-1]).abs() <= rank_threshold),
        "basis0_determinant": float(torch.linalg.det(B0)),
        "basis1_determinant": float(torch.linalg.det(B1)),
        "overlap_determinant": float(torch.linalg.det(overlap)),
        "original_basis_cost": float((B0 - R.T @ B1).square().sum()),
        "selected_candidate_index": selected, "selected_candidate_cost": float(costs[selected]),
        "candidate_cost_margin": float(sorted_costs[1] - sorted_costs[0]),
        "candidate_exact_tie_count": int((costs == costs[selected]).sum()),
        "candidate_near_tie_count": int((costs - costs[selected] <= COST_TIE_ATOL).sum()),
        "candidate_near_tie_atol": COST_TIE_ATOL,
        "analysis_jump_rms": _rms(jump), "source_jump_rms": _rms(source_jump),
        "matched_jump_rms": _rms(matched_jump),
        "basis_term_mean_squared": _mean(basis2), "shape_term_mean_squared": _mean(shape2),
        "basis_shape_cross_mean": _mean(cross), "analysis_jump_mean_squared": _mean(_squared_rows(jump)),
        "decomposition_max_abs_error": float((jump - basis_term - shape_term).abs().max()),
        "squared_decomposition_max_abs_error": float((_squared_rows(jump) - basis2 - shape2 - cross).abs().max()),
        "source_reconstruction0_max_abs_error": float(reconstruction0.abs().max()),
        "source_reconstruction1_max_abs_error": float(reconstruction1.abs().max()),
        "source_vs_analysis_jump_max_abs_error": float((source_jump - jump).abs().max()),
        "source_centering0_norm": float(centroid_residual0.norm()),
        "source_centering1_norm": float(centroid_residual1.norm()),
        "all_pair_distance_rms_change": float(pair_change.square().mean().sqrt()) if len(pairs) else None,
        "H_raw_increment_mean_squared": _mean(raw2),
        "H_centered_increment_mean_squared": _mean(centered2),
        "H_channel_increment_mean_squared": _mean(channel2),
        "H_radial_increment_mean": _mean(radial, tangent_valid),
        "H_radial_increment_mean_squared": _mean(radial2, tangent_valid),
        "H_tangent_increment_mean_squared": _mean(tangent2, tangent_valid),
        "H_raw_norm_squared_change_mean": _mean(raw_norm_change),
        "H_centered_norm_squared_change_mean": _mean(centered_norm_change),
        "H_tangent_valid_count": int(tangent_valid.sum()),
        "H_tangent_undefined_count": int((~tangent_valid).sum()),
        "H_finite_chord_closure_max_abs_error": float(finite_chord_closure[tangent_valid].abs().max()) if bool(tangent_valid.any()) else None,
        "H_centered_increment_closure_max_abs_error": float((raw2 - centered2 - channel2).abs().max()),
    }
    return {"arrays": arrays, "summary": summary}
