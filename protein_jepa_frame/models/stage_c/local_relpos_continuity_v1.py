"""Raw local-centroid feature audit on fixed support; no model or PCA/Kabsch fit.

Local frames are the frozen encoder's float32 B1 row frames. The new feature is
F_i @ (CA_i-visible_centroid), computed in float32. Analysis promotes recorded
float32 centered coordinates and frames. Finite degenerate rows remain present;
normalization epsilon diagnostics do not certify chemical validity or stability.
PCA inputs and the proper row-Kabsch rotation are supplied by the caller.
"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as functional

from protein_jepa_frame.models.encoder import _local_backbone_frames

EPS = 1.0e-8
ROTATION_ATOL = 1.0e-10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _tensor(value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, label: str) -> None:
    _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
             and value.dtype == dtype and value.shape == shape, f'invalid {label} shape/dtype/device')


def _mask(mask: torch.Tensor, length: int) -> None:
    _tensor(mask, (length,), torch.bool, 'mask')
    _require(bool(mask.any()), 'empty fixed support')


def _sq(value: torch.Tensor) -> torch.Tensor:
    return value.square().sum(-1)


def _stats(value: torch.Tensor) -> dict[str, float | None]:
    if not value.numel():
        return dict(mean=None, min=None, max=None)
    return dict(mean=float(value.mean()), min=float(value.min()), max=float(value.max()))


@torch.no_grad()
def frame_audit(
    coords: torch.Tensor, mask: torch.Tensor, centered: torch.Tensor,
    pca_canonical: torch.Tensor, pca_basis: torch.Tensor,
) -> dict[str, Any]:
    """Extract source local frames once; require byte-exact source centering.

    Padding is sanitized before the frozen frame function, exactly as in the
    parent geometry audit. Its algorithm uses eps=1e-8 for bond norms (Angstrom)
    and normalized cross/y norms (dimensionless). At/below-epsilon flags only
    describe that computation. Geometric bond-angle sine uses promoted source
    bond vectors, with NaN plus a mask if either bond is exactly zero.
    """
    _require(isinstance(coords, torch.Tensor) and coords.ndim == 3, 'coords must be [L,4,3]')
    L = len(coords)
    _tensor(coords, (L, 4, 3), torch.float32, 'coords')
    _mask(mask, L)
    for value, shape, label in ((centered, (L, 3), 'centered'),
                                (pca_canonical, (L, 3), 'pca canonical'),
                                (pca_basis, (3, 3), 'pca basis')):
        _tensor(value, shape, torch.float32, label)
        _require(bool(torch.isfinite(value).all()), f'nonfinite {label}')
    _require(bool(torch.isfinite(coords[mask]).all()), 'nonfinite coordinates on fixed support')
    safe = torch.where(mask[:, None, None], coords, torch.zeros_like(coords))[None]
    ca = safe[:, :, 1, :]
    m = mask[None].to(torch.float32).unsqueeze(-1)
    centroid = (ca*m).sum(dim=1, keepdim=True)/m.sum(dim=1, keepdim=True).clamp_min(1.)
    source_centered = (ca-centroid)*m
    _require(_bits(centered, source_centered[0]), 'supplied centered differs from source float32 centering')
    _require(bool((pca_canonical[~mask] == 0).all()), 'PCA padding must remain zero')
    frames = _local_backbone_frames(safe)[0]
    local = torch.einsum('blkc,blc->blk', frames[None], source_centered)[0]
    zero = torch.zeros_like(local)
    X, F = centered.double(), frames.double()
    analysis_local = torch.einsum('lkc,lc->lk', F, X)
    analysis_pca = X @ pca_basis.double()
    c = safe[0, :, 2]-safe[0, :, 1]
    n = safe[0, :, 0]-safe[0, :, 1]
    c_norm, n_norm = c.norm(dim=-1), n.norm(dim=-1)
    x_axis = functional.normalize(c, dim=-1, eps=EPS)
    n_axis = functional.normalize(n, dim=-1, eps=EPS)
    cross = torch.cross(x_axis, n_axis, dim=-1)
    cross_norm = cross.norm(dim=-1)
    z_axis = functional.normalize(cross, dim=-1, eps=EPS)
    y_pre_norm = torch.cross(z_axis, x_axis, dim=-1).norm(dim=-1)
    _require(all(bool(torch.isfinite(value).all()) for value in
                 (frames, local, c_norm, n_norm, cross_norm, y_pre_norm)),
             'nonfinite derived source geometry')
    c64, n64 = c.double(), n.double()
    nc64, nn64 = c64.norm(dim=-1), n64.norm(dim=-1)
    raw_cross_norm64 = torch.cross(c64, n64, dim=-1).norm(dim=-1)
    angle_defined = mask & (nc64 > 0) & (nn64 > 0)
    sine = torch.full((L,), float('nan'), dtype=torch.float64)
    sine[angle_defined] = raw_cross_norm64[angle_defined]/(nc64[angle_defined]*nn64[angle_defined])
    orth_error = (F @ F.transpose(-1, -2)-torch.eye(3, dtype=torch.float64)).abs().amax(dim=(-2, -1))
    determinant = torch.linalg.det(F)
    flags = dict(c_norm_at_or_below_eps=c_norm <= EPS, n_norm_at_or_below_eps=n_norm <= EPS,
                 normalized_cross_at_or_below_eps=cross_norm <= EPS,
                 y_pre_norm_at_or_below_eps=y_pre_norm <= EPS,
                 exactly_degenerate=(nc64 == 0) | (nn64 == 0) | (raw_cross_norm64 == 0))
    arrays = dict(mask=mask.clone(), source_backbone=safe[0], ca=ca[0], centroid=centroid[0, 0],
        centered=centered.clone(), pca_canonical=pca_canonical.clone(), pca_basis=pca_basis.clone(),
        local_frames=frames, local_canonical=local, zero_canonical=zero,
        analysis_local=analysis_local, analysis_pca=analysis_pca, analysis_zero=zero.double(),
        local_source_reconstruction_error=analysis_local-local.double(),
        pca_source_reconstruction_error=analysis_pca-pca_canonical.double(),
        c_bond_norm=c_norm, n_bond_norm=n_norm, normalized_cross_norm=cross_norm,
        y_pre_norm=y_pre_norm, bond_angle_sine=sine, bond_angle_defined=angle_defined,
        orthogonality_max_abs_error=orth_error, frame_determinant=determinant, **flags)
    summary: dict[str, Any] = dict(residue_count=int(mask.sum()), normalization_epsilon=EPS,
        bond_angle_defined_count=int(angle_defined.sum()),
        bond_angle_undefined_count=int((mask & ~angle_defined).sum()))
    for name, value in flags.items():
        summary[name+'_count'] = int(value[mask].sum())
    for name in ('c_bond_norm', 'n_bond_norm', 'normalized_cross_norm', 'y_pre_norm',
                 'orthogonality_max_abs_error', 'frame_determinant'):
        summary[name] = _stats(arrays[name][mask].double())
    summary['bond_angle_sine'] = _stats(sine[angle_defined])
    for name in ('pca', 'local', 'zero'):
        feature = arrays[name+'_canonical'][mask].double()
        residual = arrays['analysis_'+name][mask]-feature
        summary[name] = dict(source_amplitude_mean_squared=float(_sq(feature).mean()),
                             source_amplitude_rms=float(_sq(feature).mean().sqrt()),
                             source_reconstruction_max_abs_error=float(residual.abs().max()))
    return dict(arrays=arrays, summary=summary)


@torch.no_grad()
def pair_audit(frame0: dict[str, Any], frame1: dict[str, Any], R: torch.Tensor,
               mask: torch.Tensor) -> dict[str, Any]:
    """Use the supplied proper row rotation; never fit PCA or a new alignment.

    Exact decomposition uses basis_term=L1-(Y R)F0^T. The equivalent transported
    expression (Y R)(F1 R-F0)^T is retained separately, with its numerical
    orthogonality-roundoff residual. No angles are assigned to degenerate local
    frames. Per-residue frame differences and lever arms are descriptive only.
    """
    _tensor(R, (3, 3), torch.float64, 'R')
    _require(bool(torch.isfinite(R).all())
             and bool(torch.allclose(R.T@R, torch.eye(3, dtype=torch.float64), atol=ROTATION_ATOL, rtol=0))
             and abs(float(torch.linalg.det(R))-1.) <= ROTATION_ATOL, 'R must be proper orthogonal')
    a0, a1 = frame0['arrays'], frame1['arrays']
    _mask(mask, len(a0['mask']))
    _require(torch.equal(a0['mask'], mask) and torch.equal(a1['mask'], mask), 'fixed frame masks differ')
    X, Y = a0['centered'][mask].double(), a1['centered'][mask].double()
    F0, F1 = a0['local_frames'][mask].double(), a1['local_frames'][mask].double()
    YR = Y@R
    transported_F1 = F1@R
    frame_change = transported_F1-F0
    old_basis_next = torch.einsum('lkc,lc->lk', F0, YR)
    shape_term = torch.einsum('lkc,lc->lk', F0, YR-X)
    basis_term = a1['analysis_local'][mask]-old_basis_next
    transported_basis_term = torch.einsum('lkc,lc->lk', frame_change, YR)
    cross_term = 2*(shape_term*basis_term).sum(-1)
    arrays = dict(valid_indices=torch.nonzero(mask, as_tuple=False).flatten(), R=R.clone(),
        local_shape_term=shape_term, local_basis_term=basis_term,
        local_shape_squared=_sq(shape_term), local_basis_squared=_sq(basis_term), local_cross=cross_term,
        local_transported_basis_term=transported_basis_term,
        local_transport_roundoff=transported_basis_term-basis_term,
        transported_next_local_frames=transported_F1, local_frame_change=frame_change,
        local_frame_change_frobenius=frame_change.square().sum((-2, -1)).sqrt(),
        source_centered_lever_arm0=X.norm(dim=-1), source_centered_lever_arm1=Y.norm(dim=-1),
        aligned_source_shape_displacement=YR-X,
        aligned_source_shape_displacement_norm=(YR-X).norm(dim=-1))
    summary: dict[str, Any] = dict(residue_count=int(mask.sum()),
        supplied_R_determinant=float(torch.linalg.det(R)),
        supplied_R_orthogonality_max_abs_error=float((R.T@R-torch.eye(3, dtype=torch.float64)).abs().max()))
    for name in ('pca', 'local', 'zero'):
        source0, source1 = a0[name+'_canonical'][mask].double(), a1[name+'_canonical'][mask].double()
        analysis0, analysis1 = a0['analysis_'+name][mask], a1['analysis_'+name][mask]
        source_jump, analysis_jump = source1-source0, analysis1-analysis0
        arrays.update({name+'__source_jump': source_jump, name+'__analysis_jump': analysis_jump,
                       name+'__source_jump_squared': _sq(source_jump),
                       name+'__analysis_jump_squared': _sq(analysis_jump),
                       name+'__source_amplitude0': source0.norm(dim=-1),
                       name+'__source_amplitude1': source1.norm(dim=-1),
                       name+'__analysis_minus_source_jump': analysis_jump-source_jump})
        summary[name] = dict(source_jump_mean_squared=float(_sq(source_jump).mean()),
            source_jump_rms=float(_sq(source_jump).mean().sqrt()),
            analysis_jump_mean_squared=float(_sq(analysis_jump).mean()),
            analysis_jump_rms=float(_sq(analysis_jump).mean().sqrt()),
            source_amplitude0_rms=float(_sq(source0).mean().sqrt()),
            source_amplitude1_rms=float(_sq(source1).mean().sqrt()),
            analysis_minus_source_jump_max_abs_error=float((analysis_jump-source_jump).abs().max()))
    local_jump = arrays['local__analysis_jump']
    arrays['local_decomposition_residual'] = local_jump-shape_term-basis_term
    arrays['local_squared_decomposition_residual'] = _sq(local_jump)-_sq(shape_term)-_sq(basis_term)-cross_term
    summary.update(local_shape_mean_squared=float(_sq(shape_term).mean()),
        local_basis_mean_squared=float(_sq(basis_term).mean()), local_cross_mean=float(cross_term.mean()),
        local_decomposition_max_abs_error=float(arrays['local_decomposition_residual'].abs().max()),
        local_squared_decomposition_max_abs_error=float(arrays['local_squared_decomposition_residual'].abs().max()),
        local_transport_roundoff_max_abs_error=float(arrays['local_transport_roundoff'].abs().max()))
    for name in ('local_frame_change_frobenius', 'source_centered_lever_arm0',
                 'source_centered_lever_arm1', 'aligned_source_shape_displacement_norm'):
        summary[name] = _stats(arrays[name])
    return dict(arrays=arrays, summary=summary)
