"""Observed-frame transport using only explicit peptide topology and CA inputs.

One shared PSF edge list supplies LOCAL row indices, not residue numbers or
``geometry_row_ids``. Its undirected graph is restricted to observed-valid
vertices before taking the complete radius-two neighborhood. The center itself
is included; there is no five-point cap on branched graphs. Vectors are relative
to that residue's CA, never to the neighborhood centroid.

This is a deterministic, non-trainable orientation prior. It neither predicts
coordinates nor enforces peptide closure. Target atoms/masks have no interface.
"""
from __future__ import annotations

import torch


VERSION = "backbone_frame_transport_v1_psf_two_hop_anchored_kabsch"
RANK_RELATIVE_TOLERANCE = 1e-6
RANK_ABSOLUTE_TOLERANCE = 1e-10  # singular values of X.T @ Y, in Angstrom**2


def _integer_tensor(value, *, name, device):
    value = torch.as_tensor(value, device=device)
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must contain integer IDs")
    return value.to(torch.long)


@torch.no_grad()
def transport_observed_frames(
    observed_ca: torch.Tensor,
    current_ca: torch.Tensor,
    observed_R: torch.Tensor,
    *,
    residue_mask: torch.Tensor,
    peptide_edges: torch.Tensor,
    geometry_row_ids: torch.Tensor,
) -> dict:
    """Return transported frame seeds and auditable local fit statistics.

    ``current_ca`` is [B,L,3]; observed CA and native frames are respectively
    [1 or B,L,3] and [1 or B,L,3,3]. All three share a floating dtype/device.
    A bool mask [L], [1,L] or [B,L] describes OBSERVED validity; invalid vertices
    cannot bridge paths. One integer edge list [E,2] and unique identities [L]
    are shared by the batch. Reverse directed edges collapse to one undirected
    link; duplicate identical directed edges, self edges and bad indices fail.

    Float64 Kabsch uses H=X.T@Y=U S Vh and Q=V diag(1,1,det(V@U.T)) U.T.
    Rank counts singular values strictly greater than BOTH fixed tolerances.
    Rank < 2 retains observed_R. A sample with every valid current CA exactly
    equal to observed CA gets an exact identity Q, including degenerate cases.

    ``frame_seed`` and ``rotation`` have coordinate dtype and [B,L,3,3] shape.
    ``audit`` tensors have [B,L] shape (singular_values additionally ends in 3):
    valid, fallback, exact_identity, neighbor_count (includes center), rank,
    fit_rmsd (Angstrom; center-anchored vector residual, includes center), det,
    orthogonality (Frobenius norm of Q.T@Q-I), and singular_values. Float audit
    tensors are float64. Invalid rows return identity frames/Q, fallback=True,
    zero counts/rank/residual, det=1 and orthogonality=0; valid disambiguates them.
    Fits audit the emitted, dtype-cast Q. Inputs are never modified, and no
    output retains a gradient connection to the inputs.
    """
    if (not torch.is_tensor(current_ca) or current_ca.ndim != 3
            or current_ca.shape[-1] != 3 or min(current_ca.shape[:2]) < 1
            or not current_ca.is_floating_point()):
        raise ValueError("current_ca must be nonempty floating [B,L,3]")
    batch, length, _ = current_ca.shape
    device, dtype = current_ca.device, current_ca.dtype
    for name, value, suffix in (("observed_ca", observed_ca, (length, 3)),
                                ("observed_R", observed_R, (length, 3, 3))):
        if (not torch.is_tensor(value) or value.ndim != len(suffix) + 1
                or value.shape[0] not in (1, batch) or value.shape[1:] != suffix):
            raise ValueError(f"{name} must have batch 1 or B and trailing shape {suffix}")
        if value.dtype != dtype or value.device != device:
            raise ValueError("CA coordinates and observed_R must share floating dtype/device")
    observed_ca = observed_ca.expand(batch, length, 3)
    observed_R = observed_R.expand(batch, length, 3, 3)
    valid = torch.as_tensor(residue_mask, device=device)
    if valid.dtype != torch.bool or valid.shape not in ((length,), (1, length), (batch, length)):
        raise ValueError("residue_mask must be bool [L], [1,L], or [B,L]")
    valid = valid.expand(batch, length)
    ids = _integer_tensor(geometry_row_ids, name="geometry_row_ids", device=device)
    if ids.shape != (length,) or ids.unique().numel() != length:
        raise ValueError("geometry_row_ids must contain L unique row identities")
    edges = _integer_tensor(peptide_edges, name="peptide_edges", device=device)
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("peptide_edges must be [E,2], including empty [0,2]")
    if bool(((edges < 0) | (edges >= length)).any()):
        raise ValueError("peptide edge outside local row indices")
    if bool((edges[:, 0] == edges[:, 1]).any()):
        raise ValueError("peptide self edge is invalid")
    if edges.unique(dim=0).shape[0] != edges.shape[0]:
        raise ValueError("duplicate directed peptide edge")
    for name, value in (("observed_ca", observed_ca), ("current_ca", current_ca),
                         ("observed_R", observed_R)):
        if not bool(torch.isfinite(value[valid]).all()):
            raise ValueError(f"nonfinite valid {name}")

    eye = torch.eye(3, dtype=dtype, device=device)
    obs = torch.where(valid[..., None], observed_ca, 0).double()
    cur = torch.where(valid[..., None], current_ca, 0).double()
    old_R = torch.where(valid[..., None, None], observed_R, eye)
    adjacency = torch.zeros(length, length, dtype=torch.bool, device=device)
    adjacency[edges[:, 0], edges[:, 1]] = True
    adjacency[edges[:, 1], edges[:, 0]] = True
    adjacency = adjacency[None] & valid[:, :, None] & valid[:, None, :]
    within_two = adjacency | (adjacency.double() @ adjacency.double() > 0)
    within_two |= torch.eye(length, dtype=torch.bool, device=device)[None] & valid[:, :, None]
    counts = within_two.sum(-1)
    X = torch.where(within_two[..., None], obs[:, None] - obs[:, :, None], 0)
    Y = torch.where(within_two[..., None], cur[:, None] - cur[:, :, None], 0)
    covariance = X.transpose(-2, -1) @ Y
    if not bool(torch.isfinite(covariance).all()):
        raise ValueError("nonfinite anchored Kabsch covariance")
    U, singular, Vh = torch.linalg.svd(covariance)
    threshold = torch.maximum(singular[..., :1] * RANK_RELATIVE_TOLERANCE,
                              singular.new_tensor(RANK_ABSOLUTE_TOLERANCE))
    rank = (singular > threshold).sum(-1)
    fallback = ~valid | (rank < 2)
    V = Vh.transpose(-2, -1)
    correction = torch.ones_like(singular)
    correction[..., -1] = torch.where(torch.linalg.det(V @ U.transpose(-2, -1)) < 0, -1., 1.)
    Q = (V * correction[..., None, :]) @ U.transpose(-2, -1)
    # Masked NaNs have already been removed; equality is exact in input dtype.
    unchanged_sample = (obs == cur).all(dim=(-2, -1))
    exact_identity = valid & unchanged_sample[:, None]
    keep_observed = fallback | exact_identity
    Q = torch.where(keep_observed[..., None, None], eye.double(), Q)
    rotation = Q.to(dtype)
    frame_seed = (Q @ old_R.double()).to(dtype)
    frame_seed = torch.where(keep_observed[..., None, None], old_R, frame_seed)
    emitted_Q = rotation.double()
    residual = (emitted_Q[:, :, None] @ X[..., None]).squeeze(-1) - Y
    fit_rmsd = (residual.square().sum(dim=(-2, -1)) / counts.clamp_min(1)).sqrt()
    determinant = torch.linalg.det(emitted_Q)
    orthogonality = torch.linalg.matrix_norm(emitted_Q.transpose(-2, -1) @ emitted_Q - eye.double())
    if not all(bool(torch.isfinite(value).all()) for value in
               (frame_seed, rotation, fit_rmsd, determinant, orthogonality, singular)):
        raise ValueError("nonfinite frame transport output")
    return {"frame_seed": frame_seed, "rotation": rotation,
            "audit": {"valid": valid.clone(), "fallback": fallback,
                      "exact_identity": exact_identity, "neighbor_count": counts,
                      "rank": rank, "fit_rmsd": fit_rmsd, "det": determinant,
                      "orthogonality": orthogonality, "singular_values": singular}}
