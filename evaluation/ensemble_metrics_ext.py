"""Extended ensemble-level metrics (new file; the original ``ensemble_metrics``
module is left untouched).

Implements the metric family requested for the dual-dataset ensemble report:

    * ``rmsf_profile``            -- already in ``ensemble_metrics`` (re-exported)
    * ``root_mean_w2``            -- Root-Mean W2  (per-residue position W2)
    * ``variance_w2``             -- Variance W2   (per-residue fluctuation W2)
    * ``md_pca_w2``               -- MD-PCA W2     (slow-mode projection W2)
    * ``weak_contact_jaccard``    -- Weak-contact Jaccard
    * ``transient_contact_jaccard`` -- Transient-contacts Jaccard

All W2 quantities below are the 1-D Wasserstein-2 distance between two empirical
distributions, computed exactly from sorted samples (no histogram binning).

Conventions match ``ensemble_metrics``: coordinates are ``[N, L, 3]`` CA traces
(one ensemble = N sampled conformers of the same protein), ``mask`` is a boolean
``[L]`` per-residue validity mask.
"""

from __future__ import annotations

import torch
from torch import Tensor

from evaluation.ensemble_metrics import (  # noqa: F401  (re-exported for the suite)
    align_ensemble,
    contact_probability_map,
    covariance_slow_modes,
    rmsf_profile,
)

__all__ = [
    "w2_1d",
    "rmsf_profile_aligned",
    "root_mean_w2",
    "variance_w2",
    "md_pca_w2",
    "weak_contact_jaccard",
    "weak_contact_jaccard_full",
    "transient_contact_jaccard",
    "transient_contact_jaccard_full",
    "extended_ensemble_summary",
]


# --------------------------------------------------------------------------- #
# core: exact 1-D Wasserstein-2 between two sample sets
# --------------------------------------------------------------------------- #
def w2_1d(a: Tensor, b: Tensor) -> Tensor:
    """Exact Wasserstein-2 distance between two 1-D empirical distributions.

    ``a`` and ``b`` may have different lengths; both are flattened and sorted.
    The value is the L2 distance between the (quantile) inverse CDFs, i.e.

        W2(a, b)^2 = integral_0^1 (F_a^{-1}(q) - F_b^{-1}(q))^2 dq

    approximated on the union grid ``linspace(0, 1, M)`` with ``M = max(na, nb)``.
    Returns a scalar tensor.
    """
    a = a.reshape(-1).to(torch.float64)
    b = b.reshape(-1).to(torch.float64)
    if a.numel() == 0 and b.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=a.device)
    if a.numel() == 0 or b.numel() == 0:
        return torch.tensor(float("inf"), dtype=torch.float64, device=a.device)
    a = torch.sort(a).values
    b = torch.sort(b).values
    m = max(a.numel(), b.numel())
    q = torch.linspace(0.0, 1.0, m, dtype=torch.float64, device=a.device)
    # linear-interpolation quantile function
    qa = _quantile(a, q)
    qb = _quantile(b, q)
    diff = (qa - qb) ** 2
    w2_sq = torch.trapz(diff, q) if m > 1 else diff.mean()
    return torch.sqrt(w2_sq.clamp_min(0.0))


def _quantile(sorted_x: Tensor, q: Tensor) -> Tensor:
    """Quantile function of a *sorted* sample at probabilities ``q`` (linear interp)."""
    n = sorted_x.numel()
    if n == 1:
        return sorted_x.expand_as(q)
    pos = q * (n - 1)
    lo = torch.floor(pos).long().clamp(0, n - 2)
    frac = pos - lo.to(pos.dtype)
    return sorted_x[lo] * (1.0 - frac) + sorted_x[lo + 1] * frac


def _masked(x: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return x
    return x[..., mask.to(x.device).bool()]


# --------------------------------------------------------------------------- #
# ROOT MEAN W2 -- per-residue CA position distribution
# --------------------------------------------------------------------------- #
def root_mean_w2(
    pred_x: Tensor,
    true_x: Tensor,
    mask: Tensor | None = None,
    align: bool = True,
    ref: Tensor | None = None,
) -> Tensor:
    """Root-Mean W2 over residues of the per-residue CA-position W2 distance.

    For each residue ``i`` we take the two scalar distributions
    ``{pred_x[n, i, :]}_n`` and ``{true_x[n, i, :]}_n`` (3·N values each), compute
    their 1-D W2, and return the root-mean-square over residues::

        RootMeanW2 = sqrt( mean_i  W2_i^2 )

    When ``align=True`` both ensembles are first superposed onto ``ref`` (an
    ``[L, 3]`` reference; defaults to the true-ensemble mean) so that the W2
    measures *internal* position spread rather than rigid-body drift.
    Returns a scalar tensor (Angstrom).
    """
    if pred_x.dim() != 3 or true_x.dim() != 3:
        raise ValueError("pred_x / true_x must be [N, L, 3]")
    if align:
        pred_x, true_x = _align_pair(pred_x, true_x, mask, _default_ref(true_x, ref))
    L = min(pred_x.shape[1], true_x.shape[1])
    per_res = torch.empty(L, dtype=torch.float64, device=pred_x.device)
    for i in range(L):
        per_res[i] = w2_1d(pred_x[:, i, :], true_x[:, i, :]) ** 2
    per_res = _masked(per_res, mask)
    return torch.sqrt(per_res.mean())


# --------------------------------------------------------------------------- #
# VARIANCE W2 -- per-residue fluctuation magnitude distribution
# --------------------------------------------------------------------------- #
def variance_w2(
    pred_x: Tensor,
    true_x: Tensor,
    mask: Tensor | None = None,
    align: bool = True,
    ref: Tensor | None = None,
) -> Tensor:
    """W2 between the predicted and true *per-residue fluctuation* distributions.

    For each ensemble we compute the per-residue mean-squared displacement from
    its own mean (the "variance profile", i.e. RMSF^2) and then take the 1-D W2
    between the two profiles over residues::

        VarianceW2 = W2( { RMSF_pred(i)^2 }_i , { RMSF_true(i)^2 }_i )

    This compares *how much fluctuation* the model reproduces, independent of
    which residue moves (that is what ``root_mean_w2`` / RMSF MAE capture).
    """
    if align:
        pred_x, true_x = _align_pair(pred_x, true_x, mask, _default_ref(true_x, ref))
    vp = _variance_profile(pred_x)                               # [L]
    vt = _variance_profile(true_x)
    vp = _masked(vp, mask)
    vt = _masked(vt, mask)
    return w2_1d(vp, vt)


# --------------------------------------------------------------------------- #
# MD-PCA W2 -- projections onto the true slow modes
# --------------------------------------------------------------------------- #
def md_pca_w2(
    pred_x: Tensor,
    true_x: Tensor,
    mask: Tensor | None = None,
    k: int = 3,
    align: bool = True,
    ref: Tensor | None = None,
) -> Tensor:
    """W2 of the slow-mode projections: RMS over the top-``k`` MD-PCA modes.

    The slow modes (eigenvectors of the true CA covariance, see
    :func:`covariance_slow_modes`) define an interpretable collective coordinate
    basis.  We project both ensembles onto each of the ``k`` mode directions and
    take the 1-D W2 between the projected distributions::

        MDPCAW2 = sqrt( mean_{j<k}  W2(P_true @ v_j , P_pred @ v_j)^2 )

    This is a distribution-aware analogue of comparing leading principal
    component histograms.  Returns a scalar tensor (Angstrom).
    """
    if align:
        pred_x, true_x = _align_pair(pred_x, true_x, mask, _default_ref(true_x, ref))
    k = int(k)
    if k <= 0:
        raise ValueError("k must be >= 1")
    L = pred_x.shape[1]                                          # residues (unmasked)
    modes = _true_slow_modes(true_x, mask, k, L=L)              # [k, L, 3]
    acc = torch.zeros((), dtype=torch.float64, device=pred_x.device)
    used = 0
    for v in modes:                                              # [L, 3]
        # projection of the (unmasked, common-L) clouds onto the mode direction
        pp = (pred_x - pred_x.mean(dim=0, keepdim=True)) * v[None]
        pt = (true_x - true_x.mean(dim=0, keepdim=True)) * v[None]
        if mask is not None:
            m = mask.to(pp.device).bool()
            pp = pp[:, m, :]
            pt = pt[:, m, :]
        acc = acc + w2_1d(pp, pt) ** 2
        used += 1
    return torch.sqrt(acc / max(used, 1))


# --------------------------------------------------------------------------- #
# WEAK CONTACT JACCARD
# --------------------------------------------------------------------------- #
def weak_contact_jaccard(
    pred_x: Tensor,
    true_x: Tensor,
    mask: Tensor | None = None,
    cutoff: float = 8.0,
    tau_lo: float = 0.05,
    tau_hi: float = 0.5,
) -> Tensor:
    """Jaccard of *weak / transiently-populated* contacts.

    Contacts are defined from the ensemble contact probability map
    ``P(i,j) = P(d_ij < cutoff)``.  "Weak" contacts are those whose probability
    lies in the ambiguous band ``[tau_lo, tau_hi)`` -- i.e. not always formed and
    not essentially never formed.  We take the binary set on the *true* ensemble
    and the binary set on the *predicted* ensemble and return

        J = |A ∩ B| / |A ∪ B|   in [0, 1]  (1 = perfect agreement).

    ``oracle`` (pred = true) gives 1.0 by construction.
    """
    J, _na, _nb = _contact_band_jaccard(
        pred_x, true_x, mask, cutoff, tau_lo, tau_hi,
        geo_mask="upper", per_contact=False)
    return J


# --------------------------------------------------------------------------- #
# TRANSIENT CONTACTS JACCARD
# --------------------------------------------------------------------------- #
def transient_contact_jaccard(
    pred_x: Tensor,
    true_x: Tensor,
    mask: Tensor | None = None,
    cutoff: float = 8.0,
    tau_lo: float = 0.01,
    tau_hi: float = 0.99,
    per_contact: bool = True,
) -> Tensor:
    """Jaccard of *transient* contacts (those that form and break over time).

    A contact is *transient* when, at the level of the ensemble, it is neither
    permanently formed nor permanently absent -- its probability strictly
    exceeds the always/never bands.  With ``per_contact=True`` we instead score,
    for every residue pair, whether the model and truth agree on the transient
    (ambiguity) label, and return the Jaccard of the two binary label maps off
    the union of both label sets.  This makes the metric sensitive to *which*
    pairs are dynamic, not just the global fraction.

    ``oracle`` (pred = true) gives 1.0 by construction.
    """
    if not per_contact:
        J, _na, _nb = _contact_band_jaccard(
            pred_x, true_x, mask, cutoff, tau_lo, tau_hi,
            geo_mask="upper", per_contact=False)
        return J
    pm = _contact_prob(pred_x, mask, cutoff)
    tm = _contact_prob(true_x, mask, cutoff)
    geo = _upper_geo(pm, mask)
    a = ((tm > tau_lo) & (tm < tau_hi) & geo)
    b = ((pm > tau_lo) & (pm < tau_hi) & geo)
    J, _na, _nb = _jaccard(a, b)
    return J


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _default_ref(true_x: Tensor, ref: Tensor | None) -> Tensor | None:
    """Reference cloud used by the W2 metrics.

    If ``ref`` is given we return it; otherwise we return ``None`` and let the
    caller simply align each ensemble to its own first frame (the MD convention
    used by :func:`evaluation.ensemble_metrics.align_ensemble`).
    """
    return ref


def _align_pair(pred_x: Tensor, true_x: Tensor, mask: Tensor | None, ref: Tensor | None):
    """Align both ensembles consistently before a W2 comparison.

    * ``ref is None``: align each ensemble to its own first frame (sufficient for
      removing intra-ensemble rigid-body drift when both ensembles are centred
      on the same protein).
    * ``ref`` given ([L, 3]): align *both* ensembles onto the same cloud so the
      comparison is a true per-residue superposition.
    """
    if ref is None:
        return align_ensemble(pred_x, mask), align_ensemble(true_x, mask)
    return _align_to(pred_x, ref, mask), _align_to(true_x, ref, mask)


def _align_to(x: Tensor, ref: Tensor, mask: Tensor | None) -> Tensor:
    """Kabsch-align every frame of ``x`` [N,L,3] onto the ``[L,3]`` cloud ``ref``."""
    if mask is None:
        m = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
    else:
        m = mask.to(x.device, x.dtype)
    mm = m[:, None]
    denom = m.sum().clamp_min(1.0)
    ref_centroid = (ref * mm).sum(0, keepdim=True) / denom
    refc = (ref - ref_centroid) * mm
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        p = (x[i] - (x[i] * mm).sum(0, keepdim=True) / denom) * mm
        Hm = p.T @ refc
        U, S, Vh = torch.linalg.svd(Hm.to(torch.float64))
        dd = 1.0 if float(torch.det((U @ Vh).detach())) >= 0 else -1.0
        D = torch.diag(torch.tensor([1.0, 1.0, dd], device=x.device, dtype=torch.float64))
        R = (U @ D @ Vh).to(x.dtype)
        out[i] = p @ R + ref_centroid
    return out


def rmsf_profile_aligned(x: Tensor, mask: Tensor | None = None) -> Tensor:
    """Aligned per-residue RMSF profile [L] (for pooling into a global RMSF-r).

    Reuses :func:`evaluation.ensemble_metrics.rmsf_profile` after Kabsch-aligning
    each frame to the ensemble's own first frame.  Masked residues are returned as
    ``nan`` so the caller can concatenate profiles across windows and drop them.
    """
    r = rmsf_profile(align_ensemble(x, mask), mask)              # [L]
    if mask is not None:
        m = mask.to(r.device).bool()
        r = r.clone()
        r[~m] = float("nan")
    return r


def _variance_profile(x: Tensor) -> Tensor:
    """Per-residue mean squared displacement from the ensemble mean [L]."""
    mu = x.mean(dim=0, keepdim=True)                             # [1, L, 3]
    return (x - mu).square().sum(dim=-1).mean(dim=0)             # [L]


def _true_slow_modes(true_x: Tensor, mask: Tensor | None, k: int,
                     L: int | None = None) -> Tensor:
    """Top-k CA-covariance eigenvectors, shape [k, L, 3].

    When ``L`` is given the modes are computed on all ``L`` residues (mask only
    selecting which residues inform the covariance fit would change the vector
    length; we keep the full length so projection is well defined).
    """
    x = true_x.float()
    L = x.shape[1] if L is None else L
    X = x.reshape(x.shape[0], -1)                                # [N, 3L]
    X = X - X.mean(dim=0, keepdim=True)
    # Compute on CPU in float64: eigh on a 3L x 3L matrix (L up to ~330) can
    # fail to converge on GPU for ill-conditioned / near-degenerate spectra.
    cov = ((X.t() @ X) / max(X.shape[0] - 1, 1)).double().cpu()
    try:
        evals, evecs = torch.linalg.eigh(cov)
    except Exception:
        # extremely degenerate: fall back to SVD of the centered data
        _, _, Vh = torch.linalg.svd(X.double().cpu(), full_matrices=False)
        v = Vh[:k]
        return v.to(true_x.device).reshape(-1, L, 3)
    idx = torch.argsort(evals, descending=True)[:k]              # top-k
    v = evecs[:, idx].t()                                        # [k, 3L]
    return v.to(true_x.device).reshape(-1, L, 3)


def _contact_prob(x: Tensor, mask: Tensor | None, cutoff: float) -> Tensor:
    return contact_probability_map(x, mask=mask, cutoff=cutoff)  # [L, L]


def _upper_geo(p: Tensor, mask: Tensor | None) -> Tensor:
    L = p.shape[0]
    iu = torch.triu(torch.ones(L, L, dtype=torch.bool, device=p.device), 1)
    if mask is not None:
        m = mask.to(p.device).bool()
        iu = iu & m[:, None] & m[None, :]
    return iu


def _jaccard(a: Tensor, b: Tensor):
    """Jaccard of two boolean sets.

    Returns ``(J, na, nb)`` where ``na = |A|``, ``nb = |B|``.

    Degenerate guard: if BOTH sets are empty we return ``J = 1.0`` ONLY when the
    truth set is *also* empty (``na == 0``).  If the truth set is non-empty but
    the prediction set is empty, ``J = 0.0`` (a genuine miss, e.g. a delta /
    binary baseline that cannot express probabilistic contacts at all).  The set
    sizes are returned so callers can expose *why* a value is 0 or 1.
    """
    a = a.bool()
    b = b.bool()
    na = int(a.sum().item())
    nb = int(b.sum().item())
    union = na + nb - int((a & b).sum().item())
    if union == 0:
        return torch.tensor(1.0, dtype=torch.float64, device=a.device), na, nb
    inter = int((a & b).sum().item())
    return (torch.tensor(inter / union, dtype=torch.float64, device=a.device), na, nb)


def _contact_band_jaccard(
    pred_x, true_x, mask, cutoff, tau_lo, tau_hi, geo_mask="upper", per_contact=False,
):
    pm = _contact_prob(pred_x, mask, cutoff)
    tm = _contact_prob(true_x, mask, cutoff)
    geo = _upper_geo(pm, mask)
    a = (tm >= tau_lo) & (tm < tau_hi) & geo
    b = (pm >= tau_lo) & (pm < tau_hi) & geo
    return _jaccard(a, b)


def weak_contact_jaccard_full(
    pred_x, true_x, mask=None, cutoff: float = 8.0,
    tau_lo: float = 0.05, tau_hi: float = 0.5,
) -> dict:
    """Weak-contact Jaccard **plus the set sizes** (``n_true``, ``n_pred``, ``n_inter``).

    Exposing the sizes is essential: a single-frame *persistence* ensemble has a
    strictly binary ({0,1}) contact-probability map, so its weak-band set is
    always EMPTY (``n_pred = 0``).  When the truth set is non-empty this makes
    ``J = 0`` by construction -- not because the model is bad, but because a delta
    distribution cannot express a fractional contact probability at all.
    """
    pm = _contact_prob(pred_x, mask, cutoff)
    tm = _contact_prob(true_x, mask, cutoff)
    geo = _upper_geo(pm, mask)
    a = (tm >= tau_lo) & (tm < tau_hi) & geo
    b = (pm >= tau_lo) & (pm < tau_hi) & geo
    J, na, nb = _jaccard(a, b)
    n_inter = int((a & b).sum().item())
    return {"weak_contact_j": float(J), "n_true": na, "n_pred": nb, "n_inter": n_inter}


def transient_contact_jaccard_full(
    pred_x, true_x, mask=None, cutoff: float = 8.0,
    tau_lo: float = 0.01, tau_hi: float = 0.99,
) -> dict:
    """Transient-contacts Jaccard **plus the set sizes** (see the weak variant)."""
    pm = _contact_prob(pred_x, mask, cutoff)
    tm = _contact_prob(true_x, mask, cutoff)
    geo = _upper_geo(pm, mask)
    a = ((tm > tau_lo) & (tm < tau_hi) & geo)
    b = ((pm > tau_lo) & (pm < tau_hi) & geo)
    J, na, nb = _jaccard(a, b)
    n_inter = int((a & b).sum().item())
    return {"transient_contacts_j": float(J), "n_true": na, "n_pred": nb, "n_inter": n_inter}


# --------------------------------------------------------------------------- #
# aggregate summary
# --------------------------------------------------------------------------- #
def extended_ensemble_summary(
    pred_x: Tensor,
    true_x: Tensor,
    mask: Tensor | None = None,
    *,
    cutoff: float = 8.0,
    pca_k: int = 3,
    weak_lo: float = 0.05,
    weak_hi: float = 0.5,
    trans_lo: float = 0.01,
    trans_hi: float = 0.99,
    align: bool = True,
    ref: Tensor | None = None,
) -> dict:
    """Compute the full extended metric family for one (pred, true) ensemble pair.

    Returns a dict of *scalars* (python floats) with keys::

        rmsf_mae, root_mean_w2, variance_w2, md_pca_w2,
        weak_contact_j, transient_contacts_j
    """
    from evaluation.ensemble_metrics import rmsf_profile_error

    ref = _default_ref(true_x, ref)
    out: dict[str, float] = {}
    rmsf_err = rmsf_profile_error(pred_x, true_x, mask)
    out["rmsf_mae"] = float(rmsf_err["rmsf_mae"])
    # RMSF-r: Pearson correlation of the two flexibility profiles (dimensionless).
    out["rmsf_corr"] = float(rmsf_err["rmsf_corr"])
    out["root_mean_w2"] = float(root_mean_w2(pred_x, true_x, mask, align=align, ref=ref))
    out["variance_w2"] = float(variance_w2(pred_x, true_x, mask, align=align, ref=ref))
    out["md_pca_w2"] = float(md_pca_w2(pred_x, true_x, mask, k=pca_k, align=align, ref=ref))
    wf = weak_contact_jaccard_full(pred_x, true_x, mask, cutoff, weak_lo, weak_hi)
    tf = transient_contact_jaccard_full(pred_x, true_x, mask, cutoff, trans_lo, trans_hi)
    out["weak_contact_j"] = wf["weak_contact_j"]
    out["transient_contacts_j"] = tf["transient_contacts_j"]
    out["weak_n_true"] = float(wf["n_true"])
    out["weak_n_pred"] = float(wf["n_pred"])
    out["transient_n_true"] = float(tf["n_true"])
    out["transient_n_pred"] = float(tf["n_pred"])
    return out
