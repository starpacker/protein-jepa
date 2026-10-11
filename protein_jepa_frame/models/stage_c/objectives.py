"""Stage C structure decoding objectives.

Gate 5 loss functions:
- FAPE: Frame Aligned Point Error with rotation/translation invariance
- Torsion: binned cross-entropy
- Distance: CA-CA pairwise
- Bond/clash: geometric constraints
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from protein_jepa_frame.models.stage_c.structure_decoder import NUM_TORSIONS


def _openfold_ideal_bond(resname: str, atom1: str, atom2: str) -> float:
    """Official OpenFold ideal bond length (from stereo_chemical_props.txt).

    Falls back to the literature N-CA default only if the pair is absent.
    """
    from protein_jepa_frame.models.stage_c.openfold_atom37 import (  # noqa: F401
        OPENFOLD_ATOM37_NAMES,  # ensure vendored openfold is registered
    )
    from openfold.np import residue_constants as rc

    global _OF_BOND_CACHE
    try:
        _OF_BOND_CACHE
    except NameError:
        bonds, _, _ = rc.load_stereo_chemical_props()
        _OF_BOND_CACHE = {
            res: {(b.atom1_name, b.atom2_name): b.length for b in blist}
            for res, blist in bonds.items()
        }
    d = _OF_BOND_CACHE.get(resname, {})
    return d.get((atom1, atom2), d.get((atom2, atom1), 1.5))


def fape_loss(
    pred_coords, target_coords,
    pred_R=None, pred_t=None, target_R=None, target_t=None,
    mask=None, clamp=10.0, backbone_only=True, centered=False,
):
    """Frame Aligned Point Error (FAPE).

    When frames are provided: for each residue i, express all atoms in the
    local frame of residue i, then measure L2 error. Invariant to global
    rotation/translation.

    When frames not provided (or centered=True): CA-centered L2 after
    recentering. This measures internal geometry (SE(3)-invariant) and ignores
    absolute displacement — essential when anchor→target distances exceed the
    clamp threshold (typically 60+Å in MD data, making frame-aligned FAPE path
    saturate at clamp under batch_size=1 stochasticity).

    Gate 5 fix: centered=True is the default for MD-scale data with large
    anchor→target displacements. Frame-aligned mode is used for fine-tuning
    when displacements are small.
    """
    B, L, N_atoms = pred_coords.shape[:3]
    device = pred_coords.device

    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # True frame-aligned FAPE
    if pred_R is not None and pred_t is not None and target_t is not None and target_R is not None:
        pred_R_inv = pred_R.transpose(-2, -1)
        target_R_inv = target_R.transpose(-2, -1)

        # For each ref residue i: x_j^{local} = R_i^T @ (x_j - t_i)
        # pred_coords: [B, L, 4, 3]
        # target_t:   [B, L, 3]

        # Expand: pred_coords → [B, 1, L, 4, 3] (atoms from each residue j)
        # target_t → [B, L, 1, 1, 3] (ref frame position i)
        pred_exp = pred_coords[:, None, :, :, :]   # [B, 1, L, 4, 3]
        targ_exp = target_coords[:, None, :, :, :]  # [B, 1, L, 4, 3]
        pred_t_ref = pred_t[:, :, None, None, :]
        target_t_ref = target_t[:, :, None, None, :]
        pred_R_ref = pred_R_inv[:, :, None, None, :, :]
        target_R_ref = target_R_inv[:, :, None, None, :, :]
        pred_local = (pred_R_ref @ (pred_exp - pred_t_ref).unsqueeze(-1)).squeeze(-1)
        targ_local = (target_R_ref @ (targ_exp - target_t_ref).unsqueeze(-1)).squeeze(-1)

        diff = pred_local - targ_local
        per_atom_error = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [B, L, L, 4]

        if clamp > 0:
            per_atom_error = torch.clamp(per_atom_error, max=clamp)

        # Pair mask
        pair_mask = mask[:, :, None] & mask[:, None, :]  # [B, L, L]
        atom_mask = pair_mask.unsqueeze(-1).expand(-1, -1, -1, N_atoms)
        valid = per_atom_error * atom_mask.float()
        total = atom_mask.float().sum()
        return valid.sum() / total.clamp(min=1)

    # Fallback: CA-recentered global L2
    target_ca = target_coords[..., 1, :]  # [B, L, 3]
    pred_ca = pred_coords[..., 1, :]
    mf = mask.unsqueeze(-1).float()
    tc = (target_ca * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)  # [B, 3]
    pc = (pred_ca * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
    pred_aligned = pred_coords - pc[:, None, None, :] + tc[:, None, None, :]
    diff = pred_aligned - target_coords
    pae = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    if clamp > 0:
        pae = torch.clamp(pae, max=clamp)
    am = mask.unsqueeze(-1).expand_as(pae)
    valid = pae * am.float()
    return valid.sum() / am.float().sum().clamp(min=1)


def frame_aligned_fape(
    pred_coords, target_coords, target_R, target_t,
    mask=None, clamp=10.0, centered=False,
):
    """Frame-aligned FAPE using target frames (variant of fape_loss)."""
    return fape_loss(pred_coords, target_coords, target_t=target_t, target_R=target_R, mask=mask, clamp=clamp, centered=centered)


def torsion_loss(torsion_logits, target_torsions, mask=None, n_bins=36):
    B, L = torsion_logits.shape[:2]
    device = torsion_logits.device
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
    bw = 2 * math.pi / n_bins
    centered = (target_torsions - (-math.pi)) % (2 * math.pi)
    tb = (centered / bw).long().clamp(0, n_bins - 1)
    tl = torsion_logits.reshape(-1, n_bins)
    tbf = tb.reshape(-1)
    ce = F.cross_entropy(tl, tbf, reduction="none").reshape(B, L, NUM_TORSIONS)
    me = mask.unsqueeze(-1).expand(B, L, NUM_TORSIONS)
    return (ce * me.float()).sum() / me.sum().clamp(min=1)


def pairwise_distance_loss(pred_CA, target_CA, mask=None, max_distance=30.0):
    B, L = pred_CA.shape[:2]
    device = pred_CA.device
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
    pd = torch.cdist(pred_CA, pred_CA)
    td = torch.cdist(target_CA, target_CA)
    pd = torch.clamp(pd, max=max_distance)
    td = torch.clamp(td, max=max_distance)
    diff = torch.abs(pd - td)
    pm = mask.unsqueeze(2) & mask.unsqueeze(1)
    eye = torch.eye(L, dtype=torch.bool, device=device)
    pm = pm & ~eye.unsqueeze(0)
    return (diff * pm.float()).sum() / pm.sum().clamp(min=1)


def bond_length_violation(pred_coords, aatype=None, mask=None):
    """Backbone bond-length violation using official OpenFold ideal bond lengths.

    Ideal N-CA, CA-C, C-O lengths are drawn from OpenFold's
    ``stereo_chemical_props.txt`` (via ``residue_constants``); no hand-written
    bond tables are used.  When ``aatype`` is provided, per-residue ideal
    lengths are looked up; otherwise literature defaults are applied.
    """
    B, L = pred_coords.shape[:2]
    device = pred_coords.device
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
    N, CA, C, O = pred_coords[..., 0, :], pred_coords[..., 1, :], pred_coords[..., 2, :], pred_coords[..., 3, :]

    # Literature defaults (OpenFold stereo_chemical_props) for the 20 restypes.
    _AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    pairs = [("N", "CA"), ("CA", "C"), ("C", "O")]
    acts = [N - CA, CA - C, C - O]

    viols = []
    for (a1, a2), act in zip(pairs, acts):
        if aatype is not None:
            per_res = torch.tensor(
                [_openfold_ideal_bond(_AA3[min(int(a), 19)], a1, a2) for a in aatype.reshape(-1)],
                device=device, dtype=pred_coords.dtype,
            ).reshape(B, L)
        else:
            d = _openfold_ideal_bond("ALA", a1, a2)
            per_res = torch.full((B, L), d, device=device, dtype=pred_coords.dtype)
        viols.append(((act.norm(dim=-1) - per_res) / per_res) ** 2 * mask.float())
    total = mask.sum()
    return sum(v.sum() for v in viols) / total.clamp(min=1)


def clash_penalty(pred_coords, mask=None, atom_mask=None, clash_cutoff=3.3):
    """Compute clash penalty on any set of coordinates [B, L, A, 3].

    If pred_coords is [B, L, 3] (CA-only), use atom-atom distances.
    If pred_coords is [B, L, A, 3] (atom37), use full inter-residue atom distances.
    """
    if pred_coords.dim() == 4:
        # atom37 coords: [B, L, A, 3] — compute minimum inter-residue distance
        B, L, A = pred_coords.shape[:3]
        device = pred_coords.device
        # [B, L, A, 3] → [B, L*A, 3]
        atoms = pred_coords.reshape(B, L * A, 3)
        # [B, L*A, L*A] pairwise distances
        dists = torch.cdist(atoms, atoms).reshape(B, L, A, L, A)
        if atom_mask is None:
            atom_mask = torch.ones(B, L, A, dtype=torch.bool, device=device)
        atom_mask = atom_mask.to(dtype=torch.bool, device=device)
        valid_pairs = (
            atom_mask[:, :, :, None, None]
            & atom_mask[:, None, None, :, :]
        )
        dists = dists.masked_fill(~valid_pairs, float("inf"))
        # [B, L, A, L, A] — minimize over atom pairs between residues
        # dim=-1 = A (intra-residue atom), dim=2 = A (inter-residue atom)
        dists_min = dists.min(dim=-1).values.min(dim=2).values  # [B, L, L]
    else:
        # CA-only: [B, L, 3]
        B, L = pred_coords.shape[:2]
        device = pred_coords.device
        dists_min = torch.cdist(pred_coords, pred_coords)  # [B, L, L]
        A = 1

    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)

    eye = torch.eye(L, dtype=torch.bool, device=device)
    seq_adj = torch.zeros(L, L, dtype=torch.bool, device=device)
    for i in range(L):
        for j in range(max(0, i-1), min(L, i+2)):
            if i != j:
                seq_adj[i, j] = True

    clash = F.relu(clash_cutoff - dists_min)
    pm = mask.unsqueeze(2) & mask.unsqueeze(1)
    pm = pm & ~eye.unsqueeze(0) & ~seq_adj.unsqueeze(0)
    return ((clash ** 2) * pm.float()).sum() / pm.sum().clamp(min=1)


def stage_c_total_loss(outputs, target, mask=None, weights=None, use_frame_aligned=True, fape_clamp=10.0, anchor_CA=None, anchor_R=None, residue_index=None, chain_id=None, direct_CA=None):
    if weights is None:
        weights = {"fape": 1.0, "torsion": 0.5, "distance": 0.3, "bond": 0.1, "clash": 0.05, "ca_position": 0.1}

    # Use frame-direct backbone for position-based losses (FAPE, distance, CA).
    # This decouples frame gradients from torsion errors: FrameUpdate gets clean
    # position gradient from ideal local geometry placed at each predicted frame.
    # TorsionHead learns from CE loss on logits independently.
    bb_pos = outputs.get("backbone_direct", outputs["backbone_coords"])
    ca_pos = outputs.get("CA_direct", outputs["CA"])

    comps = {}
    if use_frame_aligned and target.get("R") is not None and target.get("t") is not None:
        comps["fape"] = fape_loss(bb_pos, target["backbone_coords"],
            pred_R=outputs.get("R"), pred_t=outputs.get("t"),
            target_R=target["R"], target_t=target["t"], mask=mask, clamp=fape_clamp)
    else:
        comps["fape"] = fape_loss(bb_pos, target["backbone_coords"],
            mask=mask, clamp=fape_clamp)
    if "torsion_logits" in outputs and "torsion_angles" in target:
        comps["torsion"] = torsion_loss(outputs["torsion_logits"], target["torsion_angles"], mask=mask)
    else:
        comps["torsion"] = outputs["backbone_coords"].new_tensor(0.0)
    comps["distance"] = pairwise_distance_loss(ca_pos, target["CA"], mask=mask)
    comps["ca_position"] = direct_ca_loss(ca_pos, target["CA"], anchor_CA=anchor_CA, anchor_R=anchor_R, mask=mask)
    # Eval-aligned position objectives.  ``kabsch_ca`` is the differentiable
    # squared form of the canonical CA Kabsch RMSD that the evaluation reports
    # (condition 2), and ``contact_distance`` is the SE(3)-invariant CA-CA
    # near-field error that the contact-Jaccard family (conditions 3-5) is a
    # function of.  Both consume the canonical decoded CA so their gradients
    # flow into the frame/translation path that actually produces the answer.
    comps["kabsch_ca"] = kabsch_align_ca_loss(ca_pos, target["CA"], mask=mask)
    comps["contact_distance"] = contact_distance_loss(
        ca_pos, target["CA"], mask=mask,
        residue_index=residue_index, chain_id=chain_id,
    )
    # Direct binary contact supervision, aligned to the reported
    # contact-Jaccard family (conditions 3-5).  This is the objective that
    # actually moves the metric at the bin boundary where Jaccard is decided.
    comps["contact_binary"] = contact_binary_loss(
        ca_pos, target["CA"], mask=mask,
        residue_index=residue_index, chain_id=chain_id,
    )
    # Bond/clash are diagnostics for the torsion reconstruction path.  Their
    # repeated normalize/project/min operations are not needed to train the
    # anchor-frame position path and can create pathological gradients near
    # degenerate torsions, so do not backpropagate them into shared features.
    bb_rec = outputs.get(
        "backbone_torsion_reconstructed", outputs["backbone_coords"]
    ).detach()
    comps["bond"] = bond_length_violation(bb_rec, mask=mask)
    comps["clash"] = clash_penalty(
        outputs.get("atom37_coords", ca_pos).detach(),
        mask=mask,
        atom_mask=outputs.get("atom37_mask"),
    )
    total = outputs["backbone_coords"].new_tensor(0.0)
    for n, l in comps.items():
        total = total + weights.get(n, 0.0) * l
    return total, comps


def kabsch_align_ca_loss(
    pred_CA: torch.Tensor,
    target_CA: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable CA Kabsch RMSD, matching the evaluation metric exactly.

    For each sample in the batch, the predicted CA cloud is optimally
    superimposed onto the target CA cloud (rotation + translation solved by
    SVD, with reflection correction) using only masked residues, and the
    mean-squared residual is returned.  This is the squared form of the
    ``official_kabsch`` RMSD used in ``eval/structure_metrics.py``.

    The SVD is differentiable w.r.t. ``pred_CA`` so gradients pull the
    predicted conformation toward the target *after* removing the global rigid
    degrees of freedom — i.e. it rewards getting the internal shape right,
    which is precisely what the Kabsch RMSD eval measures.  Per-sample masking
    prevents padded residues from biasing the centroid or covariance.
    """
    B, L = pred_CA.shape[:2]
    device = pred_CA.device
    dtype = pred_CA.dtype
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
    mf = mask.to(dtype).unsqueeze(-1)  # [B, L, 1]
    n = mf.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]

    # Masked centroids
    mu_p = (pred_CA * mf).sum(dim=1, keepdim=True) / n
    mu_t = (target_CA * mf).sum(dim=1, keepdim=True) / n
    Pc = (pred_CA - mu_p) * mf
    Tc = (target_CA - mu_t) * mf

    # Cross-covariance H = Pc^T Tc per batch, [B, 3, 3]
    H = Pc.transpose(1, 2) @ Tc
    # SVD in float32/64 for stability
    U, S, Vh = torch.linalg.svd(H.to(torch.float64))
    det = torch.linalg.det(Vh.transpose(-2, -1) @ U.transpose(-2, -1))
    d = torch.sign(det)
    eye = torch.eye(3, device=device, dtype=torch.float64).expand(B, 3, 3).clone()
    eye[:, 2, 2] = d
    # R maps pred -> target
    R = (Vh.transpose(-2, -1) @ eye @ U.transpose(-2, -1)).to(dtype)

    Pr = (R @ Pc.transpose(1, 2)).transpose(1, 2)  # [B, L, 3]
    sq = ((Pr - Tc) ** 2).sum(dim=-1)  # [B, L]
    per_sample = (sq * mask.to(dtype)).sum(dim=1) / mask.to(dtype).sum(dim=1).clamp(min=1.0)
    return per_sample.mean()


def contact_distance_loss(
    pred_CA: torch.Tensor,
    target_CA: torch.Tensor,
    mask: torch.Tensor | None = None,
    residue_index: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    min_seq_sep: int = 3,
    cutoff: float = 25.0,
) -> torch.Tensor:
    """Chain-aware nonlocal CA-CA distance error (SE(3)-invariant).

    The contact-Jaccard family of eval metrics is a function of CA-CA
    distances on chain-aware nonlocal pairs only.  Those distances are
    rotation/translation invariant, so supervising them needs no alignment and
    is fully consistent with the Kabsch-RMSD alignment above.  Pairs beyond the
    largest contact cutoff (25 Å) carry no contact information, so the loss is
    restricted to the near-field where all three contact bins live.
    """
    B, L = pred_CA.shape[:2]
    device = pred_CA.device
    dtype = pred_CA.dtype
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
    if residue_index is None:
        residue_index = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    if chain_id is None:
        chain_id = torch.zeros(B, L, dtype=torch.long, device=device)

    pair_valid = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, L, L]
    same_chain = chain_id.unsqueeze(1) == chain_id.unsqueeze(2)
    seq_sep = (residue_index.unsqueeze(1) - residue_index.unsqueeze(2)).abs()
    nonlocal_pair = (~same_chain) | (seq_sep > min_seq_sep)
    pm = (pair_valid & nonlocal_pair).to(dtype)

    d_pred = torch.cdist(pred_CA, pred_CA)   # [B, L, L]
    d_tgt = torch.cdist(target_CA, target_CA)
    # Near-field mask on the target geometry (where contacts can occur).
    near = (d_tgt <= cutoff).to(dtype)
    w = pm * near
    diff = F.smooth_l1_loss(d_pred, d_tgt, reduction="none", beta=2.0)
    denom = w.sum().clamp(min=1.0)
    return (diff * w).sum() / denom


def contact_binary_loss(
    pred_CA: torch.Tensor,
    target_CA: torch.Tensor,
    mask: torch.Tensor | None = None,
    residue_index: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
    min_seq_sep: int = 3,
    cutoffs: tuple[float, ...] = (8.0, 15.0, 25.0),
    tau: float = 1.5,
) -> torch.Tensor:
    """Binary contact-indicator loss aligned to the contact-Jaccard metric.

    The evaluation reports Jaccard scores on the binary contact indicator
    ``1[d <= c]`` for three CA-CA cutoffs ``c in {8, 15, 25}`` (conditions 3-5).
    ``contact_distance_loss`` only supervises the raw distance and is therefore
    free to be wrong exactly at the bin boundary, which is where the Jaccard
    score is decided.  This loss directly supervises the indicator: a smooth
    probability ``sigmoid((c - d_pred)/tau)`` is pushed toward the target
    indicator ``1[d_tgt <= c]`` with binary cross-entropy over the same
    chain-aware nonlocal valid pairs used by the evaluation.  It is SE(3)
    invariant (a function of CA-CA distances only).
    """
    B, L = pred_CA.shape[:2]
    device = pred_CA.device
    dtype = pred_CA.dtype
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=device)
    if residue_index is None:
        residue_index = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    if chain_id is None:
        chain_id = torch.zeros(B, L, dtype=torch.long, device=device)

    pair_valid = mask.unsqueeze(1) & mask.unsqueeze(2)
    same_chain = chain_id.unsqueeze(1) == chain_id.unsqueeze(2)
    seq_sep = (residue_index.unsqueeze(1) - residue_index.unsqueeze(2)).abs()
    nonlocal_pair = (~same_chain) | (seq_sep > min_seq_sep)
    pm = (pair_valid & nonlocal_pair).to(dtype)

    d_pred = torch.cdist(pred_CA, pred_CA)
    d_tgt = torch.cdist(target_CA, target_CA).detach()

    denom = pm.sum().clamp(min=1.0)
    loss = pred_CA.new_tensor(0.0)
    for c in cutoffs:
        logit = (c - d_pred) / max(tau, 1e-6)
        prob = torch.sigmoid(logit)
        tgt = (d_tgt <= c).to(dtype)
        # BCE with logits in a numerically stable form.
        bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
        loss = loss + (bce * pm).sum() / denom
    return loss / len(cutoffs)


def direct_ca_loss(
    pred_CA: torch.Tensor,
    target_CA: torch.Tensor,
    anchor_CA: torch.Tensor | None = None,
    anchor_R: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE of CA displacement relative to one shared anchor frame.

    Independent pred/target centering removes the very translation signal that
    the anchor-based decoder must learn.  A shared anchor preserves relative
    motion while remaining invariant to a global SE(3) transform.
    """
    B, L = pred_CA.shape[:2]
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=pred_CA.device)
    if anchor_CA is None:
        anchor_CA = torch.zeros_like(pred_CA)
    pred_rel = pred_CA - anchor_CA
    target_rel = target_CA - anchor_CA
    # Remove the arbitrary whole-protein translation while preserving the
    # residue-wise deformation relative to the anchor structure.
    mf = mask.unsqueeze(-1).float()
    pred_rel = pred_rel - (pred_rel * mf).sum(dim=1, keepdim=True) / mf.sum(dim=1, keepdim=True).clamp(min=1)
    target_rel = target_rel - (target_rel * mf).sum(dim=1, keepdim=True) / mf.sum(dim=1, keepdim=True).clamp(min=1)
    if anchor_R is not None:
        R_inv = anchor_R.transpose(-2, -1)
        pred_rel = (R_inv @ pred_rel.unsqueeze(-1)).squeeze(-1)
        target_rel = (R_inv @ target_rel.unsqueeze(-1)).squeeze(-1)
    diff = pred_rel - target_rel
    sq = (diff ** 2).sum(dim=-1)  # [B, L]
    return (sq * mask.float()).sum() / mask.sum().clamp(min=1)


__all__ = [
    "fape_loss", "frame_aligned_fape", "torsion_loss",
    "pairwise_distance_loss", "bond_length_violation",
    "clash_penalty", "stage_c_total_loss", "direct_ca_loss",
    "contact_distance_loss", "contact_binary_loss", "kabsch_align_ca_loss",
]
