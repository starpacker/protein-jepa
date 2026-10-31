"""Differentiable anchor-length constraints for CA component experiments."""
import torch


def project_ca_bonds(pred, anchor, mask, residue_index, chain_id, iterations=16):
    """Symmetric local constraint iterations; preserve gaps, padding and SE(3).

    Uses observed anchor bond lengths, never future target lengths. This is a CA
    probe constraint, not a full peptide or all-atom geometry validation.
    """
    valid=(mask[:,1:] & mask[:,:-1] & (chain_id[:,1:]==chain_id[:,:-1])
           & ((residue_index[:,1:]-residue_index[:,:-1])==1))
    lengths=(anchor[:,1:]-anchor[:,:-1]).norm(dim=-1)
    result=pred
    for _ in range(iterations):
        edge=result[:,1:]-result[:,:-1]
        distance=edge.norm(dim=-1).clamp_min(1e-8)
        correction=.25*(1-lengths/distance)[...,None]*edge*valid[...,None]
        shift=torch.zeros_like(result)
        shift[:,:-1]=correction
        shift[:,1:]=shift[:,1:]-correction
        result=result+shift
    return result


def retract_ca_bond_lengths(pred, anchor, mask, residue_index, chain_id):
    """Set anchor bond lengths, preserving input edge directions and segment means.

    Each contiguous valid chain segment is reconstructed from its normalized
    edges, then translated to its input centroid. Consequently bond angles and
    torsions are unchanged for nondegenerate edges. This is opt-in, not the
    nearest Cartesian projection or an all-atom validity guarantee. Singular
    edges fail closed; derivatives scale with inverse input edge length.
    """
    valid = (mask[:, 1:] & mask[:, :-1]
             & (chain_id[:, 1:] == chain_id[:, :-1])
             & (residue_index[:, 1:] - residue_index[:, :-1] == 1))
    edges = pred[:, 1:] - pred[:, :-1]
    lengths = edges.norm(dim=-1)
    target = (anchor[:, 1:] - anchor[:, :-1]).norm(dim=-1)
    if not bool(torch.isfinite(pred).all() and torch.isfinite(anchor).all()):
        raise ValueError('nonfinite retraction coordinates')
    if bool(((lengths <= 1e-6) | (target <= 1e-6))[valid].any()):
        raise ValueError('singular adjacent edge in bond retraction')
    scaled = edges * (target / lengths.clamp_min(1e-6))[..., None]
    batches = []
    for b in range(pred.shape[0]):
        segments = []
        start = 0
        boundaries = (~valid[b]).nonzero().flatten().tolist() + [pred.shape[1]-1]
        for end in boundaries:
            original = pred[b, start:end+1]
            relative = torch.cat([torch.zeros_like(original[:1]),
                                  scaled[b, start:end].cumsum(0)], dim=0)
            reconstructed = relative - relative.mean(0) + original.mean(0)
            segments.append(reconstructed if end > start else original)
            start = end+1
        batches.append(torch.cat(segments, dim=0))
    return torch.stack(batches)
