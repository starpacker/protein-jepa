"""Correctness tests for the from-none Stage B components."""
import math
import torch

from protein_jepa_frame.models.stage_b1b_fromnone.decoder import FromNoneConfig, SimpleCADecoder
from protein_jepa_frame.models.stage_b1b_fromnone.objectives import (
    aligned_ca_rmsd, energy_score, kabsch_rotation,
)


def _rand_rotation(dtype=torch.float64):
    A = torch.randn(3, 3, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def test_zero_dx_is_persistence():
    """With head weights zeroed the decoder must reproduce the anchor exactly."""
    cfg = FromNoneConfig(latent_dim=8, bath_dim=0, hidden=16, depth=2)
    dec = SimpleCADecoder(cfg)
    for p in dec.parameters():
        p.data.zero_()
    B, L = 2, 5
    Hf, Hc = torch.randn(B, L, 8), torch.randn(B, L, 8)
    Xc = torch.randn(B, L, 3)
    R = torch.eye(3).expand(B, L, 3, 3).clone()
    m = torch.ones(B, L, dtype=torch.bool)
    out = dec(Hf, Hc, Xc, m, R)
    assert torch.allclose(out["ca_pred"], Xc, atol=1e-6)
    assert torch.allclose(out["dx"], torch.zeros_like(out["dx"]), atol=1e-6)


def test_decoder_equivariance():
    """A global rotation of (X_anchor, R_anchor) must rotate ca_pred identically."""
    torch.manual_seed(0)
    cfg = FromNoneConfig(latent_dim=8, bath_dim=0, hidden=16, depth=2)
    dec = SimpleCADecoder(cfg)
    B, L = 1, 6
    Hf, Hc = torch.randn(B, L, 8), torch.randn(B, L, 8)
    Xc = torch.randn(B, L, 3)
    R = torch.eye(3).expand(B, L, 3, 3).contiguous()
    m = torch.ones(B, L, dtype=torch.bool)
    Q = _rand_rotation().float()
    Xc_rot = Xc @ Q
    R_rot = Q.T @ R  # rotate the local frames' global embedding
    out = dec(Hf, Hc, Xc, m, R)
    out_rot = dec(Hf, Hc, Xc_rot, m, R_rot)
    # predicted global displacement should rotate with Q
    assert torch.allclose(out_rot["ca_pred"], out["ca_pred"] @ Q, atol=1e-5)


def test_kabsch_rotation_recovers_known():
    torch.manual_seed(1)
    X = torch.randn(1, 20, 3, dtype=torch.float64)
    Q = _rand_rotation()
    t = torch.randn(1, 1, 3, dtype=torch.float64)
    Y = X @ Q + t
    R, tt = kabsch_rotation(X, Y)
    aligned = X @ R + tt
    assert torch.allclose(aligned, Y, atol=1e-6)


def test_batched_aligned_rmsd_matches_manual():
    torch.manual_seed(2)
    B, L = 3, 10
    pred = torch.randn(B, L, 3)
    true = torch.randn(B, L, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    got = aligned_ca_rmsd(pred, true, mask)
    # manual reference per batch element
    refs = []
    for b in range(B):
        p, q = pred[b], true[b]
        R, t = kabsch_rotation(p[None], q[None])
        refs.append(((p @ R[0] + t[0]) - q).square().sum(-1).mean().sqrt())
    ref = torch.stack(refs).mean()
    assert torch.allclose(got, ref, atol=1e-4), (got.item(), ref.item())


def test_masked_residues_do_not_bias_rmsd():
    """Padding residues (mask=0) with garbage must not change the RMSD."""
    torch.manual_seed(3)
    B, L = 2, 12
    pred = torch.randn(B, L, 3)
    true = torch.randn(B, L, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 6:] = False
    r1 = aligned_ca_rmsd(pred, true, mask)
    pred2 = pred.clone(); true2 = true.clone()
    pred2[0, 6:] = 1e3; true2[0, 6:] = -1e3
    r2 = aligned_ca_rmsd(pred2, true2, mask)
    assert torch.allclose(r1, r2, atol=1e-5), (r1.item(), r2.item())


def test_energy_score_proper():
    """ES is minimized (on average) by the true distribution; here check the
    spread penalty and identity: samples centred on target beat spread-out ones."""
    target = torch.zeros(10)
    tight = torch.randn(64, 10) * 0.01
    wide = torch.randn(64, 10) * 5.0
    assert energy_score(tight, target) < energy_score(wide, target)


def test_split_batch_preserves_per_frame_keys():
    """aatype/residue_index must NOT be frame-sliced even when L == clip length."""
    import torch as _t
    from protein_jepa_frame.training.train_stage_cb1b import split_batch
    B, clip, L = 2, 24, 24
    b = {
        "coords": _t.randn(B, clip, L, 4, 3),
        "residue_mask": _t.ones(B, clip, L, dtype=_t.bool),
        "aatype": _t.zeros(B, L, dtype=_t.long),
        "residue_index": _t.arange(L).expand(B, L),
        "chain_id": _t.zeros(B, L, dtype=_t.long),
        "time_ps": _t.arange(clip).float().expand(B, clip) * 1000,
    }
    h, tgt, m, dt, b2 = split_batch(b, 8, 16, 1000.0, clip)
    assert h.shape == (B, 8, L, 4, 3)
    assert tgt.shape == (B, 17, L, 4, 3)
    assert b2["aatype"].shape == (B, L)
    assert b2["residue_mask"].shape == (B, 24, L)
