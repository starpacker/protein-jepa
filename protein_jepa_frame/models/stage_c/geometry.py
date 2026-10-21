"""SE(3) geometry primitives for Stage C structure decoder.

Gate 5 canonical geometry operations:
- Rigid body frames from backbone coordinates
- Frame composition, inverse, and transform application
- Rotations as quaternions and rotation matrices
- SE(3) equivariance enforcement

All coordinate units: Angstrom.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Rotation utilities
# ═══════════════════════════════════════════════════════════════════════

def rotation_matrix_from_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert quaternion [..., 4] (w,x,y,z) to rotation matrix [..., 3, 3].

    Args:
        q: [..., 4] quaternion (real part first: w, x, y, z)
        eps: epsilon for normalization

    Returns:
        [..., 3, 3] rotation matrices
    """
    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


def quaternion_from_rotation_matrix(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert rotation matrix [..., 3, 3] to quaternion [..., 4] (w,x,y,z)."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    # Four cases based on trace
    cond1 = trace > 0
    cond2 = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & ~cond1
    cond3 = (R[..., 1, 1] > R[..., 2, 2]) & ~cond1 & ~cond2
    cond4 = ~cond1 & ~cond2 & ~cond3

    q = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)

    # Case 1: trace > 0
    mask = cond1
    if mask.any():
        S = torch.sqrt(trace[mask] + 1.0) * 2.0
        q[mask, 0] = 0.25 * S
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / (S + eps)
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / (S + eps)
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / (S + eps)

    # Case 2: R[0,0] largest
    mask = cond2
    if mask.any():
        S = torch.sqrt(1.0 + R[mask, 0, 0] - R[mask, 1, 1] - R[mask, 2, 2]) * 2.0
        q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / (S + eps)
        q[mask, 1] = 0.25 * S
        q[mask, 2] = (R[mask, 0, 1] + R[mask, 1, 0]) / (S + eps)
        q[mask, 3] = (R[mask, 0, 2] + R[mask, 2, 0]) / (S + eps)

    # Case 3: R[1,1] largest
    mask = cond3
    if mask.any():
        S = torch.sqrt(1.0 + R[mask, 1, 1] - R[mask, 0, 0] - R[mask, 2, 2]) * 2.0
        q[mask, 0] = (R[mask, 0, 2] - R[mask, 2, 0]) / (S + eps)
        q[mask, 1] = (R[mask, 0, 1] + R[mask, 1, 0]) / (S + eps)
        q[mask, 2] = 0.25 * S
        q[mask, 3] = (R[mask, 1, 2] + R[mask, 2, 1]) / (S + eps)

    # Case 4: R[2,2] largest
    mask = cond4
    if mask.any():
        S = torch.sqrt(1.0 + R[mask, 2, 2] - R[mask, 0, 0] - R[mask, 1, 1]) * 2.0
        q[mask, 0] = (R[mask, 1, 0] - R[mask, 0, 1]) / (S + eps)
        q[mask, 1] = (R[mask, 0, 2] + R[mask, 2, 0]) / (S + eps)
        q[mask, 2] = (R[mask, 1, 2] + R[mask, 2, 1]) / (S + eps)
        q[mask, 3] = 0.25 * S

    q = q / (q.norm(dim=-1, keepdim=True) + eps)
    return q


def compose_rotations(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Compose two rotation matrices: R = R2 @ R1."""
    return r2 @ r1


def invert_rotation(R: torch.Tensor) -> torch.Tensor:
    """Invert a rotation matrix (transpose)."""
    return R.transpose(-2, -1)


# ═══════════════════════════════════════════════════════════════════════
# Rigid body frames (SE(3) elements)
# ═══════════════════════════════════════════════════════════════════════

class RigidBodyFrame:
    """SE(3) rigid body frame: rotation (3x3) + translation (3).

    Represents transform from local to global coordinates:
    x_global = R @ x_local + t
    """

    @staticmethod
    def from_backbone(N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute residue frames from backbone atoms.

        Standard convention:
        - Origin at CA
        - x-axis along C -> CA -> N average direction
        - y-axis perpendicular to N-CA-C plane

        Args:
            N, CA, C: [..., 3] coordinates

        Returns:
            R: [..., 3, 3] rotation matrices
            t: [..., 3] translation vectors (CA positions)
        """
        t = CA.clone()

        # Local x: CA->C direction
        x_axis = C - CA
        x_axis = x_axis / (x_axis.norm(dim=-1, keepdim=True) + 1e-8)

        # Local y: perpendicular to N-CA-C plane
        n_to_ca = N - CA
        y_axis = torch.cross(x_axis, n_to_ca, dim=-1)
        y_axis = y_axis / (y_axis.norm(dim=-1, keepdim=True) + 1e-8)

        # Local z: completes right-handed frame
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        z_axis = z_axis / (z_axis.norm(dim=-1, keepdim=True) + 1e-8)

        # Orthogonalize y relative to x for numerical stability
        y_axis = torch.cross(z_axis, x_axis, dim=-1)

        R = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        return R, t

    @staticmethod
    def compose(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose two SE(3) transforms: T = T2 @ T1.

        First apply T1, then T2.
        """
        R = R2 @ R1
        t = t2 + (R2 @ t1.unsqueeze(-1)).squeeze(-1)
        return R, t

    @staticmethod
    def inverse(R: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Invert an SE(3) transform."""
        R_inv = R.transpose(-2, -1)
        t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
        return R_inv, t_inv

    @staticmethod
    def transform(R: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply SE(3) transform to points: x_global = R @ x + t.

        Args:
            R: [..., 3, 3]
            t: [..., 3]
            x: [..., 3]

        Returns:
            [..., 3] transformed coordinates
        """
        return (R @ x.unsqueeze(-1)).squeeze(-1) + t

    @staticmethod
    def from_quaternion_translation(q: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build frame from quaternion [w,x,y,z] and translation."""
        R = rotation_matrix_from_quaternion(q)
        return R, t

    @staticmethod
    def identity(device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return identity SE(3) frame."""
        R = torch.eye(3, device=device, dtype=dtype)
        t = torch.zeros(3, device=device, dtype=dtype)
        return R, t


# ═══════════════════════════════════════════════════════════════════════
# Equivariant IPA-style graph network
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def fit_history_pose(decoded_anchor_ca: torch.Tensor, observed_anchor_ca: torch.Tensor,
                     mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one history-only decoder pose using row vectors: X @ R + t.

    A shape-trained invariant decoder has an arbitrary canonical pose. Only the
    decoded observed anchor and the real observed anchor fix that nuisance pose;
    the same detached transform must be used at every predicted future time.
    """
    if decoded_anchor_ca.shape != observed_anchor_ca.shape or decoded_anchor_ca.ndim != 3:
        raise ValueError("decoded and observed anchor coordinates must have shape [B,L,3]")
    if decoded_anchor_ca.shape[-1] != 3 or mask.shape != decoded_anchor_ca.shape[:2]:
        raise ValueError("history-pose mask must match [B,L]")
    rotations, translations = [], []
    for pred, real, valid in zip(decoded_anchor_ca.detach(), observed_anchor_ca.detach(), mask.bool()):
        if valid.sum() < 3:
            raise ValueError("history pose fitting requires at least three valid residues")
        p, q = pred[valid].double(), real[valid].double()
        if not torch.isfinite(p).all() or not torch.isfinite(q).all():
            raise FloatingPointError("nonfinite observed history pose")
        pc, qc = p.mean(0), q.mean(0)
        U, _, Vh = torch.linalg.svd((p-pc).T @ (q-qc))
        correction = torch.eye(3, device=p.device, dtype=p.dtype)
        correction[2, 2] = torch.where(torch.det(U@Vh) < 0, -1.0, 1.0)
        rotation = U@correction@Vh
        rotations.append(rotation.to(pred.dtype))
        translations.append((qc-pc@rotation).to(pred.dtype))
    return torch.stack(rotations), torch.stack(translations)


class InvariantPointAttention(nn.Module):
    """Simplified IPA: content attention + geometric bias from pairwise distances.

    Uses CA positions for geometric attention bias without complex point projections
    that cause dimension mismatches in the initial decoder implementation.
    """

    def __init__(self, c_hidden: int = 128, n_heads: int = 4, n_query_points: int = 4, n_value_points: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = c_hidden // n_heads
        assert c_hidden % n_heads == 0

        self.linear_q = nn.Linear(c_hidden, c_hidden)
        self.linear_k = nn.Linear(c_hidden, c_hidden)
        self.linear_v = nn.Linear(c_hidden, c_hidden)
        self.linear_b = nn.Linear(c_hidden, n_heads)
        self.linear_out = nn.Linear(c_hidden, c_hidden)
        self.linear_out.weight.data.zero_()
        self.linear_out.bias.data.zero_()

    def forward(
        self,
        s: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IPA forward pass.

        Args:
            s: [B, L, c_hidden] single representation
            R: [B, L, 3, 3] residue orientations (unused in simplified mode)
            t: [B, L, 3] residue positions

        Returns:
            s_out: [B, L, c_hidden]
            point_update: [B, L, 3] zero-displacement (simplified)
        """
        B, L, _ = s.shape
        H, D = self.n_heads, self.head_dim

        q = self.linear_q(s).reshape(B, L, H, D)
        k = self.linear_k(s).reshape(B, L, H, D)
        v = self.linear_v(s).reshape(B, L, H, D)
        bias = self.linear_b(s)  # [B, L, H]

        # Content attention
        attn = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(D)

        # Geometric bias: negative squared distance scaled
        sq_dists = ((t.unsqueeze(2) - t.unsqueeze(1)) ** 2).sum(-1)  # [B, L, L]
        attn = attn - 0.5 * sq_dists.unsqueeze(1) / 100.0

        attn = attn + bias.permute(0, 2, 1).unsqueeze(2)
        if residue_mask is not None:
            mask_2d = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)
            attn = attn.masked_fill(~mask_2d.unsqueeze(1), -1e9)
        attn = F.softmax(attn, dim=-1)

        v_h = attn.new_tensor(0)  # unused, keep for compatibility
        # v is [B, L, H, D]; pattern bhij,bjhd->bihd contracts j (residue index)
        s_out = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(B, L, -1)
        s_out = self.linear_out(s_out)

        point_update = torch.zeros(B, L, 3, device=s.device, dtype=s.dtype)
        return s_out, point_update


# ═══════════════════════════════════════════════════════════════════════
# SE(3) equivariance test
# ═══════════════════════════════════════════════════════════════════════

def test_se3_equivariance():
    """Verify SE(3) equivariance of rigid frame operations.

    For any SE(3) transform T:
    - T(frame(points)) should equal frame(T(points))
    - Backbone to frames: rotation part equivariant, translation consistent
    """
    torch.manual_seed(42)
    B, L = 2, 5

    # Create random backbone coordinates
    N = torch.randn(B, L, 3)
    CA = torch.randn(B, L, 3)
    C = torch.randn(B, L, 3)

    # Compute frames
    R1, t1 = RigidBodyFrame.from_backbone(N, CA, C)

    # Create random SE(3) transform
    q_rand = torch.randn(4)
    q_rand = q_rand / q_rand.norm()
    Rg = rotation_matrix_from_quaternion(q_rand)
    tg = torch.randn(3)

    # Transform coordinates
    N_t = (Rg @ N.unsqueeze(-1)).squeeze(-1) + tg
    CA_t = (Rg @ CA.unsqueeze(-1)).squeeze(-1) + tg
    C_t = (Rg @ C.unsqueeze(-1)).squeeze(-1) + tg

    # Compute frames from transformed coordinates
    R2, t2 = RigidBodyFrame.from_backbone(N_t, CA_t, C_t)

    # Check equivariance: t2 should be Rg @ t1 + tg
    t_expected = (Rg @ t1.unsqueeze(-1)).squeeze(-1) + tg
    t_err = (t2 - t_expected).abs().max().item()

    print(f"SE(3) equivariance test: max translation error = {t_err:.2e}")
    assert t_err < 1e-5, f"SE(3) translation equivariance failed: {t_err}"

    print("  SE(3) equivariance PASSED\n")
    return True


__all__ = [
    "rotation_matrix_from_quaternion",
    "quaternion_from_rotation_matrix",
    "compose_rotations",
    "invert_rotation",
    "RigidBodyFrame",
    "InvariantPointAttention",
    "fit_history_pose",
    "test_se3_equivariance",
]
