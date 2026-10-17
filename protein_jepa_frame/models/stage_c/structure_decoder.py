"""Stage C structure decoder: equivariant frame/torsion decoder.

Gate 5 canonical architecture:
- SE(3)-equivariant IPA backbone with residue frames
- Per-residue rigid body updates + torsion angle prediction
- Atom37 reconstruction via the OFFICIAL OpenFold geometry package
  (``torsion_angles_to_frames`` + ``frames_and_literature_positions_to_atom14_pos``
  + ``atom14_to_atom37``); no hand-written atom placement
- Single shared decoder for both direct_endpoint and autonomous_recursive
- FAPE, torsion, distance, bond, angle, clash, chirality losses

Architecture invariant (from audit plan Gate 5):
- No future H or coordinates as input (only frozen h_i(t0) from last observation)
- Input: sequence, aatype, frozen context H, dynamics Q_t, last-observed frames
- Anchor-based design: predicts delta from last observed structure
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_jepa_frame.models.stage_c.geometry import (
    RigidBodyFrame,
    InvariantPointAttention,
    rotation_matrix_from_quaternion,
)


# ═══════════════════════════════════════════════════════════════════════
# Ideal bond geometry constants
#
# NOTE: frame -> atom37 reconstruction does NOT use these tables.  That path
# is delegated entirely to the official OpenFold package (see
# ``openfold_atom37.py``).  The tables below are retained only for the
# diagnostic torsion-chain-extension path (``reconstruct_backbone_from_torsions``)
# and legacy callers; they are NOT used to place sidechain atoms.
# ═══════════════════════════════════════════════════════════════════════

IDEAL_BOND = {
    "N_CA": 1.458,  "CA_C": 1.525,  "C_N": 1.329,  "C_O": 1.231,
    "CA_CB": 1.525,  "CB_CG": 1.525,  "CB_CG1": 1.525,  "CB_CG2": 1.525,
    "CG_CD": 1.525,  "CG_CD1": 1.525,  "CG_CD2": 1.525,
    "CD_NE": 1.460,  "CD_CE": 1.525,
    "CD1_CG": 1.378,  "CD2_CG": 1.378,  "CE1_CZ": 1.378,  "CE2_CZ": 1.378,
    "NE_CZ": 1.460,  "NE_CD": 1.460,
    "CE_NZ": 1.460,  "CZ_NH1": 1.330,  "CZ_NH2": 1.330,
    "CG_OD1": 1.208,  "CG_OD2": 1.208,  "CD_OE1": 1.208,  "CD_OE2": 1.208,
    "CB_OG": 1.430,  "CB_OG1": 1.430,  "CB_SG": 1.808,
    "CG_ND1": 1.378,  "CG_ND2": 1.330,
    "CD1_NE1": 1.378,  "CD2_NE2": 1.378,
    "CE1_NE2": 1.378,  "CE2_NE1": 1.378,
    "CZ_OH": 1.360,
    "CE_NZ2": 1.460,  "CZ2_CH2": 1.378,  "CZ3_CH2": 1.378,
    "SD_CE": 1.810,  "OG1_CG2": 1.430,
    "C_OXT": 1.231,
}

IDEAL_ANGLE = {
    "N_CA_C": math.radians(111.0), "CA_C_N": math.radians(116.2),
    "C_N_CA": math.radians(121.7), "CA_C_O": math.radians(120.5),
    "N_CA_CB": math.radians(109.5), "C_CA_CB": math.radians(109.5),
    "CA_CB_CG": math.radians(113.5), "CA_CB_CG1": math.radians(109.5),
    "CA_CB_CG2": math.radians(109.5), "CA_CB_OG": math.radians(109.5),
    "CA_CB_OG1": math.radians(109.5), "CA_CB_SG": math.radians(109.5),
    "CB_CG_CD": math.radians(113.5), "CB_CG_ND1": math.radians(109.5),
    "CB_CG_OD1": math.radians(120.5), "CB_CG_OD2": math.radians(120.5),
    "CG_CD_NE": math.radians(109.5), "CG_CD_OE1": math.radians(120.5),
    "CG_CD_OE2": math.radians(120.5),
    "ND1_CG_CD2": math.radians(107.0), "CD2_CG_ND1": math.radians(107.0),
    "CG_ND1_CE1": math.radians(107.0), "CG_CD2_NE2": math.radians(107.0),
    "ND1_CE1_NE2": math.radians(109.5), "CD2_NE2_CE1": math.radians(109.5),
    "NE2_CE1_ND1": math.radians(109.5),
    "CD_NE_CZ": math.radians(109.5), "NE_CZ_NH1": math.radians(120.0),
    "NE_CZ_NH2": math.radians(120.0),
    "CB_CG_CD1": math.radians(120.0), "CB_CG_CD2": math.radians(120.0),
    "CG_CD1_CE1": math.radians(120.0), "CG_CD2_CE2": math.radians(120.0),
    "CD1_CE1_CZ": math.radians(120.0), "CD2_CE2_CZ": math.radians(120.0),
    "CE1_CZ_CE2": math.radians(120.0), "CE1_CZ_OH": math.radians(120.0),
    "CE2_CZ_OH": math.radians(120.0),
    "CE_NZ_CE": math.radians(109.5),
    "CB_CG1_CD1": math.radians(109.5), "CG1_CD1_NE1": math.radians(109.5),
    "CD1_NE1_CE2": math.radians(109.5), "NE1_CE2_CZ2": math.radians(109.5),
    "CE2_CZ2_CH2": math.radians(120.0), "CZ2_CH2_CZ3": math.radians(120.0),
    "CH2_CZ3_CE3": math.radians(120.0), "CZ3_CE3_CD2": math.radians(120.0),
    "NE2_CE2_CZ3": math.radians(109.5), "CE2_CZ3_CH2": math.radians(120.0),
}

NUM_TORSIONS = 7
ATOM37_DIM = 37


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for structure decoder."""
    c_hidden: int = 128          # hidden dimension for IPA
    c_ipa: int = 64              # IPA hidden dimension
    n_heads: int = 4             # IPA attention heads
    n_ipa_blocks: int = 4        # number of IPA blocks
    n_query_points: int = 4      # IPA query points
    n_value_points: int = 4      # IPA value points
    torsion_bins: int = 36       # bins for torsion angle prediction (10-degree bins)
    freeze_ab: bool = True       # freeze Stage A/B encoders (must be True initially)
    # State conditioning.  The decoder has ONE state pathway (``q_dynamics``),
    # which carries the dynamics state Q := H_t for the mainline D_chi and
    # (optionally) the bath state S_t appended for the D_chi' comparison
    # (see stage_b2/bath_aware_decoder.py).  ``q_dim`` is the width of that
    # pathway on the mainline and must equal dH.
    q_dim: int = 128
    # Extra bath-state width fed through the SAME conditioning path when > 0
    # (D_chi' only).  Kept a separate knob so the mainline q_dim == dH is
    # unambiguous and the D_chi' comparison is capacity-matched on H.
    bath_extra_dim: int = 0
    dropout: float = 0.1
    # Anchor-local displacement budget: the ONLY positional head is
    #
    #     X_hat = anchor_frame  x  d_local(Q:=H_t, c [, S])
    #
    # with d_local bounded component-wise to +/- max_local_position_update (A).
    # anchor (R_anchor, t_anchor) is the equivariant rigid-body basis and
    # supplies absolute placement only; all internal conformation comes from
    # the invariant state (H_t, c, [S]).  Legacy heads (direct_h_position,
    # q_translation_update, basin_reversion, COM) have been removed.
    max_local_position_update: float = 12.0

    # Anchor-free positional decoding.
    #
    # When True the decoder no longer *starts* from the anchor structure and
    # adds a bounded residual.  Instead H alone predicts the full internal CA
    # cloud (expressed in the anchor-local rigid frame) BEFORE the IPA trunk
    # runs, the IPA geometric bias / contact graph are built from those
    # H-predicted positions, and the anchor (R_anchor, t_anchor) is applied only
    # at the very end as the equivariant placement
    #
    #     X_hat = R_anchor @ x_internal(H) + t_anchor .
    #
    # Consequences:
    #   * H = 0 => x_internal is the head's constant output => a single "mean
    #     fold", i.e. the best the decoder can do without H.  Real H must beat
    #     it, so the diagnostic real-H vs zero-H gap is no longer structurally
    #     zero.
    #   * The anchor can no longer satisfy the FAPE / distance losses by itself
    #     (it enters only through a fixed rigid transform, which the SE(3)
    #     quotient target is invariant to), so the positional gradient reaches H.
    #
    # coord_scale bounds the predicted internal radius (Å); the head output is
    # multiplied by it before placement.
    anchor_free: bool = False
    coord_scale: float = 30.0

    # Global-context pooling over the state pathway.  When True, pool the
    # masked mean over residues of the state and add its projection to every
    # residue in the single representation, giving each residue a protein-global
    # summary (used only by the invariant pathway, not by anchor).
    global_h_context: bool = False


def reconstruct_atom37(
    N: torch.Tensor,
    CA: torch.Tensor,
    C: torch.Tensor,
    O: torch.Tensor,
    aatype: torch.Tensor,
    torsions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct full atom37 coordinates from backbone + predicted torsions.

    Delegates ENTIRELY to the official OpenFold geometry pipeline (vendored,
    aqlaboratory/openfold).  No hand-written ideal bond/angle tables, no manual
    sidechain trees, no NEAT placement.

    The backbone is converted into OpenFold's canonical residue frames via
    ``Rigid.make_transform_from_reference`` (Algorithm 21), sidechain frames are
    built with ``openfold.utils.feats.torsion_angles_to_frames`` and all heavy
    atoms are placed with
    ``frames_and_literature_positions_to_atom14_pos`` + ``atom14_to_atom37``.

    The O atom is derived from the official frame geometry (it is not a separate
    rigid group in the OpenFold frame algebra), so the primary backbone frame is
    built from N, CA, C exactly as OpenFold does.

    Args:
        N:    [B, L, 3]
        CA:   [B, L, 3]
        C:    [B, L, 3]
        O:    [B, L, 3]  (unused for frame construction; retained for API compat)
        aatype: [B, L] (0-19, OpenFold restype ordering)
        torsions: [B, L, 7]  [phi, psi, omega, chi1, chi2, chi3, chi4]

    Returns:
        atom37_coords: [B, L, 37, 3] in OpenFold atom37 ordering.
        atom37_mask:   [B, L, 37] bool presence mask (OpenFold ordering).
    """
    from protein_jepa_frame.models.stage_c.openfold_atom37 import (
        frames_to_atom37,
        backbone_frame_from_coords,
    )

    rigid_backbone = backbone_frame_from_coords(N, CA, C)
    atom37_coords, atom37_mask = frames_to_atom37(
        rigid_backbone.get_rots().get_rot_mats(),
        rigid_backbone.get_trans(),
        aatype,
        torsions,
    )
    return atom37_coords, atom37_mask


def reconstruct_atom37_from_frames(
    R: torch.Tensor,
    t: torch.Tensor,
    aatype: torch.Tensor,
    torsions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame -> atom37 for the decoder's native frame convention.

    Thin delegation to the official OpenFold adapter; the decoder's ``(R, t)``
    frames (``RigidBodyFrame.from_backbone`` convention) are converted to
    OpenFold's residue-frame convention and all heavy atoms are placed by
    OpenFold's frame algebra.

    Args:
        R: [B, L, 3, 3] predicted frame orientations.
        t: [B, L, 3] predicted frame translations (CA positions).
        aatype: [B, L] residue type indices (OpenFold ordering).
        torsions: [B, L, 7] (phi, psi, omega, chi1..chi4).

    Returns:
        atom37_coords: [B, L, 37, 3] in OpenFold atom37 ordering.
        atom37_mask:   [B, L, 37] bool presence mask (OpenFold ordering).
    """
    from protein_jepa_frame.models.stage_c.openfold_atom37 import (
        decoder_frames_to_atom37,
    )

    return decoder_frames_to_atom37(R, t, aatype, torsions)


# ═══════════════════════════════════════════════════════════════════════
# Torsion angle reconstruction
# ═══════════════════════════════════════════════════════════════════════

def reconstruct_backbone_from_torsions(
    R_anchor: torch.Tensor,
    t_anchor: torch.Tensor,
    torsions: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Reconstruct backbone N, CA, C, O from residue frames + predicted torsions.

    Uses phi ([:,:,0]), psi ([:,:,1]), omega ([:,:,2]) to chain-extend the
    backbone atom-by-atom, so torsion losses propagate through coordinates.

    Residue 0 uses ideal geometry from the anchor frame.
    """
    B, L = R_anchor.shape[:2]
    device, dtype = R_anchor.device, R_anchor.dtype
    cos69, sin69 = math.cos(math.radians(69.)), math.sin(math.radians(69.))

    N = torch.zeros(B, L, 3, device=device, dtype=dtype)
    CA = torch.zeros(B, L, 3, device=device, dtype=dtype)
    C = torch.zeros(B, L, 3, device=device, dtype=dtype)
    O = torch.zeros(B, L, 3, device=device, dtype=dtype)

    # ── Residue 0 from anchor frame ──
    CA[:, 0] = t_anchor[:, 0]
    n_loc = torch.tensor([-cos69, 0., sin69], device=device, dtype=dtype)
    c_loc = torch.tensor([cos69, 0., sin69], device=device, dtype=dtype)
    o_loc = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
    N[:, 0] = CA[:, 0] + IDEAL_BOND["N_CA"] * (R_anchor[:, 0] @ n_loc)
    C[:, 0] = CA[:, 0] + IDEAL_BOND["CA_C"] * (R_anchor[:, 0] @ c_loc)
    O[:, 0] = C[:, 0] + IDEAL_BOND["C_O"] * (R_anchor[:, 0] @ o_loc)

    # ── Chain extension ──
    for i in range(1, L):
        omega = torsions[:, i-1, 2]
        phi   = torsions[:, i, 0]
        psi   = torsions[:, i, 1]

        # N_i from C_{i-1}: use CA_{i-1}→C_{i-1} as reference axis
        axis_ca_c = _unit(C[:, i-1] - CA[:, i-1])
        n_ref = torch.tensor([-cos69, 0., sin69], device=device, dtype=dtype)
        n_dir = _rodrigues(axis_ca_c, _reject(n_ref, axis_ca_c), omega,
                           IDEAL_ANGLE["C_N_CA"])
        N[:, i] = C[:, i-1] + IDEAL_BOND["C_N"] * n_dir

        # CA_i from N_i
        axis_n_c = _unit(N[:, i] - C[:, i-1])
        ca_dir = _rodrigues(axis_n_c, _reject(n_ref, axis_n_c), phi,
                            IDEAL_ANGLE["C_N_CA"])
        CA[:, i] = N[:, i] + IDEAL_BOND["N_CA"] * ca_dir

        # C_i from CA_i
        axis_n_ca = _unit(CA[:, i] - N[:, i])
        c_dir = _rodrigues(axis_n_ca, _reject(c_loc, axis_n_ca), psi,
                           IDEAL_ANGLE["N_CA_C"])
        C[:, i] = CA[:, i] + IDEAL_BOND["CA_C"] * c_dir

        # O_i from C_i (planar, non-rotatable)
        axis_ca_c2 = _unit(C[:, i] - CA[:, i])
        o_ref = torch.tensor([0., 1., 0.], device=device, dtype=dtype)
        o_dir = _reject(o_ref, axis_ca_c2)
        cos_aca, sin_aca = math.cos(IDEAL_ANGLE["CA_C_O"]), math.sin(IDEAL_ANGLE["CA_C_O"])
        o_final = cos_aca * axis_ca_c2 + sin_aca * _unit(o_dir)
        O[:, i] = C[:, i] + IDEAL_BOND["C_O"] * o_final

    return N, CA, C, O


def _unit(v: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Unit vector with a smooth, finite gradient at degenerate geometry."""
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    denom = norm.clamp_min(eps)
    unit = v / denom
    # A zero/near-zero direction has no meaningful orientation.  Return a
    # zero fallback and, importantly, do not expose the 1/eps derivative of
    # v / clamp(norm) to the optimizer.
    return torch.where(norm > eps, unit, torch.zeros_like(v))


def _reject(v: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Project v onto plane perpendicular to axis, return unit vector."""
    proj = v - (v * axis).sum(dim=-1, keepdim=True) * axis
    return _unit(proj)


def _rodrigues(
    axis: torch.Tensor,       # [B] + [3] unit axis
    perp: torch.Tensor,        # [B] + [3] unit perpendicular direction
    angle: torch.Tensor,       # [B] rotation angle
    default_angle: float,
) -> torch.Tensor:
    """Rotate perp around axis by angle (or default_angle if tensor)."""
    ang = angle
    cos_a, sin_a = torch.cos(ang), torch.sin(ang)
    cross = torch.linalg.cross(axis, perp, dim=-1)
    dot = (axis * perp).sum(dim=-1, keepdim=True)
    result = (perp * cos_a.unsqueeze(-1) +
              cross * sin_a.unsqueeze(-1) +
              axis * dot * (1.0 - cos_a.unsqueeze(-1)))
    return _unit(result)



# ═══════════════════════════════════════════════════════════════════════
# Structure decoder modules
# ═══════════════════════════════════════════════════════════════════════

class FrameUpdate(nn.Module):
    """Predicts the per-residue frame *orientation* update (quaternion).

    Translation is not produced here — it is owned solely by the decoder's
    anchor-local displacement head.
    """

    def __init__(self, c_hidden: int = 128):
        super().__init__()
        # Orientation update only.  Translation is owned exclusively by the
        # anchor-local displacement head (see StructureDecoder); there is no
        # separate frame-translation residual.
        self.quat_net = nn.Sequential(
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, 4),
        )
        # Initialize quaternion to identity (w=1) with small weight
        nn.init.normal_(self.quat_net[-1].weight, std=1e-3)
        self.quat_net[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Predict the frame orientation update.

        Args:
            s: [B, L, c_hidden]

        Returns:
            dq: [B, L, 4] quaternion update (w,x,y,z), unit-norm.
        """
        dq = self.quat_net(s)
        dq = dq / (dq.norm(dim=-1, keepdim=True) + 1e-8)
        return dq


class TorsionHead(nn.Module):
    """Predicts per-residue torsion angles via binned classification.

    Ramachandran-aware bias initialisation: the final linear layer's bias is
    set so that the initial (uniform-input) expected angle for phi lands in
    [-90°, -40°] (alpha-helix) and for psi lands in [-60°, 0°] — reasonable
    protein torsion values rather than 0° which produces broken backbone.
    """

    def __init__(self, c_hidden: int = 128, num_torsions: int = 7, n_bins: int = 36):
        super().__init__()
        self.num_torsions = num_torsions
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, num_torsions * n_bins),
        )
        self._init_ramachandran_bias()

    def _init_ramachandran_bias(self):
        """Initialise the final bias so a zero-mean input gives
        Ramachandran-plausible phi/psi expectations.

        Without this, uniform logits → expected angle = 0° (bin centre),
        producing a broken backbone that diverges linearly along the chain.
        """
        import math
        n_bins = self.n_bins
        bin_width = 2 * math.pi / n_bins
        bin_centres = torch.linspace(-math.pi + bin_width / 2,
                                      math.pi - bin_width / 2, n_bins)

        bias = self.net[2].bias      # [num_torsions * n_bins]
        bias_2d = bias.data.reshape(self.num_torsions, n_bins)

        # phi (index 0): prefer ~ -65° with std ~30°
        # psi (index 1): prefer ~ -45° with std ~30°
        # omega (index 2): prefer ~ 180° (±π) with std ~5° (strong trans preference)
        # chi1-4 (indices 3-6): uniform (no preference for sidechain rotamers)
        _ramachandran_preferences: dict[int, tuple[float, float]] = {
            0: (-65.0, 30.0),   # phi → alpha-helix
            1: (-45.0, 30.0),   # psi → alpha-helix
            2: (180.0, 60.0),   # omega → trans (loose, avoid -612 bias)
        }

        for idx, (target_deg, std_deg) in _ramachandran_preferences.items():
            target_rad = math.radians(target_deg)
            std_rad = math.radians(std_deg)
            # Gaussian log-prob: log p = -0.5 * ((x - mu) / sigma)^2
            # Bias[i] = -(angle_i - target)^2 / (2 * sigma^2)
            angular_dist2 = (bin_centres - target_rad) ** 2
            # Use circular distance for omega (180° → -π/+π are equivalent)
            if idx == 2:
                angular_dist2 = torch.min(
                    angular_dist2,
                    (bin_centres - target_rad + 2 * math.pi) ** 2,
                )
                angular_dist2 = torch.min(
                    angular_dist2,
                    (bin_centres - target_rad - 2 * math.pi) ** 2,
                )
            bias_2d[idx] = -angular_dist2 / (2.0 * std_rad ** 2)

        # Clamp extreme bias values to [-5, 5] so logits stay in
        # numerically stable range. The omega prior (180°±5°) was
        # producing -612 bias for far bins — far beyond this range.
        bias_2d = bias_2d.clamp(-5.0, 5.0)

        # Copy back
        bias.data.copy_(bias_2d.reshape(-1))

        # Small final weights: initial logits ≈ bias (Ramachandran prior, CE≈4).
        # The IPA single representation has norm ≈185, so weight std must be
        # scaled to keep logits numerically stable while allowing gradient flow.
        nn.init.normal_(self.net[0].weight, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, std=0.005)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Predict torsion angle distributions.

        Args:
            s: [B, L, c_hidden]

        Returns:
            [B, L, num_torsions, n_bins] logits
        """
        B, L, _ = s.shape
        logits = self.net(s).reshape(B, L, self.num_torsions, self.n_bins)
        return logits


class ContactGraphBuilder(nn.Module):
    """Builds frozen contact graph from last observed structure."""

    def __init__(self, cutoff: float = 8.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(
        self,
        CA_coords: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build binary contact map.

        Args:
            CA_coords: [B, L, 3]
            residue_mask: [B, L] optional mask

        Returns:
            [B, L, L] binary contact matrix (1=contact)
        """
        dists = torch.cdist(CA_coords, CA_coords)  # [B, L, L]
        contacts = (dists < self.cutoff).float()

        # No self-contacts
        eye = torch.eye(CA_coords.shape[1], device=CA_coords.device)
        contacts = contacts * (1 - eye.unsqueeze(0))

        if residue_mask is not None:
            mask_2d = residue_mask.unsqueeze(2) & residue_mask.unsqueeze(1)
            contacts = contacts * mask_2d.float()

        return contacts


# ═══════════════════════════════════════════════════════════════════════
# Main Structure Decoder
# ═══════════════════════════════════════════════════════════════════════

class StructureDecoder(nn.Module):
    """Canonical Stage C structure decoder.

    Anchored on last observed structure. Predicts frame deltas and torsions
    from frozen context H + dynamics Q_t.

    Architecture:
    1. Embed sequence + aatype + frozen H + Q_t
    2. N IPA blocks producing single representations
    3. FrameUpdate head → per-residue frame updates
    4. TorsionHead → phi/psi/omega/chi1-4 classification
    5. Atom37 reconstruction via ideal bond geometry
    """

    def __init__(self, config: DecoderConfig | None = None):
        super().__init__()
        self.config = config or DecoderConfig()
        c = self.config

        # Input projections.  One state pathway only: q_dynamics carries the
        # dynamics state Q := H_t on the mainline, optionally with the bath
        # state S_t appended on the D_chi' comparison path (bath_extra_dim > 0).
        self.aatype_embed = nn.Embedding(21, c.c_hidden)  # 20 AA + unknown
        state_dim = c.q_dim + c.bath_extra_dim
        self.state_input_norm = nn.LayerNorm(state_dim)
        self.state_proj = nn.Linear(state_dim, c.c_hidden)

        # Anchor-local displacement head: the ONE positional output.  Emits
        # d_local in the anchor frame; applied as X_hat = R_anchor @ d_local +
        # t_anchor (in forward).  Small-norm (not zero) init so the state
        # pathway receives a gradient from step 0 while starting near the
        # anchor (persistence-like) — zero init would starve the trunk.
        self.local_position_head = nn.Sequential(
            nn.Linear(c.c_hidden, c.c_hidden),
            nn.SiLU(),
            nn.Linear(c.c_hidden, 3),
        )
        nn.init.normal_(self.local_position_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.local_position_head[-1].bias)
        self.max_local_position_update = float(c.max_local_position_update)

        # ── Anchor-free positional head (optional) ────────────────────────────
        # When c.anchor_free, H predicts the internal CA cloud in the anchor-local
        # frame directly (no anchor baseline).  Two heads:
        #   internal_position_head : s -> x_internal  (before IPA; seeds geometry)
        #   internal_refine_head   : s -> Δx_internal (after IPA; refines it)
        # Both are zero-ish init so training starts stable; the scale comes from
        # c.coord_scale.
        self.anchor_free = bool(getattr(c, "anchor_free", False))
        self.coord_scale = float(getattr(c, "coord_scale", 30.0))
        if self.anchor_free:
            # Residue-order signal.  s carries sequence identity + H but no
            # explicit ordering, which the anchored decoder got for free from
            # the anchor positions.  The anchor-free decoder predicts the whole
            # chain, so it needs to know where a residue sits along it.  This is
            # a *canonical* (invariant) embedding of the sequence position and
            # does not affect SE(3) equivariance (placement still happens via
            # R0/c0).  Small init so it modulates rather than dominates H.
            self.res_pos_embed = nn.Embedding(1024, c.c_hidden)
            nn.init.normal_(self.res_pos_embed.weight, std=0.02)
            self.internal_position_head = nn.Sequential(
                nn.Linear(c.c_hidden, c.c_hidden),
                nn.SiLU(),
                nn.Linear(c.c_hidden, 3),
            )
            nn.init.normal_(self.internal_position_head[-1].weight, std=1e-2)
            nn.init.zeros_(self.internal_position_head[-1].bias)
            self.internal_refine_head = nn.Sequential(
                nn.Linear(c.c_hidden, c.c_hidden),
                nn.SiLU(),
                nn.Linear(c.c_hidden, 3),
            )
            nn.init.normal_(self.internal_refine_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.internal_refine_head[-1].bias)

        # Global-context pooling over the state pathway (see DecoderConfig).
        # Zero-init the output layer so the decoder starts identical to the
        # no-global-context decoder and the channel is learned in.
        self.global_h_context = bool(getattr(c, "global_h_context", False))
        if self.global_h_context:
            self.global_h_proj = nn.Sequential(
                nn.Linear(state_dim, c.c_hidden),
                nn.SiLU(),
                nn.Linear(c.c_hidden, c.c_hidden),
            )
            nn.init.zeros_(self.global_h_proj[-1].weight)
            nn.init.zeros_(self.global_h_proj[-1].bias)

        # Contact graph builder (frozen after first forward)
        self.contact_builder = ContactGraphBuilder(cutoff=8.0)

        # IPA blocks
        self.ipa_blocks = nn.ModuleList([
            InvariantPointAttention(
                c_hidden=c.c_hidden,
                n_heads=c.n_heads,
                n_query_points=c.n_query_points,
                n_value_points=c.n_value_points,
            )
            for _ in range(c.n_ipa_blocks)
        ])

        # Transition layers between IPA blocks
        self.transitions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c.c_hidden, c.c_hidden * 2),
                nn.SiLU(),
                nn.Linear(c.c_hidden * 2, c.c_hidden),
            )
            for _ in range(c.n_ipa_blocks)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(c.c_hidden) for _ in range(c.n_ipa_blocks)
        ])

        self.dropout = nn.Dropout(c.dropout)

        # Frame orientation update only (quaternion).  Translation is handled
        # exclusively by local_position_head; there is no separate frame
        # translation residual (removed, see DecoderConfig).
        self.frame_update = FrameUpdate(c.c_hidden)
        self.torsion_head = TorsionHead(c.c_hidden, num_torsions=NUM_TORSIONS, n_bins=c.torsion_bins)

    def embed_inputs(
        self,
        aatype: torch.Tensor,
        q_dynamics: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed the invariant inputs into the single representation.

        Args:
            aatype: [B, L] residue type indices (sequence c)
            q_dynamics: [B, L, state_dim] invariant state = Q := H_t (mainline),
                or [Q := H_t || S_t] on the D_chi' comparison path.
            residue_mask: [B, L] optional valid-residue mask; required for a
                correct global-mean pool when sequences are zero-padded.

        Returns:
            [B, L, c_hidden] single representation
        """
        B, L = aatype.shape

        state_in = self.state_input_norm(q_dynamics)     # [B, L, state_dim]
        s_aa = self.aatype_embed(aatype)                 # [B, L, c_hidden]
        s_q = self.state_proj(state_in)                  # [B, L, c_hidden]

        # Combine invariant embeddings.  No separate h_frozen context pathway:
        # the dynamics state IS H_t on the mainline (see DecoderConfig).
        s = s_aa + s_q

        # Global-context pooling: broadcast masked mean_j state[j] to every
        # residue.  Uses only the invariant state — no anchor geometry — so it
        # stays inside the §2.0.0/§3.1 decoder contract.
        if self.global_h_context:
            if residue_mask is None:
                h_mean = state_in.mean(dim=1, keepdim=True)          # [B, 1, D]
            else:
                m = residue_mask.to(state_in.dtype).unsqueeze(-1)    # [B, L, 1]
                h_mean = (state_in * m).sum(dim=1, keepdim=True) / m.sum(
                    dim=1, keepdim=True).clamp(min=1.0)              # [B, 1, D]
            s = s + self.global_h_proj(h_mean)                       # broadcast [B,1,C]

        return s

    def forward(
        self,
        aatype: torch.Tensor,
        q_dynamics: torch.Tensor,
        R_anchor: torch.Tensor,
        t_anchor: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the structure decoder.

        Decoding follows the invariant x equivariant decomposition:

            X_hat = anchor_frame  x  d_local(Q := H_t, c [, S])

        ``R_anchor/t_anchor`` supply only the equivariant rigid-body basis
        (absolute placement); all internal conformation is predicted from the
        invariant state ``q_dynamics`` (+ sequence ``aatype``).

        Args:
            aatype: [B, L] residue type indices (sequence c)
            q_dynamics: [B, L, state_dim] invariant state (Q := H_t, [+ S_t])
            R_anchor: [B, L, 3, 3] anchor frame orientations (last observed frame)
            t_anchor: [B, L, 3] anchor frame positions
            residue_mask: [B, L] optional mask

        Returns:
            dict with:
                R: [B, L, 3, 3] predicted frame orientations
                t: [B, L, 3] predicted frame positions
                torsion_logits: [B, L, 7, 36] torsion angle classification logits
                backbone_coords: [B, L, 4, 3] N, CA, C, O
                atom37_coords: [B, L, 37, 3] full heavy-atom coordinates
                atom37_mask: [B, L, 37] atom presence mask
        """
        B, L = aatype.shape
        device = aatype.device

        # Invariant single representation from (Q := H_t, c [, S]) only.
        s = self.embed_inputs(aatype, q_dynamics,
                              residue_mask=residue_mask)  # [B, L, c_hidden]

        if self.anchor_free:
            # ── Anchor-free path ──────────────────────────────────────────────
            # The decoder predicts the WHOLE internal structure from H, in one
            # shared canonical frame, and the anchor enters only as a single
            # rigid-body transform at the very end:
            #
            #     X_internal = f(H, c)                       (H alone)
            #     X_global   = R0 @ X_internal + c0          (anchor, post-decode)
            #
            # The canonical frame (R0, c0) is derived deterministically from the
            # anchor: c0 = masked centroid of the anchor CA cloud, R0 = the
            # rotation of the anchor's first valid residue frame.  Because the
            # frame is *shared* by all residues (not per-residue), X_internal is
            # an actual structure — H has to produce the fold.  A zero/constant
            # H yields a single mean structure and cannot satisfy the structural
            # losses, so there is no anchor shortcut left.
            c0 = self._masked_centroid(t_anchor, residue_mask)   # [B, 3]
            R0 = self._reference_rotation(R_anchor, residue_mask)  # [B, 3, 3]

            # Add a canonical residue-order embedding so the decoder can place
            # residues along the chain (the anchor used to supply this).  Clamp
            # to the table size for very long chains.
            pos_idx = torch.arange(L, device=s.device).clamp(max=1023)  # [L]
            s = s + self.res_pos_embed(pos_idx)[None]                   # [B,L,C]

            # H → internal CA positions in the canonical frame.
            #
            # Start from a compact, protein-like chain prior so the decoder does
            # not begin from a degenerate point (every residue on the centroid)
            # or from an over-extended line (which is 100s of Å long for real L).
            # The prior is a deterministic self-avoiding-ish random-walk coil
            # with the correct 3.8 Å CA-CA bond length; its radius of gyration
            # scales like L^(1/2)·bond, close to a real folded chain.  H predicts
            # a bounded correction on top.  The prior is purely internal; the
            # anchor still enters only through (R0, c0) below.
            prior = self._coil_prior(L, s.dtype, s.device)        # [L, 3]
            delta = self.coord_scale * torch.tanh(
                self.internal_position_head(s)
            )                                                    # [B, L, 3]
            x_int = prior.unsqueeze(0) + delta                   # [B, L, 3]

            # Place into the global frame for the IPA geometric bias.
            #   x_glob = R0 @ x_int + c0     (column-vector convention)
            R0_l = R0.unsqueeze(1)                               # [B, 1, 3, 3]
            t_ipa = (R0_l @ x_int.unsqueeze(-1)).squeeze(-1) + c0  # [B, L, 3]

            self.contact_builder(t_ipa, residue_mask)
            R = R0_l.expand(B, L, 3, 3).clone()
            t = t_ipa

            for block, norm, transition in zip(
                self.ipa_blocks, self.layer_norms, self.transitions
            ):
                s_residual, _ = block(s, R, t, residue_mask)
                s = norm(s + self.dropout(s_residual))
                s = s + transition(s)

            # IPA-refined internal coordinates (residual in the canonical frame).
            dx = (self.coord_scale * 0.1) * torch.tanh(
                self.internal_refine_head(s)
            )                                                    # [B, L, 3]
            x_final = x_int + dx                                 # [B, L, 3]

            # ── Anchor placement (the ONLY place the anchor is used) ──────────
            dt_global = (R0_l @ x_final.unsqueeze(-1)).squeeze(-1)  # [B, L, 3]
            t = dt_global + c0                                    # global CA

            # Per-residue frame orientation: anchor-independent update composed
            # onto the canonical basis (also rigged to H via s).
            dq = self.frame_update(s)
            R_update = rotation_matrix_from_quaternion(dq)
            R = R0_l @ R_update                                  # [B, L, 3, 3]

            # ``local_position_update_local`` keeps its historical meaning as a
            # per-residue displacement measured from the anchor's own position,
            # so downstream diagnostics that compare against a Kabsch-aligned
            # displacement target stay valid.
            d_local = (
                R_anchor.transpose(-1, -2) @ (t - t_anchor).unsqueeze(-1)
            ).squeeze(-1)
            x_internal_out = x_final

        else:
            # ── Original anchor-anchored path ─────────────────────────────────
            # Build frozen contact graph from anchor structure
            self.contact_builder(t_anchor, residue_mask)

            # Initialize frames at anchor
            R = R_anchor.clone()
            t = t_anchor.clone()

            # Process through IPA blocks (refines orientation on anchor-positioned frames)
            for block, norm, transition in zip(self.ipa_blocks, self.layer_norms, self.transitions):
                s_residual, _ = block(s, R, t, residue_mask)
                s = norm(s + self.dropout(s_residual))
                s = s + transition(s)

            # ── Anchor-local internal displacement (the ONE positional path) ──
            # d_local is bounded component-wise and expressed in each residue's
            # anchor-local frame; H (the invariant state) is responsible for it.
            # Applied as X_hat = R_anchor @ d_local + t_anchor below.
            d_local = self.max_local_position_update * torch.tanh(
                self.local_position_head(s)
            )                                          # [B, L, 3]

            # Orientation update (quaternion), composed onto the anchor frame.
            dq = self.frame_update(s)
            R_update = rotation_matrix_from_quaternion(dq)
            R = R_anchor @ R_update

            # Equivariant placement: local -> global via the anchor rotation basis.
            dt_global = (R_anchor @ d_local.unsqueeze(-1)).squeeze(-1)
            t = t + dt_global
            x_internal_out = None

        # Predict torsions
        torsion_logits = self.torsion_head(s)
        torsions = _torsion_logits_to_angles(torsion_logits)

        # ── Frame -> atom37 via OFFICIAL OpenFold geometry ──
        # The canonical decoded structure places every heavy atom (backbone +
        # sidechain) from the predicted per-residue frames (R, t) and predicted
        # torsion angles using OpenFold's `torsion_angles_to_frames` +
        # `frames_and_literature_positions_to_atom14_pos` + `atom14_to_atom37`.
        # No hand-written ideal bond/angle tables or sidechain trees are used.
        atom37_coords, atom37_mask = reconstruct_atom37_from_frames(
            R, t, aatype, torsions
        )
        openfold_atom37 = atom37_coords  # OpenFold atom37 ordering

        # Backbone view (N, CA, C, O) extracted from the reconstructed atom37.
        # Indices per OpenFold ordering: N=0, CA=1, C=2, O=4.
        N_direct = atom37_coords[..., 0, :]
        CA_direct = atom37_coords[..., 1, :]
        C_direct = atom37_coords[..., 2, :]
        O_direct = atom37_coords[..., 4, :]
        backbone_direct = torch.stack(
            [N_direct, CA_direct, C_direct, O_direct], dim=2
        )  # [B, L, 4, 3]

        # Torsion-chain-extension diagnostic (unchanged, not canonical output).
        N_t, CA_t, C_t, O_t = reconstruct_backbone_from_torsions(R, t, torsions)
        backbone_torsion = torch.stack([N_t, CA_t, C_t, O_t], dim=2)  # diagnostic

        return {
            "R": R,
            "t": t,
            "local_position_update_local": d_local,
            "translation_update_local": d_local,   # alias (legacy name)
            "torsion_logits": torsion_logits,
            "torsions": torsions,
            "backbone_direct": backbone_direct,    # frame-only, clean gradient
            "CA_direct": CA_direct,                # frame-only CA
            "backbone_coords": backbone_direct,   # canonical decoded backbone
            "backbone_torsion_reconstructed": backbone_torsion,
            "atom37_coords": atom37_coords,
            "atom37_mask": atom37_mask,
            "N": N_direct,
            "CA": CA_direct,
            "C": C_direct,
            "O": O_direct,
            "single_repr": s,
            "x_internal": x_internal_out,          # anchor-free canonical coords (or None)
        }


    @staticmethod
    def _coil_prior(L: int, dtype, device) -> torch.Tensor:
        """Deterministic compact chain prior [L, 3] with CA-CA ≈ 3.8 Å.

        A self-avoiding random-walk coil: each step picks the direction, among a
        fixed set of candidates, that points most away from the chain centroid
        while keeping the bond length fixed.  This gives a globular-ish chain
        whose extent grows like sqrt(L) rather than L, so long proteins do not
        start as 100s-of-Å rods.  Cached per (L, dtype, device).
        """
        key = (L, dtype, device)
        cache = getattr(StructureDecoder, "_coil_cache", None)
        if cache is None:
            cache = {}
            StructureDecoder._coil_cache = cache
        if key in cache:
            return cache[key]

        bond = 3.8
        # Deterministic direction candidates (unit vectors) on a Fibonacci sphere.
        n_dir = 32
        k = torch.arange(n_dir, dtype=torch.float64)
        golden = (1 + 5 ** 0.5) / 2
        z = 1 - 2 * (k + 0.5) / n_dir
        r = torch.sqrt(torch.clamp(1 - z * z, min=0.0))
        theta = 2 * math.pi * k / golden
        dirs = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], -1)
        dirs = dirs.to(dtype=dtype, device=device)             # [n_dir, 3]

        pts = torch.zeros(L, 3, dtype=dtype, device=device)
        rng = torch.Generator(device="cpu").manual_seed(1234 + L)
        # Self-avoiding walk: at each step pick the candidate whose endpoint is
        # FARTHEST from all existing atoms (capped to a local window for speed).
        # This folds the chain back on itself instead of shooting outward, giving
        # a real protein-like globule with Rg ~ bond * sqrt(L).
        for i in range(1, L):
            cand = pts[i - 1].unsqueeze(0) + bond * dirs        # [n_dir, 3]
            window = pts[:i]                                    # [i, 3]
            if i > 64:                                          # local window
                window = window[-64:]
            # distance from each candidate to each existing atom
            d = torch.cdist(cand, window)                       # [n_dir, w]
            min_d = d.min(dim=1).values                         # [n_dir]
            noise = (torch.randn(n_dir, generator=rng).to(dtype=dtype, device=device)
                     * 0.05)
            score = min_d + noise
            pts[i] = cand[int(torch.argmax(score))]
        pts = pts - pts.mean(dim=0, keepdim=True)
        cache[key] = pts
        return pts

    @staticmethod
    def _masked_centroid(
        t: torch.Tensor, residue_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Masked mean over residues of [B, L, 3] -> [B, 1, 3] (broadcastable)."""
        if residue_mask is None:
            return t.mean(dim=1, keepdim=True)
        m = residue_mask.to(t.dtype).unsqueeze(-1)       # [B, L, 1]
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (t * m).sum(dim=1, keepdim=True) / denom  # [B, 1, 3]

    @staticmethod
    def _reference_rotation(
        R_anchor: torch.Tensor, residue_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Pick a single canonical rotation per batch element from R_anchor.

        Uses the first VALID residue's frame as the shared canonical basis so
        the decoded internal structure is placed consistently with the anchor's
        global orientation (equivariance), while all residues share ONE basis
        (so the predicted internal coordinates form a real structure rather than
        per-residue anchored offsets).

        Returns [B, 3, 3].
        """
        B, L = R_anchor.shape[:2]
        if residue_mask is None:
            return R_anchor[:, 0]
        # index of first valid residue per batch (fallback to 0)
        valid = residue_mask.to(torch.int32)             # [B, L]
        has_valid = valid.sum(dim=1) > 0
        first_idx = torch.argmax(valid, dim=1)           # [B]
        first_idx = torch.where(has_valid, first_idx,
                                torch.zeros_like(first_idx))
        ar = torch.arange(B, device=R_anchor.device)
        return R_anchor[ar, first_idx]                   # [B, 3, 3]


def _torsion_logits_to_angles(logits: torch.Tensor) -> torch.Tensor:
    """Convert torsion logits [B, L, 7, 36] to continuous angles [B, L, 7] in [-pi, pi]."""
    probs = F.softmax(logits, dim=-1)
    bin_centers = torch.linspace(-math.pi, math.pi, logits.shape[-1] + 1, device=logits.device)
    bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2  # midpoints
    angles = (probs * bin_centers.unsqueeze(0).unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    return angles


__all__ = [
    "DecoderConfig",
    "StructureDecoder",
    "FrameUpdate",
    "TorsionHead",
    "ContactGraphBuilder",
    "reconstruct_backbone_from_torsions",
    "reconstruct_atom37",
    "reconstruct_atom37_from_frames",
    "NUM_TORSIONS",
    "ATOM37_DIM",
]
