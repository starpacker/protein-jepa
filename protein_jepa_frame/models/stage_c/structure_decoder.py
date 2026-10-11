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
    context_dim: int = 256       # H dimension from Stage A encoder
    q_dim: int = 8               # Q dimension from Stage B
    dropout: float = 0.1
    max_translation_update: float = 0.5  # per-component local-frame bound (Å)
    max_com_update: float = 4.0          # per-component COM bound (Å)
    max_q_translation_update: float = 2.0  # explicit Q→frame residual bound (Å)


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
    """Predicts SE(3) frame updates from current representation."""

    def __init__(self, c_hidden: int = 128, max_translation_update: float = 0.5):
        super().__init__()
        if max_translation_update <= 0:
            raise ValueError("max_translation_update must be positive")
        self.max_translation_update = float(max_translation_update)
        self.quat_net = nn.Sequential(
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, 4),
        )
        self.trans_net = nn.Sequential(
            nn.Linear(c_hidden, c_hidden),
            nn.SiLU(),
            nn.Linear(c_hidden, 3),
        )
        # Start from the anchor frame.  Random translation updates have no
        # useful supervision under an SE(3)-invariant objective and can create
        # large early coordinate excursions before the relative-position loss
        # has shaped the head.
        nn.init.zeros_(self.trans_net[-1].weight)
        self.trans_net[-1].bias.data.zero_()
        # Initialize quaternion to identity (w=1) with small weight
        nn.init.normal_(self.quat_net[-1].weight, std=1e-3)
        self.quat_net[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict frame updates.

        Args:
            s: [B, L, c_hidden]

        Returns:
            dq: [B, L, 4] quaternion update (w,x,y,z)
            dt: [B, L, 3] translation update (in local frame)
        """
        dq = self.quat_net(s)
        dq = dq / (dq.norm(dim=-1, keepdim=True) + 1e-8)
        # A raw translation head is an unstable integrator: large position
        # residuals can keep increasing dt even after gradient clipping.
        dt = self.max_translation_update * torch.tanh(self.trans_net(s))
        return dq, dt


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

        # Input projections
        self.aatype_embed = nn.Embedding(21, c.c_hidden)  # 20 AA + unknown
        self.h_proj = nn.Linear(c.context_dim, c.c_hidden)  # frozen H_t0
        self.q_proj = nn.Linear(c.q_dim, c.c_hidden)        # dynamics Q_t
        self.frame_condition_proj = nn.Linear(c.c_hidden * 2, c.c_hidden)

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

        # Output heads
        self.frame_update = FrameUpdate(
            c.c_hidden, max_translation_update=c.max_translation_update
        )
        if c.max_q_translation_update <= 0:
            raise ValueError("max_q_translation_update must be positive")
        # This path is deliberately Q-only.  Its zero final layer keeps the
        # decoder numerically close to the anchor at initialization, while
        # the dedicated relative-displacement loss below gives it an
        # unavoidable training signal.
        self.q_translation_head = nn.Sequential(
            nn.Linear(c.q_dim, c.c_hidden),
            nn.SiLU(),
            nn.Linear(c.c_hidden, 3),
        )
        nn.init.zeros_(self.q_translation_head[-1].weight)
        nn.init.zeros_(self.q_translation_head[-1].bias)
        self.max_q_translation_update = float(c.max_q_translation_update)
        self.torsion_head = TorsionHead(c.c_hidden, num_torsions=NUM_TORSIONS, n_bins=c.torsion_bins)

        # P3.2: Auxiliary COM predictor — bridges H→position, Q provides residual.
        # Requires batch_size >= 4 for stable gradient (Q alone is SE(3)-invariant).
        # Zero-init ensures this head does not contribute until
        # H-anchored geometry converges downstream.
        self.q_position_head = nn.Linear(c.q_dim + c.c_hidden, 3)
        if c.max_com_update <= 0:
            raise ValueError("max_com_update must be positive")
        self.max_com_update = float(c.max_com_update)
        nn.init.constant_(self.q_position_head.weight, 0.0)
        self.q_position_head.bias.data.zero_()

        # ── Basin-reversion path ──
        # Held-out measurement shows backbone dynamics at 8-96 ns are diffusive
        # with NEGATIVE velocity autocorrelation (-0.35 at 1 ns): the target
        # reverts toward the local conformational basin centre.  Predicting the
        # Kabsch-aligned context temporal mean beats persistence on CA RMSD AND
        # all contact-Jaccard metrics at every horizon (measured).  We expose
        # that geometric prior as an explicit, per-residue-gated displacement:
        #     dt_local += sigmoid(gate) * basin_dt_local
        # where basin_dt_local is the aligned context-mean CA displacement in
        # the anchor frame (supplied by caller; context-only, eval-legal).  The
        # gate is conditioned on H+Q so the model can modulate reversion
        # strength per residue (rigid cores revert little, flexible loops more).
        self.basin_gate = nn.Sequential(
            nn.Linear(c.context_dim + c.q_dim, c.c_hidden),
            nn.SiLU(),
            nn.Linear(c.c_hidden, 1),
        )
        # Bias initialised so sigmoid(gate)~0.6 — start near the measured
        # optimum beta in [0.5, 1.0] rather than at persistence (beta=0).
        nn.init.zeros_(self.basin_gate[-1].weight)
        nn.init.constant_(self.basin_gate[-1].bias, 0.4)

    def embed_inputs(
        self,
        aatype: torch.Tensor,
        h_frozen: torch.Tensor,
        q_dynamics: torch.Tensor,
    ) -> torch.Tensor:
        """Embed all inputs into single representation.

        Args:
            aatype: [B, L] residue type indices
            h_frozen: [B, L, 256] frozen H from Stage A context encoder
            q_dynamics: [B, L, 8] dynamics variables from Stage B

        Returns:
            [B, L, c_hidden] single representation
        """
        B, L = aatype.shape

        s_aa = self.aatype_embed(aatype)           # [B, L, c_hidden]
        s_h = self.h_proj(h_frozen)                 # [B, L, c_hidden]
        s_q = self.q_proj(q_dynamics)               # [B, L, c_hidden]
        s_frame = self.frame_condition_proj(
            torch.cat([s_h, s_q], dim=-1)
        )

        # Combine: sum of all embeddings
        s = s_aa + s_h + s_q + s_frame
        return s

    def q_translation_update(self, q_dynamics: torch.Tensor) -> torch.Tensor:
        """Return the explicit Q-only local translation residual."""
        return self.max_q_translation_update * torch.tanh(
            self.q_translation_head(q_dynamics)
        )

    @staticmethod
    def compute_basin_dt_local(
        context_coords: torch.Tensor,
        R_anchor: torch.Tensor,
        t_anchor: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aligned context-mean CA displacement expressed in the anchor frame.

        For each sample, every context frame's CA cloud is Kabsch-superimposed
        onto the anchor (last context frame) CA cloud using the masked
        residues, the aligned clouds are averaged to estimate the local
        conformational basin centre, and the per-residue displacement of that
        centre from the anchor CA is rotated into the anchor's local frame
        (so the caller can add it to ``dt_local`` before the ``R_anchor @``
        step, exactly like the other local translation residuals).

        Uses only context coordinates — legal for direct and autonomous modes.

        Args:
            context_coords: [B, T, L, 4, 3] context backbone (N, CA, C, O)
            R_anchor: [B, L, 3, 3] anchor frame rotations
            t_anchor: [B, L, 3] anchor frame translations (anchor CA)
            residue_mask: [B, L] valid residues (used for the rigid fit)

        Returns:
            [B, L, 3] basin-reversion displacement in the anchor local frame.
        """
        B, T, L = context_coords.shape[:3]
        device = context_coords.device
        dtype = context_coords.dtype
        ca = context_coords[..., 1, :]  # [B, T, L, 3]
        anchor_ca = ca[:, -1]           # [B, L, 3]
        if residue_mask is None:
            residue_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        mf = residue_mask.to(dtype).unsqueeze(-1)  # [B, L, 1]
        n = mf.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]

        mu_a = (anchor_ca * mf).sum(dim=1, keepdim=True) / n  # [B,1,3]
        Ac = (anchor_ca - mu_a) * mf                          # [B,L,3]

        aligned_sum = torch.zeros(B, L, 3, device=device, dtype=dtype)
        for tt in range(T):
            src = ca[:, tt]  # [B, L, 3]
            mu_s = (src * mf).sum(dim=1, keepdim=True) / n
            Sc = (src - mu_s) * mf
            H = Sc.transpose(1, 2) @ Ac  # [B,3,3] maps src -> anchor
            U, S, Vh = torch.linalg.svd(H.to(torch.float64))
            d = torch.sign(
                torch.linalg.det(Vh.transpose(-2, -1) @ U.transpose(-2, -1))
            )
            eye = torch.eye(3, device=device, dtype=torch.float64)
            eye = eye.expand(B, 3, 3).clone()
            eye[:, 2, 2] = d
            Rm = (Vh.transpose(-2, -1) @ eye @ U.transpose(-2, -1)).to(dtype)
            aligned = (Rm @ (src - mu_s).transpose(1, 2)).transpose(1, 2) + mu_a
            aligned_sum = aligned_sum + aligned
        center = aligned_sum / float(T)  # [B, L, 3] basin centre in anchor frame

        disp_global = (center - anchor_ca)  # [B, L, 3]
        disp_local = (
            R_anchor.transpose(-2, -1) @ disp_global.unsqueeze(-1)
        ).squeeze(-1)
        return disp_local * residue_mask.to(dtype).unsqueeze(-1)

    def forward(
        self,
        aatype: torch.Tensor,
        h_frozen: torch.Tensor,
        q_dynamics: torch.Tensor,
        R_anchor: torch.Tensor,
        t_anchor: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
        q_only_translation: bool = False,
        basin_dt_local: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the structure decoder.

        Args:
            aatype: [B, L] residue type indices
            h_frozen: [B, L, 256] frozen H features from context (last observed frame)
            q_dynamics: [B, L, 8] dynamics variables at target time
            R_anchor: [B, L, 3, 3] anchor frame orientations (from last observed structure)
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

        # Embed inputs
        s = self.embed_inputs(aatype, h_frozen, q_dynamics)  # [B, L, c_hidden]

        # Build frozen contact graph from anchor structure
        contact_graph = self.contact_builder(t_anchor, residue_mask)

        # Initialize frames at anchor
        R = R_anchor.clone()
        t = t_anchor.clone()

        # A global COM displacement is not identifiable under the requested
        # relative-position objective.  Keep the legacy head in the state
        # dict for checkpoint compatibility, but do not use it as a shortcut.
        delta_com_local = torch.zeros(
            B, 3, device=device, dtype=R_anchor.dtype
        )

        # Process through IPA blocks (refines internal geometry on coarsely-positioned frames)
        for block, norm, transition in zip(self.ipa_blocks, self.layer_norms, self.transitions):
            s_residual, _ = block(s, R, t, residue_mask)
            s = norm(s + self.dropout(s_residual))
            s = s + transition(s)

        # Predict frame updates (residual fine-tuning of position + rotation)
        dq, dt_local = self.frame_update(s)

        # Explicit Q→structure path.  Unlike q_proj(s), this cannot be
        # bypassed by the H/anchor-conditioned IPA representation.
        q_dt_local = self.q_translation_update(q_dynamics)
        # Phase 1 of Stage-C training intentionally removes the H/IPA
        # translation shortcut.  The CA position must then be explained by Q;
        # Phase 2 restores the normal joint residual path.
        if q_only_translation:
            dt_local = q_dt_local
        else:
            dt_local = dt_local + q_dt_local

        # ── Basin-reversion contribution ──
        # Add the gated aligned context-mean displacement.  This is the
        # dominant eval-legal signal at diffusive timescales; the IPA/Q
        # residuals above only fine-tune around it.
        basin_gate_val = None
        if basin_dt_local is not None:
            gate_in = torch.cat([h_frozen, q_dynamics], dim=-1)
            basin_gate_val = torch.sigmoid(self.basin_gate(gate_in))  # [B, L, 1]
            dt_local = dt_local + basin_gate_val * basin_dt_local

        # Apply frame update: R_new = R_anchor @ R_update
        R_update = rotation_matrix_from_quaternion(dq)
        R = R_anchor @ R_update

        # Apply translation update (in local anchor frame)
        dt_global = (R_anchor @ dt_local.unsqueeze(-1)).squeeze(-1)
        t = t + dt_global  # residual on top of coarse positioning

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
            "delta_com_local": delta_com_local,
            "translation_update_local": dt_local,
            "q_translation_update_local": q_dt_local,
            "basin_gate": basin_gate_val,
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
        }


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
