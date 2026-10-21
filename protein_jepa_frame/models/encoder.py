from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _internal_distances(coords: torch.Tensor) -> torch.Tensor:
    pairs = ((0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3))
    values = [torch.linalg.norm(coords[..., i, :] - coords[..., j, :], dim=-1) for i, j in pairs]
    return torch.stack(values, dim=-1)


def _dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    b0 = a - b
    b1 = F.normalize(c - b, dim=-1, eps=1.0e-8)
    b2 = d - c
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def _backbone_torsion_features(
    coords: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor,
    chain_id: torch.Tensor,
) -> torch.Tensor:
    """Return chirality-sensitive sin/cos(phi, psi, omega) per residue."""
    B, T, L = coords.shape[:3]
    angles = coords.new_zeros(B, T, L, 3)
    angle_mask = torch.zeros(B, T, L, 3, dtype=torch.bool, device=coords.device)
    if L < 2:
        return torch.cat([angles.sin(), angles.cos() * angle_mask], dim=-1)
    adjacent = (chain_id[:, 1:] == chain_id[:, :-1]) & (residue_index[:, 1:] - residue_index[:, :-1] == 1)
    pair_valid = residue_mask[:, :, 1:] & residue_mask[:, :, :-1] & adjacent[:, None, :]

    phi = _dihedral(coords[:, :, :-1, 2], coords[:, :, 1:, 0], coords[:, :, 1:, 1], coords[:, :, 1:, 2])
    psi = _dihedral(coords[:, :, :-1, 0], coords[:, :, :-1, 1], coords[:, :, :-1, 2], coords[:, :, 1:, 0])
    omega = _dihedral(coords[:, :, :-1, 1], coords[:, :, :-1, 2], coords[:, :, 1:, 0], coords[:, :, 1:, 1])
    angles[:, :, 1:, 0] = phi
    angles[:, :, :-1, 1] = psi
    angles[:, :, :-1, 2] = omega
    angle_mask[:, :, 1:, 0] = pair_valid
    angle_mask[:, :, :-1, 1] = pair_valid
    angle_mask[:, :, :-1, 2] = pair_valid
    mask = angle_mask.to(dtype=coords.dtype)
    return torch.cat([angles.sin() * mask, angles.cos() * mask], dim=-1)


def _local_backbone_frames(coords: torch.Tensor) -> torch.Tensor:
    ca = coords[:, :, 1]
    x_axis = F.normalize(coords[:, :, 2] - ca, dim=-1, eps=1.0e-8)
    n_direction = F.normalize(coords[:, :, 0] - ca, dim=-1, eps=1.0e-8)
    z_axis = F.normalize(torch.cross(x_axis, n_direction, dim=-1), dim=-1, eps=1.0e-8)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1.0e-8)
    return torch.stack([x_axis, y_axis, z_axis], dim=-2)


def _pair_orientation_features(coords: torch.Tensor) -> torch.Tensor:
    ca = coords[:, :, 1]
    frames = _local_backbone_frames(coords)
    delta = ca[:, None, :, :] - ca[:, :, None, :]
    direction = F.normalize(delta, dim=-1, eps=1.0e-8)
    local_direction = torch.einsum("bikc,bijc->bijk", frames, direction)
    relative_rotation = torch.einsum("bikc,bjlc->bijkl", frames, frames).flatten(-2)
    return torch.cat([local_direction, relative_rotation], dim=-1)


def _within_frame_canonical_positions(
    coords: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Express each residue's CA in a per-frame canonical frame.

    Args:
        coords: ``[Bf, L, 4, 3]`` backbone coordinates (N, CA, C, O).
        valid:  ``[Bf, L]`` bool residue mask.

    Returns:
        ``[Bf, L, 3]`` centroid-relative CA positions in the frame's principal
        axes.  SE(3)-invariant (the frame is derived from the same frame), but
        position-aware within the fold.

    The canonical frame is the PCA eigenbasis of the visible CA covariance with
    a deterministic sign convention (largest-|component| eigenvector entry made
    positive) so the frame is stable under small perturbations and identical for
    identical frames.
    """
    ca = coords[:, :, 1, :]                                  # [Bf, L, 3]
    m = valid.to(ca.dtype).unsqueeze(-1)                     # [Bf, L, 1]
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)        # [Bf, 1, 1]
    centroid = (ca * m).sum(dim=1, keepdim=True) / denom     # [Bf, 1, 3]
    centred = (ca - centroid) * m                            # [Bf, L, 3]

    # Covariance [Bf, 3, 3]; PCA basis via eigh (deterministic, no sign flip in
    # the eigenvalues, so we fix the eigenvector signs explicitly).
    cov = torch.einsum("bli,blj->bij", centred, centred)     # [Bf, 3, 3]
    # Tiny jitter keeps eigh well-posed for degenerate (e.g. single-residue) cases.
    eye = torch.eye(3, device=coords.device, dtype=coords.dtype).unsqueeze(0)
    cov = cov + 1.0e-6 * eye
    evals, evecs = torch.linalg.eigh(cov)                    # ascending eigenvalues
    basis = evecs.flip(-1)                                   # [Bf, 3, 3] cols = axes, desc eigval
    # Rotation-invariant sign convention: flip each axis so the projected cloud
    # has positive skewness along it.  The eigenvector sign is otherwise only
    # defined up to ±1, and a convention based on the basis entries themselves
    # is NOT invariant to a global rotation (the flipped entry changes).  The
    # skewness of the *projected points* is intrinsic to the cloud, so this
    # convention is identical for a globally-rotated copy.
    proj = torch.einsum("blj,bjk->blk", centred, basis)      # [Bf, L, 3]
    m3 = m.squeeze(-1)                                       # [Bf, L]
    n = m3.sum(dim=1, keepdim=True).clamp_min(1.0)           # [Bf, 1]
    skew = (proj.pow(3).sum(dim=1) / n)                      # [Bf, 3]
    sign = torch.where(skew >= 0, torch.ones_like(skew), -torch.ones_like(skew))
    basis = basis * sign.unsqueeze(1)                        # [Bf, 3, 3]
    local = proj * sign.unsqueeze(1)                         # [Bf, L, 3]
    return local * m


def _canonical_frame(
    coords: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centroid and PCA basis of a frame's visible CA cloud.

    Args:
        coords: ``[N, L, 4, 3]`` backbone coordinates.
        valid:  ``[N, L]`` bool residue mask.

    Returns:
        ``(centroid [N, 3], basis [N, 3, 3])`` with ``basis`` columns the
        principal axes in descending-eigenvalue order and a rotation-invariant
        sign convention (matches ``_within_frame_canonical_positions``).
    """
    ca = coords[:, :, 1, :]
    m = valid.to(ca.dtype).unsqueeze(-1)
    centroid = (ca * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
    centred = (ca - centroid) * m
    cov = torch.einsum("bli,blj->bij", centred, centred)
    cov = cov + 1.0e-6 * torch.eye(3, device=coords.device, dtype=coords.dtype).unsqueeze(0)
    _, evecs = torch.linalg.eigh(cov)
    basis = evecs.flip(-1)
    proj = torch.einsum("blj,bjk->blk", centred, basis)
    n = m.sum(dim=1).clamp_min(1.0)
    skew = proj.pow(3).sum(dim=1) / n
    sign = torch.where(skew >= 0, torch.ones_like(skew), -torch.ones_like(skew))
    basis = basis * sign.unsqueeze(1)
    return centroid.squeeze(1), basis


def _positions_in_anchor_frame(
    coords: torch.Tensor, valid: torch.Tensor,
    anchor_centroid: torch.Tensor, anchor_basis: torch.Tensor,
) -> torch.Tensor:
    """Express each residue's CA in the *anchor* frame's canonical coordinates.

    The anchor's centroid/basis are derived from the anchor frame's own geometry,
    so both are invariant to a global rigid motion.  Projecting the current
    frame's CA into that basis therefore yields an SE(3)-INVARIANT per-residue
    position that is directly comparable across frames: the difference between
    this and the anchor's own ``_within_frame_canonical_positions`` is the
    (rotation-canonicalised) displacement the decoder must reproduce.

    Args:
        coords: ``[N, L, 4, 3]`` the frame being encoded.
        valid:  ``[N, L]`` residue mask.
        anchor_centroid: ``[N, 3]`` anchor centroid (broadcast per row).
        anchor_basis: ``[N, 3, 3]`` anchor PCA basis (columns = axes).

    Returns:
        ``[N, L, 3]`` CA positions relative to the anchor centroid, in the
        anchor's principal-axis coordinates, zeroed on invalid residues.
    """
    ca = coords[:, :, 1, :]
    m = valid.to(ca.dtype).unsqueeze(-1)
    rel = (ca - anchor_centroid.unsqueeze(1)) * m
    return torch.einsum("blj,bjk->blk", rel, anchor_basis) * m


def _orientation_pair_bias(
    coords: torch.Tensor, weight: torch.Tensor, chunk: int = 32
) -> torch.Tensor:
    """Compute ``Linear(pair_orientation(coords))`` without materialising the
    full ``[B, L, L, 12]`` feature tensor.

    The pair-orientation tensor is O(B·L²·12) and, at K context frames × batch,
    is the dominant activation in the encoder.  Fusing the linear projection
    into the contraction lets us stream over query residues, cutting peak
    memory from O(B·L²) to O(B·L·chunk) while producing the identical bias.

    Args:
        coords: ``[B, L, 4, 3]`` backbone coordinates.
        weight: ``Linear.weight`` of shape ``[num_heads, 12]``.
        chunk: query-residue block size.

    Returns:
        ``[B, L, L, num_heads]`` orientation bias.
    """
    ca = coords[:, :, 1]
    frames = _local_backbone_frames(coords)            # [B, L, 3, 3]
    B, L, _, _ = frames.shape
    # Split the 12 input features into 3 (local direction) + 9 (relative rot).
    w_dir = weight[:, :3]                              # [H, 3]
    w_rot = weight[:, 3:].reshape(weight.shape[0], 3, 3)   # [H, 3, 3]
    # Direction of the (i -> j) unit vector expressed in residue i's frame.
    # Sign convention matches _pair_orientation_features: delta[i,j] = ca_j - ca_i.
    delta = ca[:, None, :, :] - ca[:, :, None, :]      # [B, L, L, 3]
    direction = F.normalize(delta, dim=-1, eps=1.0e-8)
    outs = []
    for s in range(0, L, chunk):
        e = min(L, s + chunk)
        f_q = frames[:, s:e]                           # [B, q, 3, 3]
        # local direction: contract the local frame vector axis with direction
        # direction[:, s:e] is [B, q, L, 3] (indexed [i, j]); label j explicitly.
        local_dir = torch.einsum(
            "bqkc,bqjc->bqjk", f_q, direction[:, s:e]
        )                                              # [B, q, L, 3]
        # relative rotation: element (k, l) = sum_c f_q[.,k,c] * f_j[.,l,c]
        rel = torch.einsum("bqkc,bjlc->bqjkl", f_q, frames)     # [B,q,L,3,3]
        # fuse projection: bias[h] = w_dir[h]·local_dir + w_rot[h]·rel
        term_dir = torch.einsum("bqja,ha->bqjh", local_dir, w_dir)
        term_rot = torch.einsum("bqjkl,hkl->bqjh", rel, w_rot)
        outs.append(term_dir + term_rot)
    return torch.cat(outs, dim=1)                      # [B, L, L, H]


class VisibleGraphAttentionLayer(nn.Module):
    """Distance-biased attention over visible residues inside each frame."""

    def __init__(self, dim: int, num_heads: int, num_rbf: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(num_rbf, num_heads, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        ca: torch.Tensor,
        valid: torch.Tensor,
        rbf: torch.Tensor,
        orientation_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Bf, L, D = h.shape
        x = self.norm1(h)
        q = self.q(x).view(Bf, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(Bf, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(Bf, L, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores + self.edge_bias(rbf).permute(0, 3, 1, 2)
        if orientation_bias is not None:
            scores = scores + orientation_bias.permute(0, 3, 1, 2)
        key_mask = valid[:, None, None, :]
        scores = scores.masked_fill(~key_mask, -1.0e4)
        attn = torch.softmax(scores, dim=-1)
        attn = attn * key_mask.float()
        update = torch.matmul(attn, v).transpose(1, 2).reshape(Bf, L, D)
        h = h + self.dropout(self.out(update))
        h = h + self.dropout(self.mlp(self.norm2(h)))
        return h * valid.unsqueeze(-1).float()


class ProteinGeometryEncoder(nn.Module):
    """SE(3)-invariant residue-frame encoder for `[B,T,L,4,3]` backbone clips.

    The layer uses intra-residue distances and visible-only pairwise CA
    distances. It deliberately avoids absolute coordinates.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        num_rbf: int = 16,
        dropout: float = 0.1,
        max_residues: int = 4096,
        orientation_features: bool = False,
        identity_embed_scale: float = 1.0,
        relative_position_features: bool = False,
        anchor_conditioned: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_rbf = num_rbf
        self.orientation_features = orientation_features
        # Within-frame canonical-frame position channel (2026-10-18).
        #
        # The encoder's other inputs (intra-residue distances, torsions, pairwise
        # CA distances/rotations) are invariant to a *global* rigid motion AND
        # carry no per-residue position within the fold: two residues with the
        # same local geometry and the same CA distance pattern get identical
        # features.  A probe on such H can *overfit* the displacement of seen
        # frames (R2=1.0) but cannot generalise to unseen frames of the SAME
        # protein (R2=0.09), even with anchor+future concat — because H has no
        # reference frame in which "how residue i moved relative to the anchor"
        # is expressible.  This channel fixes exactly that:
        #
        # For each frame independently, build a canonical orthonormal frame from
        # the visible CA cloud (centroid + principal axes of the CA covariance)
        # and express each residue's CA displacement from the centroid in it.
        # The result is still SE(3)-INVARIANT (the frame is derived from the frame
        # itself, so a global rotation/translation cancels), but it now says
        # *where* in the fold each residue sits.  Then the anchor→future change
        # in this channel is exactly the (rotation-canonicalised) displacement
        # that the decoder must reproduce.
        self.relative_position_features = relative_position_features

        # Time-invariant identity embeddings (amino-acid type, residue index,
        # chain id) are constant along a trajectory.  If they dominate the
        # geometry projection the encoder can satisfy the JEPA objective from
        # the static component alone, collapsing H to a time-invariant code
        # (cos(H_t, H_{t+Δ}) ≈ 0.999 even at Δ = 400 ns).  The scale keeps
        # them from swamping the geometry signal.
        self.identity_embed_scale = float(identity_embed_scale)
        self.aatype_embed = nn.Embedding(21, latent_dim)
        self.residue_embed = nn.Embedding(max_residues, latent_dim)
        self.chain_embed = nn.Embedding(128, latent_dim)
        # No bias in the geometry projection: a learned bias here accumulates
        # into a constant offset across all frames and proteins, which (combined
        # with the geom_gain scale) produces the ~10x shared direction that
        # dominates H and drives cross-protein cosine similarity toward 1.
        self.geom_proj = nn.Linear(12 if orientation_features else 6, latent_dim, bias=False)
        # Projection for the within-frame canonical-frame position channel
        # (centroid-relative CA in principal-axis coordinates; 3 dims).
        if relative_position_features:
            self.relpos_proj = nn.Linear(3, latent_dim, bias=False)
            nn.init.normal_(self.relpos_proj.weight, std=0.1)
        else:
            self.relpos_proj = None
        # Anchor-conditioned position channel (2026-10-18).
        #
        # The relpos channel above is per-frame: its PCA frame is re-derived from
        # each frame's own cloud, so two geometrically similar conformations get
        # almost identical features regardless of how far the atoms moved between
        # them.  A ridge probe confirmed the consequence: dH carries zero
        # linearly-decodable displacement at every horizon (best alpha -> 100,
        # i.e. the zero map), so the decoder can only output near-persistence.
        #
        # This channel fixes the *reference frame* problem: the anchor frame's
        # centroid + PCA basis are computed once and every frame's CA is projected
        # into that basis.  The result stays SE(3)-invariant (both the anchor basis
        # and the projected current-frame CA transform identically under a global
        # rotation/translation) but is now measured against a fixed reference, so
        # "where residue i is relative to the anchor's frame" is expressible and
        # changes as the protein moves.
        self.anchor_conditioned = anchor_conditioned
        if anchor_conditioned:
            self.anchor_pos_proj = nn.Linear(3, latent_dim, bias=False)
            nn.init.normal_(self.anchor_pos_proj.weight, std=0.1)
        else:
            self.anchor_pos_proj = None
        # Learnable per-channel gain on the geometry term, initialised so that
        # geometry is at least as strong as the (scaled) identity embeddings.
        self.geom_gain = nn.Parameter(torch.ones(latent_dim))
        with torch.no_grad():
            nn.init.normal_(self.aatype_embed.weight, std=self.identity_embed_scale)
            nn.init.normal_(self.residue_embed.weight, std=self.identity_embed_scale)
            nn.init.normal_(self.chain_embed.weight, std=self.identity_embed_scale)
        self.orientation_bias = nn.Linear(12, num_heads, bias=False) if orientation_features else None
        self.layers = nn.ModuleList(
            [VisibleGraphAttentionLayer(latent_dim, num_heads, num_rbf, dropout) for _ in range(depth)]
        )
        # No learnable bias: a constant bias term grows during training and
        # creates a shared "anchor" direction that all H vectors align to,
        # pushing cross-protein cosine similarity toward 1 even though the
        # per-protein geometry information is preserved in the residual
        # subspace.  elementwise_affine=False removes both weight and bias;
        # the gain is already carried by geom_gain and the attention layers.
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.register_buffer("rbf_centers", torch.linspace(2.0, 22.0, num_rbf), persistent=False)

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        width = (self.rbf_centers[1] - self.rbf_centers[0]).clamp(min=1.0)
        return torch.exp(-((distances.unsqueeze(-1) - self.rbf_centers) / width) ** 2)

    def forward(
        self,
        coords: torch.Tensor,
        residue_mask: torch.Tensor,
        aatype: torch.Tensor,
        residue_index: torch.Tensor,
        chain_id: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        anchor_centroid: torch.Tensor | None = None,
        anchor_basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coords.dim() != 5 or coords.shape[-2:] != (4, 3):
            raise ValueError("coords must be [B,T,L,4,3]")
        B, T, L = coords.shape[:3]
        valid = residue_mask if token_mask is None else (residue_mask & token_mask)
        valid_f = valid.reshape(B * T, L)
        coords_f = coords.reshape(B * T, L, 4, 3)
        ca = coords_f[:, :, 1, :]

        geom = _internal_distances(coords_f).clamp(max=50.0)
        if self.orientation_features:
            # Torsions span neighboring residues, so the visibility mask must
            # participate in their validity check just like graph attention.
            torsions = _backbone_torsion_features(coords, valid, residue_index, chain_id)
            geom = torch.cat([geom, torsions.reshape(B * T, L, 6)], dim=-1)
        aa = aatype.clamp(0, 20)
        res_pos = residue_index.clamp(min=0, max=self.residue_embed.num_embeddings - 1)
        chain_pos = chain_id.clamp(min=0, max=self.chain_embed.num_embeddings - 1)
        base = (
            self.geom_proj(geom) * self.geom_gain
            + self.aatype_embed(aa).unsqueeze(1).expand(B, T, L, -1).reshape(B * T, L, -1)
            + self.residue_embed(res_pos).unsqueeze(1).expand(B, T, L, -1).reshape(B * T, L, -1)
            + self.chain_embed(chain_pos).unsqueeze(1).expand(B, T, L, -1).reshape(B * T, L, -1)
        )
        if self.relpos_proj is not None:
            # Within-frame canonical-frame position channel. Computed per frame
            # (the B*T flattening already makes each frame its own row).
            relpos = _within_frame_canonical_positions(coords_f, valid_f)  # [B*T, L, 3]
            base = base + self.relpos_proj(relpos)
        if self.anchor_pos_proj is not None and anchor_centroid is not None:
            # Anchor-conditioned position channel: express each residue's CA in
            # the anchor's canonical PCA frame so that H_future - H_anchor
            # encodes displacement rather than pure structural identity.
            anc_pos = _positions_in_anchor_frame(
                coords_f, valid_f, anchor_centroid, anchor_basis
            )
            base = base + self.anchor_pos_proj(anc_pos)
        h = base * valid_f.unsqueeze(-1).float()

        distances = torch.cdist(ca, ca).clamp(max=80.0)
        pair_valid = valid_f[:, :, None] & valid_f[:, None, :]
        rbf = self._rbf(distances) * pair_valid.unsqueeze(-1).float()
        orientation_bias = None
        if self.orientation_bias is not None:
            # Fused projection avoids the O(B*L^2*12) intermediate tensor, which
            # dominates memory once K context frames are stacked on the batch
            # axis; numerically identical to Linear(_pair_orientation_features).
            orientation_bias = _orientation_pair_bias(
                coords_f, self.orientation_bias.weight
            ) * pair_valid.unsqueeze(-1).float()
        for layer in self.layers:
            h = layer(h, ca, valid_f, rbf, orientation_bias=orientation_bias)
        h = self.norm(h) * valid_f.unsqueeze(-1).float()
        return h.reshape(B, T, L, self.latent_dim)
