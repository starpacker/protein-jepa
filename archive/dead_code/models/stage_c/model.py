"""Stage C full model: frozen Stage A/B + structure decoder pipeline.

Gate 4 three-class evaluation:
1. oracle future Q -> decoder (decoder upper bound)
2. direct endpoint Q -> decoder (direct endpoint prediction)
3. autonomous recursive Q -> decoder (full generative system)

Main experiments use shared H->Q projection and the same decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from protein_jepa_frame.models.stage_a.model import StageAJEPA
from protein_jepa_frame.models.stage_b.model import StageBLASP
from protein_jepa_frame.models.stage_c.structure_decoder import (
    StructureDecoder,
    DecoderConfig,
)
from protein_jepa_frame.models.stage_c.direct_head import (
    DirectEndpointHead,
    DirectHeadConfig,
)
from protein_jepa_frame.models.stage_c.rollout import (
    AutonomousRollout,
    RolloutConfig,
)
from protein_jepa_frame.models.stage_c.geometry import RigidBodyFrame


class StageCModel(nn.Module):
    """Complete Stage C pipeline.

    Freezes Stage A and B encoders; only trains decoder heads.
    Supports three prediction modes:
      - oracle:   ground-truth future Q (through Stage B encoder) -> decoder
      - direct:   DirectEndpointHead predicts Q at tau -> decoder
      - recursive: AutonomousRollout advances Q through time -> decoder

    Gate 4 invariant: all modes share the same H->Q projection and decoder.
    """

    def __init__(
        self,
        stage_a: StageAJEPA | None = None,
        stage_b: StageBLASP | None = None,
        decoder_cfg: DecoderConfig | None = None,
        direct_cfg: DirectHeadConfig | None = None,
        rollout_cfg: RolloutConfig | None = None,
    ):
        super().__init__()

        # Frozen encoders
        self.stage_a = stage_a
        self.stage_b = stage_b

        # Trainable heads
        self.decoder = StructureDecoder(decoder_cfg)
        self.direct_head = DirectEndpointHead(direct_cfg)
        self.rollout = AutonomousRollout(stage_b, rollout_cfg)

    def encode_context(self, batch: dict) -> dict[str, torch.Tensor]:
        """Run frozen Stage B encoder to get canonical context features (H, Q).

        CANONICAL: uses encode_state with use_temporal=True so H and Q
        include temporal-history conditioning, matching training and
        evaluation paths.

        Returns:
            dict with h_frozen, aatype, residue_index, chain_id,
                  backbone_coords, residue_mask, h_sequence, q_context
        """
        coords = batch.get("context_coords", batch.get("coords"))
        residue_mask = batch.get("context_residue_mask", batch.get("residue_mask"))
        aatype = batch.get("context_aatype", batch.get("aatype"))
        residue_index = batch.get("context_residue_index", batch.get("residue_index"))
        chain_id = batch.get("context_chain_id", batch.get("chain_id"))
        if coords is None or residue_mask is None:
            raise KeyError("StageCModel requires context coordinates and residue mask")

        time_ps = batch.get("context_time_ps")
        temperature_k = batch.get("context_temperature_k", batch.get("temperature_k"))

        with torch.no_grad():
            if self.stage_b is not None:
                state = self.stage_b.encode_state(
                    coords, residue_mask, aatype, residue_index, chain_id,
                    time_ps=time_ps,
                    temperature_k=temperature_k,
                    use_temporal=True,
                    return_bath=False,
                    return_ca_distances=False,
                    return_forces=False,
                )
                h = state["h_last"]
                q_context = state["q"]
                # Keep h_sequence for backward compatibility (best-effort)
                h_sequence = None
            elif self.stage_a is not None:
                h_sequence = self.stage_a.context_encoder(
                    coords=coords,
                    residue_mask=residue_mask,
                    aatype=aatype,
                    residue_index=residue_index,
                    chain_id=chain_id,
                    token_mask=residue_mask,
                )
                h = h_sequence[:, -1]
                q_context = None
            else:
                B, _, L = coords.shape[:3]
                context_dim = self.decoder.config.context_dim
                h = torch.zeros(B, L, context_dim, device=coords.device)
                q_context = torch.zeros(B, L, self.decoder.config.q_dim, device=coords.device)
                h_sequence = None

        return {
            "h_frozen": h,
            "aatype": aatype,
            "residue_index": residue_index,
            "chain_id": chain_id,
            "backbone_coords": coords,
            "residue_mask": residue_mask,
            "h_sequence": h_sequence,
            "q_context": q_context,
        }

    def extract_oracle_q(
        self,
        batch: dict,
        context: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Extract oracle Q by encoding the target future through Stage B.

        Gate 4.1: Uses the frozen Stage B encoder to compute Q_target from the
        ground-truth future coordinates. This is the decoder's theoretical upper
        bound — it receives perfect dynamics information.

        CANONICAL: uses encode_state with use_temporal=True so oracle Q
        and context Q are in the same representation space.

        The oracle pipeline:
          future_coords -> Stage B encode_state(use_temporal=True) -> Q_oracle

        Args:
            batch: data batch with future_coords
            context: output from encode_context() (optional, used for topology)

        Returns:
            q_oracle: [B, L, 8] ground-truth Q at target time
        """
        if self.stage_b is None:
            if context is not None:
                B, L = context["h_frozen"].shape[:2]
                return torch.zeros(
                    B, L, self.decoder.config.q_dim,
                    device=context["h_frozen"].device,
                )
            raise ValueError("Cannot extract oracle Q: no Stage B and no context")

        fut_coords = batch.get("future_coords")
        if fut_coords is None:
            raise KeyError("oracle Q extraction requires future_coords in batch")

        aatype = batch.get("future_aatype", batch.get("context_aatype", batch.get("aatype")))
        residue_index = batch.get("future_residue_index", batch.get("context_residue_index", batch.get("residue_index")))
        chain_id = batch.get("future_chain_id", batch.get("context_chain_id", batch.get("chain_id")))
        fut_mask = batch.get("future_residue_mask", batch.get("residue_mask"))
        fut_time_ps = batch.get("future_time_ps")
        temperature_k = batch.get("context_temperature_k", batch.get("temperature_k"))

        if aatype is None:
            L = fut_coords.shape[2]
            aatype = torch.zeros(L, dtype=torch.long, device=fut_coords.device)
        if residue_index is None:
            L = fut_coords.shape[2]
            residue_index = torch.arange(L, device=fut_coords.device)
        if chain_id is None:
            L = fut_coords.shape[2]
            chain_id = torch.zeros(L, dtype=torch.long, device=fut_coords.device)

        with torch.no_grad():
            # CANONICAL: encode future through the same temporal path as context
            fut_state = self.stage_b.encode_state(
                fut_coords, fut_mask, aatype, residue_index, chain_id,
                time_ps=fut_time_ps,
                temperature_k=temperature_k,
                use_temporal=True,
                return_bath=False,
                return_ca_distances=False,
                return_forces=False,
            )
            q_oracle = fut_state["q"]

        return q_oracle

    def _decode_with_anchor(
        self,
        q_dynamics: torch.Tensor,
        context: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Decode structure from dynamics Q using shared decoder.

        Args:
            q_dynamics: [B, L, 8] dynamics variables (from any source)
            context: output from encode_context()

        Returns:
            decoder output dict
        """
        h = context["h_frozen"]
        aatype = context["aatype"]
        bb = context["backbone_coords"][:, -1]
        mask = context["residue_mask"]
        if mask.dim() > 2:
            mask = mask[:, -1]

        if aatype.dim() == 1:
            aatype = aatype.unsqueeze(0).expand(h.shape[0], -1)

        R_anchor, t_anchor = RigidBodyFrame.from_backbone(
            bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]
        )

        return self.decoder(aatype, h, q_dynamics, R_anchor, t_anchor, mask)

    def predict_oracle(
        self,
        context: dict[str, torch.Tensor],
        q_oracle: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Oracle prediction: decode from ground-truth Q (decoder upper bound).

        Gate 4.1: This uses the oracle Q extracted from the target future
        via the frozen Stage B encoder. The decoder receives perfect dynamics
        information, establishing its performance ceiling.

        Args:
            context: output from encode_context()
            q_oracle: [B, L, 8] ground-truth Q at target time

        Returns:
            dict with decoded structure
        """
        return self._decode_with_anchor(q_oracle, context)

    def predict_direct(
        self,
        context: dict[str, torch.Tensor],
        tau_ps: torch.Tensor,
        temperature_k: torch.Tensor,
        num_samples: int = 1,
    ) -> list[dict[str, torch.Tensor]]:
        """Direct endpoint prediction at horizon tau.

        Args:
            context: output from encode_context()
            tau_ps: [B] horizon times in picoseconds
            temperature_k: [B] temperatures
            num_samples: number of structure samples

        Returns:
            list of dicts with decoded structures
        """
        h = context["h_frozen"]
        aatype = context["aatype"]
        bb = context["backbone_coords"][:, -1]  # last frame
        mask = context["residue_mask"][:, -1] if context["residue_mask"].dim() > 2 else context["residue_mask"]

        # Expand 1-D topology to [B, L]
        if aatype.dim() == 1:
            aatype = aatype.unsqueeze(0).expand(h.shape[0], -1)

        R_anchor, t_anchor = RigidBodyFrame.from_backbone(
            bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]
        )

        q_samples = self.direct_head.sample(
            h, None, tau_ps, temperature_k, mask, num_samples=num_samples,
        )
        results = []
        for sample_index in range(num_samples):
            outputs = self.decoder(
                aatype, h, q_samples[:, sample_index], R_anchor, t_anchor, mask,
            )
            results.append(outputs)

        return results

    def predict_recursive(
        self,
        context: dict[str, torch.Tensor],
        horizon_ps: float,
        temperature_k: torch.Tensor,
        delta_t_ps: float = 100.0,
    ) -> dict[str, torch.Tensor]:
        """Autonomous recursive rollout to horizon.

        Args:
            context: output from encode_context()
            horizon_ps: target horizon in ps
            temperature_k: [B]
            delta_t_ps: integration step

        Returns:
            dict with decoded structure at horizon
        """
        h = context["h_frozen"]
        aatype = context["aatype"]
        bb = context["backbone_coords"][:, -1]
        mask = context["residue_mask"][:, -1] if context["residue_mask"].dim() > 2 else context["residue_mask"]

        # Expand 1-D topology to [B, L]
        if aatype.dim() == 1:
            aatype = aatype.unsqueeze(0).expand(h.shape[0], -1)

        R_anchor, t_anchor = RigidBodyFrame.from_backbone(
            bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]
        )

        # Use canonical Q from context encoding (already computed via encode_state)
        q_context = context.get("q_context")
        if q_context is not None:
            q_initial = q_context
            bath_initial = None  # Bath is optional in rollout
        elif self.stage_b is not None:
            with torch.no_grad():
                q_initial = self.stage_b.project_to_q(h)
                bath_initial = self.stage_b.initialize_bath(h, mask)
        else:
            B, L = h.shape[:2]
            q_initial = torch.zeros(B, L, 8, device=h.device)
            bath_initial = None

        ca_distances = torch.cdist(bb[..., 1, :], bb[..., 1, :])

        # Rollout
        state = self.rollout.rollout_to_horizon(
            h, q_initial, horizon_ps,
            static_condition=None,
            temperature_k=temperature_k,
            bath_initial=bath_initial,
            residue_mask=mask,
            ca_distances=ca_distances,
            delta_t_ps=delta_t_ps,
        )

        # Decode
        q_final = state["Q"]
        outputs = self.decoder(aatype, h, q_final, R_anchor, t_anchor, mask)

        return outputs

    def forward(
        self,
        batch: dict,
        mode: str = "direct",
        tau_ps: torch.Tensor | None = None,
        temperature_k: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Unified forward pass supporting all three prediction modes.

        Args:
            batch: data batch with coordinates, masks, etc.
            mode: "oracle", "direct", or "recursive"
            tau_ps: horizon time for direct mode [B]
            temperature_k: temperature [B]

        Returns:
            structure outputs dict
        """
        context = self.encode_context(batch)

        if temperature_k is None:
            temperature_k = batch.get("context_temperature_k", batch.get("temperature_k"))
            if temperature_k is None:
                temperature_k = torch.tensor([348.0], device=context["h_frozen"].device)

        if mode == "oracle":
            q_oracle = self.extract_oracle_q(batch, context)
            return self.predict_oracle(context, q_oracle)
        elif mode == "direct":
            if tau_ps is None:
                tau_ps = batch.get("horizon_time_ps",
                                   torch.tensor([1000.0], device=context["h_frozen"].device))
            results = self.predict_direct(context, tau_ps, temperature_k)
            return results[0] if results else {}
        elif mode == "recursive":
            horizon_ps = float(tau_ps.mean().item()) if tau_ps is not None else 1000.0
            return self.predict_recursive(context, horizon_ps, temperature_k)
        else:
            raise ValueError(f"Unknown prediction mode: {mode}")


__all__ = ["StageCModel"]
