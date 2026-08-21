from __future__ import annotations

import torch

from protein_jepa_frame.config import MaskConfig


class SpatiotemporalBlockMasker:
    """V-JEPA-style block masker over `[time, residue]` tokens."""

    def __init__(self, config: MaskConfig):
        self.config = config

    @staticmethod
    def _sample_span(size: int, scale: tuple[float, float], device: torch.device) -> int:
        lo, hi = scale
        ratio = torch.empty((), device=device).uniform_(float(lo), float(hi)).item()
        return max(1, min(size, int(round(size * ratio))))

    def __call__(self, residue_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if residue_mask.dim() != 3:
            raise ValueError("residue_mask must be [B,T,L]")
        if self.config.mode == "future":
            return self._future_only(residue_mask)
        if self.config.mode != "block":
            raise ValueError(f"unknown mask mode {self.config.mode}")
        B, T, L = residue_mask.shape
        device = residue_mask.device
        target = torch.zeros_like(residue_mask, dtype=torch.bool)
        for b in range(B):
            valid_res = residue_mask[b].any(dim=0).nonzero(as_tuple=False).flatten()
            if valid_res.numel() == 0:
                continue
            first = int(valid_res[0])
            last = int(valid_res[-1]) + 1
            valid_len = last - first
            for _ in range(self.config.num_blocks):
                t_span = self._sample_span(T, self.config.temporal_scale, device)
                r_span = self._sample_span(valid_len, self.config.residue_scale, device)
                t0 = int(torch.randint(0, T - t_span + 1, (), device=device))
                r0 = first + int(torch.randint(0, valid_len - r_span + 1, (), device=device))
                target[b, t0 : t0 + t_span, r0 : r0 + r_span] = True
            target[b] &= residue_mask[b]
            valid_count = int(residue_mask[b].sum().item())
            context_count = int((residue_mask[b] & ~target[b]).sum().item())
            min_context = max(1, int(round(valid_count * self.config.min_context_ratio)))
            if context_count < min_context:
                flat = target[b].flatten().nonzero(as_tuple=False).flatten()
                release = min(flat.numel(), min_context - context_count)
                if release > 0:
                    order = torch.randperm(flat.numel(), device=device)[:release]
                    target[b].flatten()[flat[order]] = False
        context = residue_mask & ~target
        empty_target = (target.flatten(1).sum(dim=1) == 0).nonzero(as_tuple=False).flatten()
        for b in empty_target.tolist():
            valid = residue_mask[b].flatten().nonzero(as_tuple=False).flatten()
            if valid.numel() > 1:
                picked = valid[torch.randint(0, valid.numel(), (), device=device)]
                target[b].flatten()[picked] = True
                context[b].flatten()[picked] = False
        return context, target

    def _future_only(self, residue_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = residue_mask.shape
        device = residue_mask.device
        context = torch.zeros_like(residue_mask, dtype=torch.bool)
        target = torch.zeros_like(residue_mask, dtype=torch.bool)
        min_context_frames = max(1, int(round(T * float(self.config.future_context_ratio))))
        min_context_frames = min(min_context_frames, max(1, T - 1))
        for b in range(B):
            if residue_mask[b].sum() == 0:
                continue
            t_span = self._sample_span(T - min_context_frames, self.config.temporal_scale, device)
            max_start = max(min_context_frames, T - t_span)
            if max_start <= min_context_frames:
                t0 = min_context_frames
            else:
                t0 = int(torch.randint(min_context_frames, max_start + 1, (), device=device))
            context[b, :t0] = residue_mask[b, :t0]
            target[b, t0 : t0 + t_span] = residue_mask[b, t0 : t0 + t_span]
            if target[b].sum() == 0 and t0 < T:
                target[b, t0:] = residue_mask[b, t0:]
        empty_target = (target.flatten(1).sum(dim=1) == 0).nonzero(as_tuple=False).flatten()
        for b in empty_target.tolist():
            valid = residue_mask[b].flatten().nonzero(as_tuple=False).flatten()
            if valid.numel() > 1:
                picked = valid[-1]
                target[b].flatten()[picked] = True
                context[b].flatten()[picked] = False
        return context, target
