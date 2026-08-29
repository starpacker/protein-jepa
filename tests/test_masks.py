import torch

from protein_jepa_frame.config import MaskConfig
from protein_jepa_frame.data.masks import SpatiotemporalBlockMasker


def test_block_masks_are_valid_and_disjoint():
    residue_mask = torch.ones(3, 6, 20, dtype=torch.bool)
    residue_mask[1, :, 15:] = False
    masker = SpatiotemporalBlockMasker(MaskConfig(num_blocks=2, min_context_ratio=0.25))

    context, target = masker(residue_mask)

    assert context.shape == residue_mask.shape
    assert target.shape == residue_mask.shape
    assert torch.all(~(context & target))
    assert torch.all((context | target) <= residue_mask)
    assert torch.all(target.flatten(1).sum(dim=1) > 0)
    assert torch.all(context.flatten(1).sum(dim=1) > 0)


def test_future_masks_use_past_context_and_future_targets():
    residue_mask = torch.ones(2, 8, 12, dtype=torch.bool)
    masker = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.25, 0.25), future_context_ratio=0.5)
    )

    context, target = masker(residue_mask)

    assert context.shape == residue_mask.shape
    assert target.shape == residue_mask.shape
    assert torch.all(~(context & target))
    assert torch.all((context | target) <= residue_mask)
    assert torch.all(target.flatten(1).sum(dim=1) > 0)
    assert torch.all(context.flatten(1).sum(dim=1) > 0)
    for b in range(residue_mask.shape[0]):
        context_times = context[b].any(dim=1).nonzero(as_tuple=False).flatten()
        target_times = target[b].any(dim=1).nonzero(as_tuple=False).flatten()
        assert context_times.max() < target_times.min()


def test_future_masks_can_leave_gap_between_context_and_target():
    residue_mask = torch.ones(2, 12, 8, dtype=torch.bool)
    masker = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.25, 0.25), future_context_ratio=0.5, future_gap_frames=2)
    )

    context, target = masker(residue_mask)

    for b in range(residue_mask.shape[0]):
        context_times = context[b].any(dim=1).nonzero(as_tuple=False).flatten()
        target_times = target[b].any(dim=1).nonzero(as_tuple=False).flatten()
        assert int(target_times.min() - context_times.max()) >= 3


def test_future_masks_reserve_requested_prediction_horizon():
    residue_mask = torch.ones(4, 24, 8, dtype=torch.bool)
    masker = SpatiotemporalBlockMasker(
        MaskConfig(mode="future", temporal_scale=(0.25, 0.5), future_context_ratio=0.5, future_gap_frames=4),
        required_future_horizon=12,
    )

    context, _ = masker(residue_mask)

    for b in range(residue_mask.shape[0]):
        last_context = int(context[b].any(dim=1).nonzero(as_tuple=False).max())
        assert last_context + 12 < residue_mask.shape[1]
