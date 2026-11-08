"""Observed-only endpoint latent prediction for the versioned Stage A candidate.

This helper consumes eight native observed frames and known endpoint metadata.
It does not accept future coordinates, encode a target, call the EMA teacher,
align/rebase frames, or normalize H. The output uses the unchanged StageAV2
online-anchor plus predicted-factor composition. Predictions at padded rows
remain the predictor's original values; only valid_mask defines readout support.
"""
from __future__ import annotations

import torch

from protein_jepa_frame.models.stage_a_relpos_v1 import StageARelposV1, _feature_mode, _require

VERSION = 'stage_a_relpos_prediction_v1'
HISTORY_FRAMES = 8
FRAME_INTERVAL_PS = 1000
HORIZONS = (1, 4, 16, 64)


def _tensor(value, shape, dtype, name):
    _require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape)
             and value.device.type == 'cpu' and value.dtype == dtype,
             name+' has invalid shape/device/dtype')
    _require(bool(torch.isfinite(value).all()), name+' must be finite, including padding')


@torch.no_grad()
def predict_endpoint(model, *, feature_mode, history_coords, history_mask,
                     aatype, residue_index, chain_id, history_time_ps,
                     endpoint_time_ps, horizon_frames, temperature_k):
    """Predict raw endpoint H from observed history, without teacher execution.

    Args: coords[B,8,L,4,3] CPU float32, fixed mask[B,8,L] bool,
    legacy IDs[B,L] int64, actual history times[B,8] and endpoint times[B]
    CPU float32/64 of matching dtype, horizon_frames[B] int64 in1/4/16/64,
    and temperature_k[B] CPU float32. Model must already be entirely eval and
    CPU float32. Parameters need not be frozen; this call has no autograd graph.
    """
    _require(isinstance(model, StageARelposV1), 'explicit versioned Stage A model required')
    _require(_feature_mode(feature_mode) == model.feature_mode, 'feature mode identity differs')
    model._runtime_contract()
    _require(all(not module.training for module in model.modules()), 'entire model must be eval')
    _require(not torch.is_autocast_enabled('cpu'), 'CPU autocast is not part of FP32 prediction')
    _require(all(t.device.type == 'cpu' and (not t.is_floating_point() or t.dtype == torch.float32)
                 for t in list(model.parameters())+list(model.buffers())), 'model must be CPU float32')
    _require(isinstance(history_coords, torch.Tensor) and history_coords.ndim == 5
             and history_coords.shape[1] == HISTORY_FRAMES and history_coords.shape[-2:] == (4,3),
             'history_coords must be [B,8,L,4,3]')
    batch, _, length, _, _ = history_coords.shape
    _require(batch > 0 and length > 0, 'nonempty batch and residue dimension required')
    _tensor(history_coords, (batch, HISTORY_FRAMES, length, 4, 3), torch.float32, 'history_coords')
    _tensor(history_mask, (batch, HISTORY_FRAMES, length), torch.bool, 'history_mask')
    valid = history_mask[:, 0]
    _require(torch.equal(history_mask, valid[:, None].expand_as(history_mask)) and bool(valid.any(-1).all()),
             'fixed nonempty observed residue mask required across all eight frames')
    for name, value in [('aatype', aatype), ('residue_index', residue_index), ('chain_id', chain_id)]:
        _tensor(value, (batch, length), torch.int64, name)
    _require(bool(((aatype >= 0) & (aatype <= 20)).all()), 'aatype must use legacy 0..20 embedding IDs')
    _require(isinstance(history_time_ps, torch.Tensor) and history_time_ps.dtype in (torch.float32, torch.float64),
             'history_time_ps must be float32 or float64 metadata')
    _tensor(history_time_ps, (batch, HISTORY_FRAMES), history_time_ps.dtype, 'history_time_ps')
    _tensor(endpoint_time_ps, (batch,), history_time_ps.dtype, 'endpoint_time_ps')
    _tensor(horizon_frames, (batch,), torch.int64, 'horizon_frames')
    _tensor(temperature_k, (batch,), torch.float32, 'temperature_k')
    _require(bool((temperature_k > 0).all()), 'temperature must be positive')
    _require(all(int(value) in HORIZONS for value in horizon_frames), 'unsupported endpoint horizon')
    times = history_time_ps.double()
    _require(bool(((times[:, 1:]-times[:, :-1]) == FRAME_INTERVAL_PS).all()),
             'history times must increase by exactly 1000 ps')
    lag64 = endpoint_time_ps.double()-times[:, -1]
    _require(bool((lag64 > 0).all()) and torch.equal(lag64, horizon_frames.double()*FRAME_INTERVAL_PS),
             'endpoint must match the declared positive horizon on the 1000 ps grid')
    lag_ps = lag64.float()
    rng_before = torch.get_rng_state().clone()
    context_H = model.encode_temporal(coords=history_coords, residue_mask=history_mask,
        aatype=aatype, residue_index=residue_index, chain_id=chain_id)
    observed = context_H[:, -1]
    pred = model.predictor(context_H, lag_ps, temperature_k, valid_mask=valid, task_id=1)
    anchor_factors = model.opf.decompose(observed)
    predicted = model.opf.compose(anchor_factors+pred)
    dim = model.config.latent_dim
    _tensor(context_H, (batch, HISTORY_FRAMES, length, dim), torch.float32, 'context_H')
    _tensor(predicted, (batch, length, dim), torch.float32, 'H_predicted')
    expected_factors = (batch, length, model.config.num_factors, dim//model.config.num_factors)
    _tensor(pred, expected_factors, torch.float32, 'pred_factors')
    _tensor(anchor_factors, expected_factors, torch.float32, 'anchor_factors')
    _require(torch.equal(rng_before, torch.get_rng_state()), 'observed-only eval prediction unexpectedly consumed RNG')
    model._runtime_contract()
    audit = dict(version=VERSION, feature_mode=feature_mode, encoder='online/context_encoder',
        history_frames=HISTORY_FRAMES, history_interval_ps=FRAME_INTERVAL_PS,
        horizon_frames=horizon_frames.tolist(), anchor_time_ps=history_time_ps[:, -1].tolist(),
        endpoint_time_ps=endpoint_time_ps.tolist(), lag_ps=lag_ps.tolist(), temperature_k=temperature_k.tolist(),
        valid_residues=valid.sum(-1).tolist(), latent_dim=dim,
        coordinate_frame='native_observed_history_unchanged', teacher_called=False,
        future_coordinates_used=False, opf_composition='compose(decompose(online_last_H)+predicted_delta_factors)',
        normalization='none', rebasing='none', padded_predictions='unchanged_predictor_output; use valid_mask',
        evaluation_mode=True, autograd_enabled=False, cpu_rng_unchanged=True)
    return dict(H_observed=observed, H_predicted=predicted, context_H=context_H,
        anchor_factors=anchor_factors, pred_factors=pred, valid_mask=valid,
        lag_ps=lag_ps, audit=audit)
