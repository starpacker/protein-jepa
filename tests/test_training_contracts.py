import copy

import pytest
import torch

from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.protocol import (WindowProtocol, split_contiguous_window,
    validate_endpoint_times, validate_manifest_split)
from protein_jepa_frame.utils.contracts import (
    assert_same_representation, build_representation_contract,
    build_prediction_contract, validate_representation_contract,
    validate_prediction_contract,
)


def _batch():
    ds = build_dataset(DataConfig(num_frames=7, synthetic_residues=6, synthetic_size=2))
    return collate_clips([ds[0], ds[1]])


def test_contiguous_window_has_only_anchor_shared():
    batch = _batch()
    paired = split_contiguous_window(batch, 3, 4, step_ps=1.)
    assert paired['context_coords'].shape[1] == 3
    assert paired['future_coords'].shape[1] == 5
    assert torch.equal(paired['context_coords'][:, -1], paired['future_coords'][:, 0])
    for history, future in zip(paired['context_time_ps'], paired['future_time_ps']):
        assert set(history.tolist()).intersection(future.tolist()) == {float(history[-1])}
    assert torch.equal(paired['context_dt_ps'], torch.ones(2, 3))


def test_future_mutation_cannot_change_history():
    batch = _batch()
    a = split_contiguous_window(batch, 3, 4)
    modified = copy.deepcopy(batch)
    modified['coords'][:, 3:] += 999.
    b = split_contiguous_window(modified, 3, 4)
    assert torch.equal(a['context_coords'], b['context_coords'])
    assert not torch.equal(a['future_coords'][:, 1:], b['future_coords'][:, 1:])


def test_window_rejects_time_mismatch_and_leakage():
    with pytest.raises(ValueError, match='step_ps'):
        split_contiguous_window(_batch(), 3, 4, step_ps=1000.)
    bad = _batch()
    bad['time_ps'][:, 4] = bad['time_ps'][:, 3]
    with pytest.raises(ValueError, match='increasing'):
        split_contiguous_window(bad, 3, 4)
    with pytest.raises(ValueError, match='no leakage'):
        validate_endpoint_times(torch.tensor([[0.,1.,2.]]), torch.tensor([[2.,3.]]))
    with pytest.raises(ValueError, match='exactly'):
        split_contiguous_window(_batch(), 8, 8)


def test_physical_gap_grid_is_explicit_and_positive():
    p = WindowProtocol()
    assert {1,2,4,8,16,24,32,48,64,96}.issubset(p.gaps_frames)
    with pytest.raises(ValueError, match='strictly after'):
        WindowProtocol(horizons_frames=(4,), temporal_lags=(4,))


def test_heldout_split_rejects_renamed_domains_and_source_files(tmp_path):
    import json
    train, val = tmp_path/'train.jsonl', tmp_path/'val.jsonl'
    row = dict(npz_path=str(tmp_path/'train.npz'), protein_id='a', domain_id='same')
    train.write_text(json.dumps(row)+'\n')
    val.write_text(json.dumps({**row, 'protein_id':'b', 'npz_path':str(tmp_path/'val.npz')})+'\n')
    with pytest.raises(ValueError, match='domain overlap'):
        validate_manifest_split(train, val)
    val.write_text(json.dumps({**row, 'protein_id':'b', 'domain_id':'other'})+'\n')
    with pytest.raises(ValueError, match='source-file overlap'):
        validate_manifest_split(train, val)


def _representation():
    encoder = {'w': torch.randn(3, 4)}
    normalizer = {'mean_': torch.randn(4), 'std_': torch.ones(4), 'fitted_': torch.tensor(True)}
    decoder = {'w': torch.randn(4, 3)}
    contract = build_representation_contract(encoder, {'d':4}, normalizer, decoder, {'hidden':4})
    return contract, encoder, normalizer, decoder


def test_representation_contract_rejects_same_shape_wrong_weights():
    contract, encoder, norm, dec = _representation()
    validate_representation_contract(contract, encoder_state=encoder, normalizer_state=norm,
                                     decoder_state=dec, encoder_config={'d':4}, decoder_config={'hidden':4})
    with pytest.raises(ValueError, match='encoder_sha256'):
        validate_representation_contract(contract, encoder_state={'w': encoder['w']+1})
    with pytest.raises(ValueError, match='normalizer_sha256'):
        validate_representation_contract(contract, normalizer_state={'mean_': torch.zeros(4)})
    altered = copy.deepcopy(contract)
    altered['decoder_input'] = 'normalized_H'
    with pytest.raises(ValueError):
        assert_same_representation(contract, altered)


def test_prediction_contract_rejects_wrong_encoder_or_legacy_semantics():
    state = {'w': torch.eye(3)}
    contract = build_prediction_contract(state, {'d':3})
    validate_prediction_contract(contract, encoder_state=state)
    with pytest.raises(ValueError, match='encoder_sha256'):
        validate_prediction_contract(contract, encoder_state={'w': torch.zeros(3,3)})
    with pytest.raises(ValueError, match='legacy'):
        validate_prediction_contract(None)
    with pytest.raises(ValueError, match='delta_factors'):
        build_prediction_contract(state, {}, predictor_output='absolute_factors')
