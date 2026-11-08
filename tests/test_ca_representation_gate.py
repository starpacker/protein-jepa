import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.gate_ca_representation import (
    CONDITIONS, assess_rows, audit_sources, readout_metrics, score_window,
)


def _rows():
    rows = []
    for protein in range(4):
        for anchor in range(protein + 1):
            conditions = {}
            for condition in CONDITIONS:
                error = .05 * (protein + 1) if condition == 'true_future_H' else 1.
                conditions[condition] = dict(endpoint_rmsd=error,
                    nonlocal_distance_change_rmse=error, ca_bond=.01)
            rows.append(dict(protein=str(protein), trajectory='r0', temperature_k=320.,
                             anchor_frame=anchor, horizon=1, conditions=conditions))
    return rows


def test_gate_uses_equal_protein_weights_and_deterministic_paired_intervals():
    rows = _rows()
    result = assess_rows(rows, bootstrap_reps=300)
    assert result['passed']
    horizon = result['by_horizon']['1']
    assert horizon['metrics']['true_future_H']['endpoint_rmsd']['mean'] == pytest.approx(.125)
    assert result == assess_rows(rows[::-1], bootstrap_reps=300)
    # More observations of one protein do not change its total weight.
    extra = copy.deepcopy(rows[:1])
    extra[0]['anchor_frame'] = 99
    expanded = assess_rows(rows + extra, bootstrap_reps=300)
    assert expanded['by_horizon']['1']['metrics'] == horizon['metrics']
    assert expanded['by_horizon']['1']['paired_true_minus_control'] == horizon['paired_true_minus_control']


def test_gate_rejects_duplicate_or_nonfinite_rows_and_uninformative_H():
    rows = _rows()
    with pytest.raises(ValueError, match='duplicate'):
        assess_rows(rows + [rows[0]])
    rows[0]['conditions']['true_future_H']['endpoint_rmsd'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        assess_rows(rows)
    rows = _rows()
    for row in rows:
        row['conditions']['wrong_time_H'] = copy.deepcopy(row['conditions']['true_future_H'])
    report = assess_rows(rows, bootstrap_reps=300)
    assert not report['passed']
    assert not report['by_horizon']['1']['checks']['paired_endpoint_improvement']


def _path():
    anchor = torch.tensor([[0., 0., 0.], [3.8, 0., 0.], [6., 3., 0.],
                           [5., 6., 2.], [2., 8., 1.], [-1., 6., 2.]])
    change = torch.tensor([[0., 0., 0.], [0., .2, .1], [0., .8, -.1],
                           [-.5, 0., .4], [0., -.8, -.4], [.3, 0., 0.]])
    return anchor[None] + torch.arange(4.)[:, None, None] * change[None]


def test_coordinate_gate_uses_common_anchor_gauge_and_reports_shape():
    truth = _path()
    mask, ri, ci = torch.ones(6, dtype=torch.bool), torch.arange(6), torch.zeros(6, dtype=torch.long)
    from evaluation.ensemble_metrics_ext import _align_to
    perfect = _align_to(truth, truth[0], mask)
    metrics = readout_metrics(perfect, truth, truth[0], mask, ri, ci)
    assert metrics['endpoint_rmsd'] < 1e-5
    assert metrics['displacement_rmse'] < 1e-5
    assert metrics['nonlocal_distance_change_rmse'] < 1e-5
    assert metrics['rmsf_corr'] > .999
    shifted = readout_metrics(perfect + torch.tensor([4., 0., 0.]), truth, truth[0], mask, ri, ci)
    assert shifted['endpoint_rmsd'] == pytest.approx(4., abs=1e-5)
    assert shifted['endpoint_shape_rmsd'] < 1e-5
    assert shifted['nonlocal_distance_change_rmse'] < 1e-5


def test_nonlocal_pairs_follow_residue_ids_and_chains_not_array_offsets():
    truth = _path()
    mask = torch.tensor([True, False, False, False, False, True])
    # The minimum-residue guard is deliberate even with one valid nonlocal pair.
    with pytest.raises(ValueError, match='valid residues'):
        readout_metrics(truth, truth, truth[0], mask, torch.arange(6), torch.zeros(6))
    mask[:] = True
    # Repeated residue IDs in one chain must not become nonlocal from array positions.
    with pytest.raises(ValueError, match='no nonlocal pairs'):
        readout_metrics(truth, truth, truth[0], mask, torch.zeros(6), torch.zeros(6))


def test_score_window_keeps_anchor_and_wrong_time_conditions_observation_only():
    path = _path()
    backbone = path[:, :, None].expand(-1, -1, 4, -1).clone()
    backbone[:, :, 0, 0] -= 1.
    backbone[:, :, 2, 1] += 1.
    context = backbone[:1].expand(8, -1, -1, -1).clone()
    context[0, :, :, 2] += .2
    meta = dict(residue_mask=torch.ones(8, 6, dtype=torch.bool),
                aatype=torch.zeros(6, dtype=torch.long), residue_index=torch.arange(6),
                chain_id=torch.zeros(6, dtype=torch.long))
    window = SimpleNamespace(context=dict(coords=context, **meta),
        future_path=dict(coords=backbone, residue_mask=torch.ones(4, 6, dtype=torch.bool)),
        protein_id='p', trajectory_id='r', temperature_k=320., anchor_frame=7,
        target_frame=10, horizon=3, horizon_ps=3000., context_frames=np.arange(8))

    class Bundle:
        def __init__(self):
            self.decoded_h = []
        def encode(self, coords, *args):
            return coords[:, :, 1]
        def decode_ca(self, h, anchor_h, aa, ri, ci, rotation, anchor, mask):
            self.decoded_h.append((h.clone(), anchor_h.clone(), anchor.clone()))
            return h

    bundle = Bundle()
    row = score_window(bundle, window, 'cpu')
    assert row['path_frames'] == 4
    assert torch.equal(bundle.decoded_h[1][0], context[-1:, :, 1])
    assert torch.equal(bundle.decoded_h[2][0], context[:1, :, 1])
    assert all(torch.equal(anchor[0], context[-1, :, 1]) for _, _, anchor in bundle.decoded_h)


def test_upstream_training_protein_overlap_fails_even_if_declared_main_split_is_clean(tmp_path):
    def manifest(name, protein):
        path = tmp_path / name
        path.write_text(json.dumps(dict(npz_path=str(tmp_path / (protein + '.npz')),
                                        protein_id=protein, temperature_k=320.)) + '\n')
        return str(path)
    train = manifest('train.jsonl', 'train')
    val = manifest('val.jsonl', 'val')
    source = tmp_path / 'stage_a.pt'
    torch.save(dict(train_args=dict(manifest=val)), source)
    component = dict(calibration_args=dict(manifest=train, stage_a_ckpt=str(source), b1a_ckpt=str(source)))
    with pytest.raises(ValueError, match='protein overlap'):
        audit_sources(component, component, train, val)
