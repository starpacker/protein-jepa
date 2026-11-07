"""Pure matched Stage A readouts and descriptive paired TRAIN-heldout summaries.

No file IO, checkpoint loading or training-data selection. The caller verifies
the frozen support/cache and enforces all adaptations/head fits before scoring.
Cached H is never consumed; a caller loading a whole cache may deserialize it.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from protein_jepa_frame.data import frozen_h_distance_support_v2 as support_api
from protein_jepa_frame.models.stage_a_relpos_v1 import StageARelposV1
from protein_jepa_frame.models.stage_c import frozen_h_distance_probe_v2 as probe_api
from protein_jepa_frame.utils.contracts import state_sha256

VERSION = 'stage_a_relpos_readout_v1'
ARMS = ('pca', 'zero', 'local')
HORIZONS = (1, 4, 16, 64)
TRAIN_ORDER = tuple(range(400)) + tuple(range(100))
ROLES = ('future_H', 'observed_H_substitution', 'predicted_H')
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261102
METRICS = ('log_mse', 'mae_angstrom', 'mse_angstrom2', 'rmse_angstrom')
LIMITS = ('Fixed inspected TRAIN panel; held out from adaptation and fresh head fitting, '
          'not unseen pretraining proteins or a blinded test. One adaptation seed. '
          'Stratified paired intervals are descriptive conditional on these fitted models; '
          'no seed uncertainty, multiplicity-adjusted significance, selection or adoption.')
require = probe_api.require


def _cpu_contract():
    require(torch.get_default_device().type == 'cpu' and torch.get_default_dtype() == torch.float32
            and not torch.is_autocast_enabled('cpu'), 'readout requires default CPU/float32 and no autocast')


def _snapshot(mapping):
    return {key: value.detach().cpu().clone() for key, value in mapping.items()}


def _feature_hash(features):
    return state_sha256({f'{i}/{key}': f[key] for i, f in enumerate(features)
                         for key in ('observed_H', 'target_H')})


def _features(features, prepared):
    require(len(features) == len(prepared), 'feature/row count differs')
    for feature, item in zip(features, prepared):
        for key in ('observed_H', 'target_H'):
            H = feature[key]
            require(isinstance(H, torch.Tensor) and H.shape == (1, len(item['mask']), 128)
                    and H.dtype == torch.float32 and H.device.type == 'cpu' and not H.requires_grad
                    and bool(torch.isfinite(H).all()) and bool((H[:, ~item['mask']] == 0).all()),
                    'features must be detached finite CPU float32 with zero padding')


def _input_hash(prepared):
    # Deliberately exclude cached oracle_H/anchor_H and unrelated cache fields.
    values = {}
    for i, item in enumerate(prepared):
        for key in ('observed_backbone', 'target_backbone', 'observed_mask', 'atom_mask',
                    'target_atom_mask', 'embedding_aatype', 'embedding_residue_index', 'embedding_chain_id'):
            values[f'{i}/row/{key}'] = item['row'][key]
        values[f'{i}/metadata'] = item['row']['metadata']
        values[f'{i}/pairs'] = item['pairs']
        for key, value in item['labels'].items():
            values[f'{i}/label/{key}'] = value
    return state_sha256(values)


def prepare_readout_rows(cache_rows, support):
    """Validate all500 registered rows using only consumed geometry/metadata."""
    require(len(cache_rows) == 500 and len(support['plans']) == 500, 'requires complete500 panel')
    require(len({r['metadata']['protein'] for r in cache_rows}) == 500
            and [r['metadata']['horizon'] for r in cache_rows] == list(HORIZONS)*125,
            'fixed500 protein/horizon order differs')
    prepared = []
    for index, row in enumerate(cache_rows):
        physical = support_api.attach_cached_physical_support(row, index, support)
        pairs, mask = physical['pair_indices'], physical['observed_mask']
        require(torch.equal(mask, row['observed_mask']), 'fixed observed mask changed')
        expected_split = 'fit' if index < 400 else 'readout_heldout'
        require(physical['audit']['split'] == expected_split, 'fixed400/100 split differs')
        labels = probe_api.pair_log_targets(row, pairs)
        prepared.append(dict(cache_row_index=index, row=row, pairs=pairs, mask=mask, labels=labels,
            physical_support_audit=copy.deepcopy(physical['audit']),
            labels_sha256=state_sha256(dict(pair_indices=pairs, **labels))))
    return prepared


@torch.no_grad()
def encode_readout_rows(model, prepared_rows):
    """Use frozen terminal ONLINE encoder; preserve every model state/buffer."""
    _cpu_contract()
    require(isinstance(model, StageARelposV1), 'explicit versioned Stage A model required')
    model._runtime_contract()
    require(not any(m.training for m in model.modules())
            and not any(p.requires_grad for p in model.parameters()), 'model must be fully frozen/eval')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()),
            'CPU float32 readout encoding required')
    require(all(b.device.type == 'cpu' and (not b.is_floating_point() or b.dtype == torch.float32)
                for b in model.buffers()), 'CPU float32 model buffers required')
    before = (state_sha256(model), state_sha256(dict(model.named_buffers())), _input_hash(prepared_rows))
    rng = torch.get_rng_state().clone()
    features = []
    for item in prepared_rows:
        observed, target = probe_api.encode_pair(model.select_readout_encoder('online'), item['row'])
        features.append(dict(observed_H=observed, target_H=target))
    after = (state_sha256(model), state_sha256(dict(model.named_buffers())), _input_hash(prepared_rows))
    require(before == after and torch.equal(rng, torch.get_rng_state()), 'encoding mutated state/buffers/inputs/RNG')
    return features, dict(feature_mode=model.feature_mode, readout_encoder='online',
        rows=len(features), model_state_sha256=before[0], named_buffers_sha256=before[1],
        consumed_input_sha256=before[2], features_sha256=_feature_hash(features),
        model_unchanged=True, buffers_unchanged=True, cpu_rng_unchanged=True,
        cached_H_consumed=False, dtype='float32', device='cpu')


def fit_readout(prepared400, features400, emit):
    """Fresh identical500-update head; emit detached JSON records; no writes.

    Returns (head, audit, payload). On failure the exception carries
    ``readout_failure_payload`` with the exact completed-update count and state.
    """
    _cpu_contract()
    require(len(prepared400) == len(features400) == 400 and callable(emit), 'fit requires exactly400 rows/features and emit')
    require([x['cache_row_index'] for x in prepared400] == list(range(400))
            and len({x['row']['metadata']['protein'] for x in prepared400}) == 400
            and [x['row']['metadata']['horizon'] for x in prepared400] == list(HORIZONS)*100
            and all(x['physical_support_audit']['split'] == 'fit' for x in prepared400),
            'fit order/split differs')
    _features(features400, prepared400)
    before_inputs, before_features = _input_hash(prepared400), _feature_hash(features400)
    rng = torch.get_rng_state().clone()
    head = probe_api.new_probe(seed=42).train()
    initial = _snapshot(head.state_dict()); initial_sha = state_sha256(initial)
    optimizer = torch.optim.Adam(head.parameters(), lr=.001, weight_decay=0.)
    completed = 0
    try:
        for step, index in enumerate(TRAIN_ORDER, 1):
            item = prepared400[index]
            optimizer.zero_grad(set_to_none=True)
            loss, prediction = probe_api.probe_loss(head, features400[index]['target_H'],
                item['mask'], item['pairs'], item['labels']['target_log'])
            require(loss.ndim == 0 and loss.dtype == torch.float32 and bool(torch.isfinite(loss)), 'invalid readout loss')
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()),
                    'missing/nonfinite readout gradient')
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            require(bool(torch.isfinite(norm)), 'nonfinite gradient norm')
            optimizer.step(); completed = step
            require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), 'nonfinite readout update')
            emit(dict(step=step, cache_row_index=index, protein=item['row']['metadata']['protein'],
                horizon=item['row']['metadata']['horizon'], n_pairs=len(item['pairs']),
                log_mse=float(loss.detach()), gradient_norm_before_clip=float(norm), lr=.001,
                labels_sha256=item['labels_sha256']))
            del loss, prediction, norm
        require(completed == 500 and len(optimizer.state) == len(list(head.parameters()))
                and all(float(s['step']) == 500 for s in optimizer.state.values()), 'optimizer inventory/step differs')
        require(before_inputs == _input_hash(prepared400) and before_features == _feature_hash(features400),
                'fit mutated fixed inputs/features')
        require(torch.equal(rng, torch.get_rng_state()), 'deterministic readout fit changed CPU RNG')
        head.eval()
        payload = dict(schema_version=VERSION, step=500, initial_head=initial,
            initial_head_sha256=initial_sha, head=_snapshot(head.state_dict()), head_sha256=state_sha256(head),
            optimizer=copy.deepcopy(optimizer.state_dict()), training_order=list(TRAIN_ORDER), selection='latest500 only')
        audit = dict(initial_head_sha256=initial_sha, final_head_sha256=payload['head_sha256'],
            optimizer_steps=500, fit_input_sha256=before_inputs, fit_features_sha256=before_features,
            inputs_unchanged=True, features_unchanged=True, cpu_rng_unchanged=True)
        return head, audit, payload
    except BaseException as error:
        error.readout_failure_payload = dict(schema_version=VERSION, completed_updates=completed,
            initial_head=initial, head=_snapshot(head.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
            error=f'{type(error).__name__}: {error}')
        raise


@torch.no_grad()
def score_readout(head, features, prepared, predicted_H=None):
    """One heldout protein; optional raw predicted-H uses the same fitted head."""
    _cpu_contract()
    require(not head.training and prepared['physical_support_audit']['split'] == 'readout_heldout'
            and 400 <= prepared['cache_row_index'] < 500, 'score requires heldout row and eval head')
    labels, mask, pairs = prepared['labels'], prepared['mask'], prepared['pairs']
    _features([features], [prepared])
    inputs = dict(future_H=features['target_H'], observed_H_substitution=features['observed_H'])
    if predicted_H is not None:
        inputs['predicted_H'] = predicted_H
    before = state_sha256(head)
    scores = {}; arrays = {key: value.detach().numpy().copy() for key, value in labels.items()}
    arrays['pair_indices'] = pairs.detach().numpy().copy()
    for role, H in inputs.items():
        pred = head(H, mask, pairs)
        scores[role] = probe_api.score_predictions(pred, labels['target_log'], target_distance=labels['target_distance'])
        arrays[role+'__predicted_log'] = pred.detach().numpy().copy()
    persistence = probe_api.distance_metrics(labels['observed_distance'], labels['target_distance'])
    persistent_log = torch.log1p(labels['observed_distance'])
    persistence['log_mse'] = float((persistent_log.double()-labels['target_log'].double()).square().mean())
    scores['CA_persistence'] = persistence
    arrays['persistence_log'] = persistent_log.numpy().copy()
    require(before == state_sha256(head), 'scoring changed head')
    return dict(cache_row_index=prepared['cache_row_index'], metadata=copy.deepcopy(prepared['row']['metadata']),
        n_pairs=len(pairs), labels_sha256=prepared['labels_sha256'], scores=scores, arrays=arrays)


def _records(records):
    require(len(records) == 100 and [r['cache_row_index'] for r in records] == list(range(400, 500))
            and len({r['metadata']['protein'] for r in records}) == 100
            and [r['metadata']['horizon'] for r in records] == list(HORIZONS)*25, 'exact heldout100 order required')
    keys = set(records[0]['scores'])
    required = {arm+'/'+role for arm in ARMS for role in ROLES[:2]} | {'CA_persistence'}
    predicted = {arm+'/predicted_H' for arm in ARMS}
    require(keys in (required, required | predicted) and all(set(r['scores']) == keys for r in records),
            'matched three-arm score roles differ')
    for r in records:
        for score in r['scores'].values():
            require(type(score['n_pairs']) is int and score['n_pairs'] > 0
                    and score['n_pairs'] == r['n_pairs'], 'within-protein pair support differs')
            require(type(score['negative_count']) is int and 0 <= score['negative_count'] <= score['n_pairs'],
                    'negative prediction count invalid')
            require(all(math.isfinite(score[k]) and score[k] >= 0 for k in METRICS), 'nonfinite/negative error metric')
            require(math.isclose(score['rmse_angstrom']**2, score['mse_angstrom2'], rel_tol=1e-12, abs_tol=1e-12),
                    'per-protein RMSE/MSE linkage differs')
    return sorted(keys)


def aggregate_readouts(records):
    """Equal-protein means; RMSE is sqrt(mean protein pair-MSE), not mean roots."""
    keys = _records(records)
    groups = {'overall': records, **{'h'+str(h): [r for r in records if r['metadata']['horizon'] == h] for h in HORIZONS}}
    result = {}
    for name, rows in groups.items():
        result[name] = {}
        for role in keys:
            values = [r['scores'][role] for r in rows]
            means = {k: float(np.mean([v[k] for v in values], dtype=np.float64)) for k in METRICS[:3]}
            require(all(math.isfinite(v) for v in means.values()), 'nonfinite aggregate')
            result[name][role] = dict(**means, rmse_angstrom=math.sqrt(means['mse_angstrom2']),
                n_proteins=len(rows), n_pairs=sum(v['n_pairs'] for v in values),
                negative_count=sum(v['negative_count'] for v in values),
                mean_protein_negative_fraction=float(np.mean([v['negative_count']/v['n_pairs'] for v in values])))
    return result


def paired_bootstrap(records):
    """10k paired resamples:25 proteins per horizon, equal100 after concatenation.

    Same index draws for every arm/role/metric. RMSE contrasts take the square
    root separately for each arm on each draw. Fixed percentile95% intervals;
    both primary contrasts reported without multiplicity/significance claims.
    """
    keys = _records(records)
    roles = [role for role in ROLES if 'pca/'+role in keys]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.asarray([i for i, r in enumerate(records) if r['metadata']['horizon'] == h]) for h in HORIZONS]
    require(all(len(s) == 25 for s in strata), 'bootstrap requires25 proteins per horizon')
    draws = np.concatenate([s[rng.integers(0, 25, size=(BOOTSTRAP_DRAWS, 25))] for s in strata], axis=1)
    contrasts = {}
    for arm in ARMS[1:]:
        contrasts[arm+'-pca'] = {}
        for role in roles:
            result = {}
            for metric in METRICS:
                key = 'mse_angstrom2' if metric == 'rmse_angstrom' else metric
                a = np.asarray([r['scores'][arm+'/'+role][key] for r in records], dtype=np.float64)
                b = np.asarray([r['scores']['pca/'+role][key] for r in records], dtype=np.float64)
                av, bv = a[draws].mean(1), b[draws].mean(1)
                if metric == 'rmse_angstrom':
                    distribution = np.sqrt(av)-np.sqrt(bv); point = math.sqrt(a.mean())-math.sqrt(b.mean())
                else:
                    distribution = av-bv; point = float(a.mean()-b.mean())
                lo, hi = np.quantile(distribution, [.025, .975], method='linear')
                require(np.isfinite(distribution).all() and math.isfinite(point), 'nonfinite bootstrap')
                result[metric] = dict(difference=float(point), interval95=[float(lo), float(hi)])
            contrasts[arm+'-pca'][role] = result
    return dict(schema_version=VERSION, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
        stratification='25 proteins resampled within each of h1,h4,h16,h64; equal100 protein weights',
        percentile_method='linear', primary='future_H/log_mse', contrasts=contrasts, limits=LIMITS)
