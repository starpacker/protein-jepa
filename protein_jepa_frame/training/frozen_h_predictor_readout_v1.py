"""Frozen-H predictor calibration: two fixed distance heads and score arithmetic.

Pure tensor/metadata code: no files, models/checkpoints or data readers. The
caller authenticates native-coordinate feature rows and physical PSF support.
Seed42 supplies the optimization readout; independently initialized seed43 uses
exactly the same fit data and is withheld from predictor optimization. It is
not independent data or proof against exploitation of both heads. Historical
aligned-coordinate probe results are not matched controls for these native H.
"""
from __future__ import annotations

import copy
import math
import json
import numpy as np
import torch

from protein_jepa_frame.models.stage_c import frozen_h_distance_probe_v2 as old
from protein_jepa_frame.utils.contracts import state_sha256

VERSION = 'frozen_h_predictor_readout_v1'
HEAD_SEEDS = (42, 43)
HORIZONS = (1, 4, 16, 64)
TRAIN_ORDER = tuple(range(400)) + tuple(range(100))
ROLES = ('future_H', 'observed_H_substitution', 'raw_initial', 'latent', 'physical')
METRICS = ('log_mse', 'mae_angstrom', 'mse_angstrom2', 'rmse_angstrom')
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261102
PRIMARY = '43/physical-minus-latent/log_mse'
LIMITS = ('One paired predictor seed, previously inspected TRAIN100; held out only '
          'from this fit/training, not necessarily pretraining. Intervals condition '
          'on fitted models, without seed uncertainty or multiplicity correction. '
          'Head43 shares architecture and fit data with42; no exploitation-proof, '
          'Stage B distribution/kinetics, CA decoder or BioEmu adoption claim.')
require = old.require


class SymmetricDistanceHead(old.SymmetricDistanceProbe):
    """Same384→256→256→1 head; allow co-located FP32 GPU gradient inputs.

    CPU arithmetic/order and parameter names match the frozen probe exactly.
    No target labels, coordinate geometry or additional conditioning enter it.
    """
    def forward(self, H, mask, pair_indices):
        require(isinstance(H, torch.Tensor) and H.ndim == 3 and H.shape[0] == 1
                and H.shape[-1] == 128 and H.dtype == torch.float32
                and H.device.type in ('cpu', 'cuda'), 'H must be FP32[1,L,128]')
        L = H.shape[1]
        require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
                and mask.shape == (L,) and mask.device == H.device and bool(mask.any()), 'invalid mask')
        require(isinstance(pair_indices, torch.Tensor) and pair_indices.dtype == torch.long
                and pair_indices.ndim == 2 and pair_indices.shape[1] == 2 and len(pair_indices) > 0
                and pair_indices.device == H.device, 'invalid pairs')
        require(bool(((pair_indices >= 0) & (pair_indices < L)).all()), 'pair index out of bounds')
        i, j = pair_indices.T
        codes = i * L + j
        require(bool((i < j).all()) and bool(mask[pair_indices].all())
                and bool((codes[1:] > codes[:-1]).all()), 'pairs must be unique sorted valid upper triangle')
        require(all(p.dtype == torch.float32 and p.device == H.device for p in self.parameters()),
                'head/input device or dtype differs')
        require(not torch.is_autocast_enabled(H.device.type), 'head requires no autocast')
        require(bool(torch.isfinite(H[:, mask]).all()), 'nonfinite valid H')
        clean = torch.where(mask[None, :, None], H, 0.)[0]
        pooled = clean.sum(0) / mask.sum()
        mean = pooled.expand(len(i), -1)
        first = self.net(torch.cat((clean[i], clean[j], mean), -1)).squeeze(-1)
        reverse = self.net(torch.cat((clean[j], clean[i], mean), -1)).squeeze(-1)
        prediction = (first + reverse) * .5
        require(bool(torch.isfinite(prediction).all()), 'nonfinite log prediction')
        return prediction


def new_head(seed=42):
    _cpu()
    require(type(seed) is int and seed in HEAD_SEEDS, 'only declared head seeds42/43')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = SymmetricDistanceHead()
    head.calibration_seed = seed
    return head.float().cpu()


def _cpu():
    require(torch.get_default_device().type == 'cpu' and torch.get_default_dtype() == torch.float32
            and not torch.is_autocast_enabled('cpu'), 'requires default CPU FP32/no autocast')


def _H(value, mask, history=False):
    shape = (1, 8, len(mask), 128) if history else (1, len(mask), 128)
    require(isinstance(value, torch.Tensor) and value.shape == shape
            and value.dtype == torch.float32 and value.device.type == 'cpu' and not value.requires_grad
            and bool(torch.isfinite(value).all()), 'invalid detached finite CPU H')
    require(bool((value[..., ~mask, :] == 0).all()), 'nonzero masked H')


def validate_row(row, *, split):
    """Validate consumed row structure; caller must authenticate original plans.

    Metadata flags alone cannot prove dataset membership or full-PSF topology.
    Native feature extraction and raw geometry/label linkage are caller-owned.
    """
    require(split in ('fit', 'score'), 'unknown split')
    index = row['cache_row_index']
    require(type(index) is int and (0 <= index < 400 if split == 'fit' else 400 <= index < 500),
            'row outside fixed split')
    require(row.get('split', 'fit' if split == 'fit' else 'readout_heldout') == ('fit' if split == 'fit' else 'readout_heldout'), 'declared split differs')
    meta = row['metadata']
    require(isinstance(meta, dict) and isinstance(meta.get('protein'), str) and bool(meta['protein'])
            and type(meta.get('horizon')) is int and meta['horizon'] == HORIZONS[index % 4], 'metadata order differs')
    require(type(row['lag_ps']) in (int, float) and math.isfinite(row['lag_ps'])
            and row['lag_ps'] == 1000. * meta['horizon'], 'physical lag differs')
    require(type(row['temp']) in (int, float) and math.isfinite(row['temp']) and row['temp'] > 0,
            'invalid temperature')
    mask = row['mask']
    require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and mask.device.type == 'cpu'
            and mask.ndim == 1 and 1 <= len(mask) <= 96 and bool(mask.any()), 'invalid fixed observed mask')
    _H(row['history_H'], mask, history=True); _H(row['target_H'], mask)
    old._pairs(row['pairs'], mask)
    for key in ('target_log', 'target_distance', 'observed_distance', 'persistence_log'):
        x = row[key]
        require(isinstance(x, torch.Tensor) and x.shape == (len(row['pairs']),)
                and x.dtype == torch.float32 and x.device.type == 'cpu' and not x.requires_grad
                and bool(torch.isfinite(x).all()) and bool((x >= 0).all()), 'invalid label '+key)
    require(torch.equal(torch.log1p(row['target_distance']), row['target_log'])
            and torch.equal(torch.log1p(row['observed_distance']), row['persistence_log']),
            'distance/log label linkage differs')
    return row


def _tree_hash(value):
    flat = {}
    def visit(x, path):
        if isinstance(x, dict):
            for key, item in x.items():
                visit(item, path + [str(key)])
        elif isinstance(x, (list, tuple)):
            for index, item in enumerate(x):
                visit(item, path + [str(index)])
        else:
            flat[json.dumps(path)] = x
    visit(value, [])
    return state_sha256(flat)


def _fingerprint(rows):
    return _tree_hash(rows)


def _snapshot(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: _snapshot(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_snapshot(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_snapshot(v) for v in x)
    return copy.deepcopy(x)


def fit_head(rows400, *, seed, emit=None):
    """Fixed500 CPU Adam updates; returns frozen head, audit, checkpoint payload.

    No caller/global RNG changes, retry, best-checkpoint selection or IO. The
    original failure carries readout_failure_payload for caller serialization.
    """
    _cpu()
    require(len(rows400) == 400 and [r['cache_row_index'] for r in rows400] == list(range(400))
            and len({r['metadata']['protein'] for r in rows400}) == 400, 'fit requires exact first400 distinct proteins')
    require(emit is None or callable(emit), 'emit must be callable or None')
    for row in rows400:
        validate_row(row, split='fit')
    before = _fingerprint(rows400); rng = torch.get_rng_state().clone()
    head = new_head(seed).train()
    initial = _snapshot(head.state_dict()); initial_sha = state_sha256(initial)
    opt = torch.optim.Adam(head.parameters(), lr=.001, weight_decay=0.)
    completed = 0
    try:
        for step, index in enumerate(TRAIN_ORDER, 1):
            row = rows400[index]; opt.zero_grad(set_to_none=True)
            pred = head(row['target_H'], row['mask'], row['pairs'])
            loss = (pred-row['target_log']).square().mean()
            require(bool(torch.isfinite(loss)), 'nonfinite head loss')
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters()),
                    'missing/nonfinite head gradient')
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            require(bool(torch.isfinite(norm)), 'nonfinite head gradient norm')
            opt.step(); completed = step
            require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), 'nonfinite head update')
            if emit is not None:
                emit(dict(step=step, seed=seed, cache_row_index=index, protein=row['metadata']['protein'],
                    horizon=row['metadata']['horizon'], n_pairs=len(row['pairs']),
                    log_mse=float(loss.detach()), gradient_norm_before_clip=float(norm), lr=.001))
            del loss, pred, norm
        require(all(float(s['step']) == 500 for s in opt.state.values())
                and len(opt.state) == len(list(head.parameters())), 'head optimizer inventory/steps differ')
        require(before == _fingerprint(rows400), 'head fit mutated inputs')
        require(torch.equal(rng, torch.get_rng_state()), 'head fit changed CPU RNG')
        head.eval().requires_grad_(False); opt.zero_grad(set_to_none=True)
        payload = dict(schema_version=VERSION, seed=seed, step=500, initial_head=initial,
            head=_snapshot(head.state_dict()), optimizer=_snapshot(opt.state_dict()),
            initial_head_sha256=initial_sha, head_sha256=state_sha256(head),
            training_order=list(TRAIN_ORDER), selection='latest500 only')
        audit = dict(seed=seed, initial_head_sha256=initial_sha, final_head_sha256=payload['head_sha256'],
            optimizer_steps=500, input_sha256=before, inputs_unchanged=True, cpu_rng_unchanged=True,
            frozen=True, evaluation_mode=True)
        return head, audit, payload
    except BaseException as error:
        error.readout_failure_payload = dict(schema_version=VERSION, seed=seed, completed_updates=completed,
            initial_head=initial, head=_snapshot(head.state_dict()), optimizer=_snapshot(opt.state_dict()),
            input_sha256_before=before, cpu_rng_before=rng, cpu_rng_after=torch.get_rng_state().clone(),
            error=f'{type(error).__name__}: {error}')
        raise


def _latent_audit(H, target, mask):
    h, t = H[0, mask].double(), target[0, mask].double()
    centered = h-h.mean(-1, keepdim=True)
    return dict(endpoint_mse=float((h-t).square().mean()),
        raw_norm_mean=float(torch.linalg.vector_norm(h, dim=-1).mean()),
        raw_norm2_mean=float(h.square().sum(-1).mean()),
        centered_norm_mean=float(torch.linalg.vector_norm(centered, dim=-1).mean()),
        centered_norm2_mean=float(centered.square().sum(-1).mean()),
        channel_mean_mean=float(h.mean(-1).mean()), n_residues=int(mask.sum()))


@torch.no_grad()
def score_row(heads, row, predictions):
    """Score both fixed heads; no tensor mutation, parameter updates or RNG draws.

    predictions contains exactly latent, physical, raw_initial. The caller
    generates raw_initial by zero-factor OPF synthesis, retaining roundoff.
    Scores are fixed-head transfer, not a predicted-H-trained decoder bound.
    """
    _cpu(); validate_row(row, split='score')
    require(set(heads) == set(HEAD_SEEDS), 'requires exactly heads42/43')
    require(set(predictions) == {'latent', 'physical', 'raw_initial'}, 'requires both arms and raw_initial')
    for H in predictions.values():
        _H(H, row['mask'])
    for seed, head in heads.items():
        require(type(head) is SymmetricDistanceHead and getattr(head, 'calibration_seed', None) == seed, 'head seed lineage differs')
        require(not any(m.training for m in head.modules()) and not any(p.requires_grad for p in head.parameters())
                and all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in head.parameters()),
                'scoring heads must be frozen eval CPU FP32')
    before = _tree_hash(dict(heads={str(k): h.state_dict() for k, h in heads.items()}, row=row,
                              predictions=predictions)); rng = torch.get_rng_state().clone()
    Hs = dict(future_H=row['target_H'], observed_H_substitution=row['history_H'][:, -1], **predictions)
    arrays = {k: row[k].detach().numpy().copy() for k in
              ('mask', 'history_H', 'target_H', 'target_log', 'target_distance', 'observed_distance', 'persistence_log')}
    arrays['pair_indices'] = row['pairs'].numpy().copy()
    for key, H in predictions.items():
        arrays[key+'_H'] = H.numpy().copy()
    scores = {}
    for seed in HEAD_SEEDS:
        for role in ROLES:
            pred = heads[seed](Hs[role], row['mask'], row['pairs'])
            scores[f'{seed}/{role}'] = old.score_predictions(pred, row['target_log'], target_distance=row['target_distance'])
            arrays[f'seed{seed}__{role}__predicted_log'] = pred.numpy().copy()
    persistence = old.distance_metrics(row['observed_distance'], row['target_distance'])
    persistence['log_mse'] = float((row['persistence_log'].double()-row['target_log'].double()).square().mean())
    scores['CA_persistence'] = persistence
    latent_audits = {role: _latent_audit(H, row['target_H'], row['mask']) for role, H in Hs.items()}
    after = _tree_hash(dict(heads={str(k): h.state_dict() for k, h in heads.items()}, row=row, predictions=predictions))
    require(before == after and torch.equal(rng, torch.get_rng_state()), 'score mutated inputs/heads/RNG')
    record = dict(cache_row_index=row['cache_row_index'], metadata=copy.deepcopy(row['metadata']),
                  n_pairs=len(row['pairs']), scores=scores, latent_audits=latent_audits)
    return record, arrays


def _records(records):
    require(len(records) == 100 and [r['cache_row_index'] for r in records] == list(range(400, 500))
            and len({r['metadata']['protein'] for r in records}) == 100
            and [r['metadata']['horizon'] for r in records] == list(HORIZONS)*25, 'requires fixed heldout100 order')
    keys = [f'{seed}/{role}' for seed in HEAD_SEEDS for role in ROLES] + ['CA_persistence']
    for row in records:
        require(set(row['scores']) == set(keys), 'score roles differ')
        for score in row['scores'].values():
            require(type(score['n_pairs']) is int and score['n_pairs'] > 0 and score['n_pairs'] == row['n_pairs'],
                    'pair support differs')
            require(type(score['negative_count']) is int and 0 <= score['negative_count'] <= score['n_pairs'],
                    'invalid negative count')
            require(all(math.isfinite(score[k]) and score[k] >= 0 for k in METRICS), 'invalid metric')
            require(math.isclose(score['rmse_angstrom']**2, score['mse_angstrom2'], rel_tol=1e-12, abs_tol=1e-12),
                    'RMSE/MSE linkage differs')
    return keys


def aggregate_scores(records):
    keys = _records(records)
    groups = dict(overall=records, **{'h'+str(h): [r for r in records if r['metadata']['horizon'] == h] for h in HORIZONS})
    result = {}
    for group, rows in groups.items():
        result[group] = {}
        for key in keys:
            values = [r['scores'][key] for r in rows]
            mean = {k: float(np.mean([v[k] for v in values], dtype=np.float64)) for k in METRICS[:3]}
            require(all(math.isfinite(v) for v in mean.values()), 'nonfinite aggregate')
            result[group][key] = dict(**mean, rmse_angstrom=math.sqrt(mean['mse_angstrom2']),
                n_proteins=len(rows), n_pairs=sum(v['n_pairs'] for v in values),
                negative_count=sum(v['negative_count'] for v in values),
                mean_protein_negative_fraction=float(np.mean([v['negative_count']/v['n_pairs'] for v in values])))
    return result


def contrast_pairs():
    """Fixed primary, head42 secondary and descriptive controls; no selection."""
    result = {}
    for seed in HEAD_SEEDS:
        result[f'{seed}/physical-minus-latent'] = (f'{seed}/physical', f'{seed}/latent')
        for arm in ('latent', 'physical', 'raw_initial'):
            result[f'{seed}/{arm}-minus-observed_H_substitution'] = (f'{seed}/{arm}', f'{seed}/observed_H_substitution')
            result[f'{seed}/{arm}-minus-CA_persistence'] = (f'{seed}/{arm}', 'CA_persistence')
    return result


def paired_bootstrap(records):
    """Fixed10k paired25-per-horizon draws; RMSE roots separately per sample."""
    _records(records)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [np.asarray([i for i, r in enumerate(records) if r['metadata']['horizon'] == h]) for h in HORIZONS]
    draws = np.concatenate([s[rng.integers(0, 25, size=(BOOTSTRAP_DRAWS, 25))] for s in strata], axis=1)
    contrasts = {}
    for name, (left, right) in contrast_pairs().items():
        contrasts[name] = {}
        for metric in METRICS:
            key = 'mse_angstrom2' if metric == 'rmse_angstrom' else metric
            a = np.asarray([r['scores'][left][key] for r in records], dtype=np.float64)
            b = np.asarray([r['scores'][right][key] for r in records], dtype=np.float64)
            av, bv = a[draws].mean(1), b[draws].mean(1)
            if metric == 'rmse_angstrom':
                distribution = np.sqrt(av)-np.sqrt(bv); point = math.sqrt(a.mean())-math.sqrt(b.mean())
            else:
                distribution = av-bv; point = float(a.mean()-b.mean())
            require(np.isfinite(distribution).all() and math.isfinite(point), 'nonfinite bootstrap')
            lo, hi = np.quantile(distribution, [.025, .975], method='linear')
            contrasts[name][metric] = dict(difference=float(point), interval95=[float(lo), float(hi)])
    return dict(schema_version=VERSION, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
        stratification='25 proteins resampled within each of h1,h4,h16,h64; equal100 protein weights',
        percentile_method='linear', primary=PRIMARY, contrasts=contrasts, limits=LIMITS)
