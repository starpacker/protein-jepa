"""Representation Sufficiency Probe (E1).

Answers: does the current JEPA state H lose information that the *observed
history* (and the exact Kalman bath state S) retain about the future?

Three input probes, three future targets, three timescales:

  P1 = anchor latent H_t (+ context-mask protein-mean H_t + temperature)
  P2 = P1 + per-residue and protein-mean H(-j)-H_t deltas over K observed frames
  P3 = P1 + exact Kalman bath posterior mean S_0  (s0_mode='exact')

  A = future CA internal-distance change   (fixed nonlocal pairs, delta d_ij)
  B = future contact-state change          (1[d<8A] flip on the same pairs)
  C = future collective-motion projection  (MD-PCA slow-mode coordinate)

For every (target x timescale) we fit a weighted ridge probe per input, select
the ridge lambda by grouped protein CV entirely within the TRAIN manifest, then
evaluate once on the VAL manifest.  G_history = R(P1) - R(P2) (in units of the
probe's squared-vector error, so G>0 means history adds predictive value beyond
the single-frame state).  P3 is reported alongside to show S's marginal
contribution.  All targets are conditioned on H_0 in the sense that the probe
families all include P1's H_0 features; the comparison is about the *increment*
history / bath add.

Frozen encoder only (exact-arm ``lasp_full_v1/representation``); no propagator
is trained.  Only train.jsonl (lambda selection) and val.jsonl (evaluation) are
read.  test.jsonl is never opened.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluation.data import EvalProtocol, iter_windows
from evaluation.loaders import sha256_file
from protein_jepa_frame.training.train_lasp_ca_dynamics import CADynamics
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.diag_lasp_ca_ceiling import _development_manifest
from scripts.probe_conditional_history import RidgeProbe

TIMESCALES = (1, 4, 16)
CONTACT_CUTOFF = 8.0
MAX_PAIRS = 256
PCA_K = 3
PROBES = ('P1', 'P2', 'P3')


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def encode_history(bundle, context, mask, aa, ri, ci):
    """Encode all K observed frames -> [K, L, dH] (oldest first).

    ``context`` is [K,L,4,3], ``mask`` [L], ``aa/ri/ci`` [L].  bundle.encode
    internally adds a time dim via backbone[:,None], so mask/aa/ri/ci must be
    [B,L]; here B=K (encode every frame as an independent batch element).
    """
    K = context.shape[0]
    m = mask[None].expand(K, -1)
    h = bundle.encode(context, m, aa[None].expand(K, -1), ri[None].expand(K, -1),
                      ci[None].expand(K, -1))
    return h  # [K, L, dH]


def feature_arrays(h, mask, temperature_k, s0=None):
    """Build P1/P2/P3 features.  ``h`` is [K,L,dH] oldest->newest, ``s0`` [L,dS] or None.

    Returns a dict probe -> [n_valid, D] numpy array (rows are context-valid
    residues only, in full-L index order) and the full-L boolean validity mask.
    """
    valid = mask.bool()
    h = h[:, valid]
    count = h.shape[1]
    means = h.mean(1, keepdim=True).expand(-1, count, -1)
    base = torch.cat((h[-1], means[-1], h.new_full((count, 1), temperature_k)), dim=-1)
    delta = torch.cat((h[:-1] - h[-1:], means[:-1] - means[-1:]), dim=-1)
    delta = delta.flip(0).permute(1, 0, 2).reshape(count, -1)
    out = {
        'P1': base,
        'P2': torch.cat((base, delta), dim=-1),
    }
    if s0 is not None:
        s0 = s0[valid]
        out['P3'] = torch.cat((base, s0), dim=-1)
    else:
        out['P3'] = None
    return {k: (None if v is None else v.detach().cpu().numpy().astype(np.float64))
            for k, v in out.items()}, valid.cpu().numpy()


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def nonlocal_pairs(L, mask, max_pairs=MAX_PAIRS, seed=0):
    """Deterministic upper-triangular nonlocal pairs (|i-j|>=4) capped at max_pairs."""
    idx = torch.arange(L)
    pairs = []
    for i in range(L):
        for j in range(i + 4, L):
            if mask[i] and mask[j]:
                pairs.append((i, j))
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    return pairs


def _pair_distance(coords_ca, mask, pairs):
    """coords_ca [L,3] -> distances [P] for the fixed pairs."""
    out = []
    for i, j in pairs:
        d = (coords_ca[i] - coords_ca[j]).norm().item()
        out.append(d)
    return np.asarray(out, dtype=np.float64)


def target_arrays(future_ca, anchor_ca, mask, pairs):
    """future_ca [h+1,L,3] (index 0 == anchor), anchor_ca [L,3].

    Returns dict target A/B/C at each timescale in TIMESCALES (index into future
    frames 1..h).  Collective-motion target C is filled later after global PCA.
    """
    d0 = _pair_distance(anchor_ca, mask, pairs)
    contact0 = (d0 < CONTACT_CUTOFF).astype(np.float64)
    out = {}
    for tau in TIMESCALES:
        if tau >= future_ca.shape[0]:
            out[tau] = None
            continue
        dtau = _pair_distance(future_ca[tau], mask, pairs)
        A = (dtau - d0)                                     # [P] distance change
        B = (dtau < CONTACT_CUTOFF).astype(np.float64) - contact0   # [P] contact flip
        # C (collective) filled after global PCA fit.
        out[tau] = dict(A=A, B=B, displacement=future_ca[tau] - anchor_ca)  # disp [L,3]
    return out


# --------------------------------------------------------------------------- #
# Ridge probe plumbing (reuse probe_conditional_history ridge_path)
# --------------------------------------------------------------------------- #
def collect_rows(bundle, kernel, protocol, limit, device, dtype):
    rows = []
    seen = set()
    for window in iter_windows(protocol):
        if len(rows) == limit:
            break
        context = window.context['coords'].to(device)
        mask = window.context['residue_mask'].to(device).all(0).bool()
        future = window.future_path
        if future is None or int(mask.sum()) < 4:
            continue
        aa, ri, ci = [window.context[k].to(device) for k in
                      ('aatype', 'residue_index', 'chain_id')]  # each [L]
        h = encode_history(bundle, context, mask, aa, ri, ci)          # [K,L,dH]
        # Exact Kalman bath posterior mean S0 from the observed H history.
        s0 = None
        if kernel is not None:
            K = h.shape[0]
            dt = torch.full((1, K), kernel.cfg.step_ps, device=device, dtype=dtype)
            temp = torch.tensor([window.temperature_k], device=device, dtype=dtype)
            m0, _chol0 = kernel.exact_bath_posterior(
                h[None], dt, temp, mask[None], seq_feature=h[-1:])
            s0 = m0[0]                                                # [L,dS]
        features, valid = feature_arrays(h, mask, window.temperature_k, s0)
        # full-L index -> feature-row index (features are context-valid only)
        L = len(mask)
        valid_idx = np.where(valid)[0]
        full_to_row = {int(full): int(row_i) for row_i, full in enumerate(valid_idx)}
        pairs = nonlocal_pairs(L, mask.cpu().bool(), seed=0)
        future_ca = future['coords'][:, :, 1].to(device)              # [h+1,L,3]
        anchor_ca = context[-1, :, 1].to(device)                      # [L,3]
        targets = target_arrays(future_ca, anchor_ca, mask, pairs)
        key = (window.protein_id, window.trajectory_id, float(window.temperature_k),
               int(window.anchor_frame), int(window.horizon))
        if key in seen:
            raise ValueError('duplicate window')
        seen.add(key)
        rows.append(dict(protein=window.protein_id, horizon=int(window.horizon),
                         features=features, targets=targets, pairs=pairs,
                         pair_rows=[(full_to_row[i], full_to_row[j]) for (i, j) in pairs],
                         mask=mask.cpu().bool()))
        if len(rows) % 20 == 0:
            print(f'  encoded {len(rows)}/{limit}', flush=True)
    return rows


def _pack(X_list, Y_list, weights):
    return (np.concatenate(X_list), np.concatenate(Y_list), np.concatenate(weights))


class ScalarRidge:
    """Weighted standardized ridge for arbitrary (scalar or vector) targets."""
    def __init__(self, mean, scale, target_mean, coefficients, regularization):
        self.mean = mean
        self.scale = scale
        self.target_mean = target_mean
        self.coefficients = coefficients
        self.regularization = regularization

    def predict(self, x):
        return ((x - self.mean) / self.scale) @ self.coefficients + self.target_mean


def scalar_ridge_fit(x, y, weights, lam):
    """Weighted standardized ridge with unpenalized intercept; y may be [N,O]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    weights = weights / weights.sum()
    mean, ym = weights @ x, weights @ y
    xc = x - mean
    scale = np.sqrt(weights @ xc ** 2)
    scale = np.where(scale > 1e-8, scale, 1.)
    xc = xc / scale
    gram = xc.T @ (xc * weights[:, None])
    gram = 0.5 * (gram + gram.T)
    rhs = xc.T @ ((y - ym) * weights[:, None])
    coeff = np.linalg.solve(gram + lam * np.eye(x.shape[1]), rhs)
    return ScalarRidge(mean, scale, ym, coeff, lam)


def build_fit_pack(rows, probe, target_key, tau, protein_weight):
    # Target at timescale tau is only defined for windows whose horizon == tau
    # (the MD future path has horizon+1 frames).  Targets A/B are per-nonlocal-
    # pair (features = concat of residue i and j features); target C is
    # per-residue.
    rows = [r for r in rows if r['horizon'] == tau]
    xs, ys, ws = [], [], []
    counts = Counter(r['protein'] for r in rows)
    for r in rows:
        feat = r['features'][probe]      # [n_valid, D]
        tgt = r['targets'].get(tau)
        if feat is None or tgt is None:
            continue
        if target_key == 'C':
            y = tgt.get('c_proj')        # [n_valid] filled later
            if y is None:
                continue
            x = feat                      # per-residue
        else:
            if tgt.get(target_key) is None:
                continue
            y = tgt[target_key]          # [P] per-pair
            # build per-pair features
            x = np.stack([np.concatenate([feat[i], feat[j]]) for (i, j) in r['pair_rows']])
        n = len(x)
        if y.ndim == 1:
            y = y[:, None]
        xs.append(x)
        ys.append(y)
        ws.append(np.full(n, 1 / (len(counts) * counts[r['protein']] * n)))
    if not xs:
        return None
    return _pack(xs, ys, ws)


def fit_one(train_rows, probe, target_key, tau, lambdas, protein_weights):
    pack = build_fit_pack(train_rows, probe, target_key, tau, protein_weights)
    if pack is None:
        return None
    x, y, w = pack
    # Select lambda by grouped protein CV within train.
    proteins = sorted({r['protein'] for r in train_rows if r['horizon'] == tau})
    from numpy.random import default_rng
    rng = default_rng(20261101)
    rng.shuffle(proteins)
    folds = 5 if len(proteins) >= 5 else max(2, len(proteins))
    assign = {p: i % folds for i, p in enumerate(proteins)}
    best_lam, best_score = None, math.inf
    for lam in lambdas:
        scores = []
        for f in range(folds):
            fit_rows = [r for r in train_rows if r['horizon'] == tau and assign[r['protein']] != f]
            ho_rows = [r for r in train_rows if r['horizon'] == tau and assign[r['protein']] == f]
            fpack = build_fit_pack(fit_rows, probe, target_key, tau, protein_weights)
            hpack = build_fit_pack(ho_rows, probe, target_key, tau, protein_weights)
            if fpack is None or hpack is None:
                continue
            model = scalar_ridge_fit(*fpack, lam)
            fx, fy, fw = hpack
            pred = model.predict(fx)
            mse = float(np.mean(np.sum((pred - fy) ** 2, axis=-1)))
            scores.append(mse)
        if not scores:
            continue
        score = float(np.mean(scores))
        if score < best_score:
            best_score, best_lam = score, lam
    if best_lam is None:
        best_lam = lambdas[0]
    return scalar_ridge_fit(*pack, best_lam), best_lam, best_score


def evaluate_pack(rows, model, probe, target_key, tau):
    pack = build_fit_pack(rows, probe, target_key, tau, None)
    if pack is None or model is None:
        return None
    x, y, w = pack
    pred = model.predict(x)
    # protein-level mean squared-vector error
    grouped = defaultdict(list)
    # rebuild per-protein
    counts = Counter(r['protein'] for r in rows)
    # simpler: global weighted mse here is enough for G; also report per-protein
    mse = float(np.mean(np.sum((pred - y) ** 2, axis=-1)))
    return dict(mse=mse, rmsd=float(np.sqrt(max(mse, 0))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--train-manifest', default='artifacts/mdcath_manifest_full/train.jsonl')
    p.add_argument('--val-manifest', default='artifacts/mdcath_manifest_full/val.jsonl')
    p.add_argument('--out', required=True)
    p.add_argument('--train-windows', type=int, default=200)
    p.add_argument('--val-windows', type=int, default=200)
    p.add_argument('--max-residues', type=int, default=96)
    p.add_argument('--context-frames', type=int, default=8)
    p.add_argument('--lambdas', type=float, nargs='+', default=[1e-5, 1e-3, .1, 1., 10., 100., 1000., 10000.])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    device = torch.device(a.device)

    val_manifest = _development_manifest(a.val_manifest)
    approved_train = Path(__file__).resolve().parents[1] / 'artifacts/mdcath_manifest_full/train.jsonl'
    if Path(a.train_manifest).resolve() != approved_train.resolve():
        raise ValueError('E1 requires the approved development train.jsonl manifest')

    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('checkpoint_kind') != 'lasp_ca_dynamics' or 'component_checkpoint' not in ck:
        raise ValueError('requires a lasp_ca_dynamics checkpoint with an embedded component')
    args = ck['args']
    with tempfile.TemporaryDirectory(prefix='rep_suff_probe_') as temp:
        component_path = Path(temp) / 'component.pt'
        torch.save(ck['component_checkpoint'], component_path)
        model = CADynamics(str(component_path), device, a.checkpoint,
                           history_conditioning=args.get('history_conditioning', False),
                           noise_mix=0.0, noise_length=args.get('noise_length', 8.),
                           collective_mobility=args.get('collective_mobility', False),
                           noise_kernel=args.get('noise_kernel', 'gaussian'),
                           noise_cutoff=args.get('noise_cutoff', 10.),
                           s0_mode=args.get('s0_mode', 'gru')).eval()
        bundle = model.bundle
        kernel = model.kernel
        dtype = next(bundle.encoder.parameters()).dtype

        def protocol(manifest):
            return EvalProtocol(manifest=str(Path(manifest).resolve()),
                horizons_frames=TIMESCALES, context_frames=a.context_frames, frame_stride=1,
                max_residues=a.max_residues, max_anchors_per_trajectory=4,
                temperatures=(320., 348., 379., 413., 450.), seed=a.seed,
                round_robin_proteins=True, round_robin_conditions=True)

        with torch.no_grad():
            print('collecting train windows...', flush=True)
            train_rows = collect_rows(bundle, kernel, protocol(a.train_manifest),
                                      a.train_windows, device, dtype)
            print('collecting val windows...', flush=True)
            val_rows = collect_rows(bundle, kernel, protocol(val_manifest),
                                    a.val_windows, device, dtype)
        train_proteins = {r['protein'] for r in train_rows}
        val_proteins = {r['protein'] for r in val_rows}
        overlap = train_proteins & val_proteins

        # Global MD-PCA slow modes fit on TRAIN displacements only (no val leakage).
        all_disp = []
        for r in train_rows:
            for tau in TIMESCALES:
                t = r['targets'].get(tau)
                if t is None:
                    continue
                all_disp.append(t['displacement'][r['mask']].cpu().numpy())
        all_disp = np.concatenate([d.reshape(-1, 3) for d in all_disp], axis=0)
        # per-residue displacement PCA (3D), slow modes = largest-variance directions
        mu = all_disp.mean(0)
        X = all_disp - mu
        cov = X.T @ X / len(X)
        evals, evecs = np.linalg.eigh(cov)
        modes = evecs[:, ::-1][:, :PCA_K]  # [3, k]
        # Project each window's per-residue displacement onto modes -> per-residue scalar (sum over modes)
        for rows in (train_rows, val_rows):
            for r in rows:
                for tau in TIMESCALES:
                    t = r['targets'].get(tau)
                    if t is None:
                        continue
                    disp = t['displacement'].detach().cpu()  # [L,3]
                    proj = (disp - mu) @ modes  # [L,k]
                    t['c_proj'] = proj.sum(-1).numpy()[r['mask']]  # [n_valid] collective coordinate

        # Fit probes per (probe, target, timescale) on train, evaluate on val.
        protein_weights = None
        result = dict(schema_version=1, test_opened=False,
                      checkpoint=a.checkpoint, args=vars(a),
                      n_train_proteins=len(train_proteins), n_val_proteins=len(val_proteins),
                      protein_overlap=len(overlap), targets={})
        TARGETS = {'A': 'CA internal-distance change', 'B': 'contact-state change',
                   'C': 'collective-motion projection'}
        for tau in TIMESCALES:
            result['targets'][str(tau)] = {}
            for tk in ('A', 'B', 'C'):
                fits = {}
                for probe in PROBES:
                    if probe == 'P3' and kernel is None:
                        fits[probe] = None
                        continue
                    fitted = fit_one(train_rows, probe, tk, tau, a.lambdas, protein_weights)
                    fits[probe] = fitted
                # evaluate on val
                val_err = {}
                for probe in PROBES:
                    f = fits[probe]
                    model_ = None if f is None else f[0]
                    val_err[probe] = evaluate_pack(val_rows, model_, probe, tk, tau)
                p1 = val_err.get('P1') or {}
                p2 = val_err.get('P2') or {}
                g = None
                if p1.get('mse') is not None and p2.get('mse') is not None:
                    g = p1['mse'] - p2['mse']
                result['targets'][str(tau)][tk] = dict(
                    target=TARGETS[tk], G_history=g,
                    R_P1=val_err.get('P1'), R_P2=val_err.get('P2'), R_P3=val_err.get('P3'),
                    fitted_lambda={probe: (None if fits[probe] is None else fits[probe][1])
                                   for probe in PROBES})
                print(f'tau={tau} target={tk} G_history={g} P1_mse={p1.get("mse")} '
                      f'P2_mse={p2.get("mse")} P3_mse={(val_err.get("P3") or {}).get("mse")}', flush=True)

        Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(output=str(a.out), protein_overlap=len(overlap))), flush=True)


if __name__ == '__main__':
    main()
