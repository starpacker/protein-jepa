"""Compare two per-window metric files using exact keys and protein bootstrap."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.eval_lasp_final import METRICS, HIGHER_BETTER, _cluster_boot_ci


def index_rows(path):
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else data['per_window']
    out = {}
    for r in rows:
        key = (r['protein'], r['traj'], r['temperature_k'], r['anchor_frame'], r['horizon'])
        if key in out:
            raise ValueError(f'duplicate window: {key}')
        out[key] = r
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate'); p.add_argument('reference')
    p.add_argument('--candidate-field', default='lasp')
    p.add_argument('--reference-field', default='bioemu2')
    p.add_argument('--out', required=True)
    p.add_argument('--strict-finite',action='store_true',help='Reject nonfinite paired metrics instead of filtering them.')
    a = p.parse_args()
    candidate, reference = index_rows(a.candidate), index_rows(a.reference)
    if candidate.keys() != reference.keys():
        raise ValueError('window sets differ; do not silently compare a selected intersection')
    result = {}
    for h in sorted({k[-1] for k in candidate}):
        keys = [k for k in candidate if k[-1] == h]
        metrics = {}
        for m in METRICS:
            x = np.array([candidate[k][a.candidate_field][m] for k in keys])
            y = np.array([reference[k][a.reference_field][m] for k in keys])
            valid = np.isfinite(x) & np.isfinite(y)
            if a.strict_finite and not valid.all():raise ValueError(f'nonfinite paired metrics: horizon={h}, metric={m}')
            direction = 1 if m in HIGHER_BETTER else -1
            delta = direction * (x[valid] - y[valid])
            clusters = np.array([k[0] for k in keys])[valid]
            lo, hi = _cluster_boot_ci(delta, clusters)
            metrics[m] = dict(candidate=float(np.mean(x[valid])), reference=float(np.mean(y[valid])),
                              improvement=float(np.mean(delta)), ci95=[lo, hi],
                              n_valid=int(valid.sum()), n_total=len(keys),
                              significant_win=bool(lo > 0), significant_loss=bool(hi < 0))
        result[str(h)] = metrics
    Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
