"""Summarize future-informed component diagnostics without causal overclaims."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(data):
    output = {}
    keys = ['encdec_ca_rmsd','constant_ca_rmsd','self_anchor_ca_rmsd',
            'encdec_rmsf_ratio','latent_geometry_pair_corr',
            'oracle_initial_ca_rmsd','oracle_optimized_ca_rmsd',
            'latent_true_path_radius','latent_roll_radius','latent_radius_w2']
    for horizon in sorted({r['horizon'] for r in data['windows']}):
        rows = [r for r in data['windows'] if r['horizon']==horizon]
        stats = {}
        for key in keys:
            values = [r[key] for r in rows if r[key] is not None]
            stats[key] = dict(mean=float(np.mean(values)),median=float(np.median(values)),n=len(values)) if values else None
        stats['metrics'] = {branch: {m:float(np.mean([r['scores'][branch][m] for r in rows]))
                            for m in rows[0]['scores'][branch]}
                            for branch in ['encoded_future','gle_endpoint','constant_latent']}
        output[str(horizon)] = stats
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+');p.add_argument('--out',required=True)
    a=p.parse_args()
    result={}
    expected=None
    for name in a.inputs:
        data=json.loads(Path(name).read_text())
        keys={(r['protein'],r['traj'],r['anchor_frame'],r['horizon'],r['temperature_k']) for r in data['windows']}
        if expected is not None and keys!=expected:
            raise ValueError('component diagnostic windows differ')
        expected=keys
        result[Path(name).stem]=dict(n_windows=len(keys),checkpoint=data['checkpoint'],by_horizon=summarize(data))
    Path(a.out).write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
