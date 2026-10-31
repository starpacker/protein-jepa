"""Merge completed BioEmu2 shards only after checking exact protocol coverage."""
import argparse
import json
from pathlib import Path
from evaluation.data import EvalProtocol,iter_windows


def key(row):
    return (row['protein'],row['traj'],float(row['temperature_k']),int(row['anchor_frame']),float(row['horizon']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+');p.add_argument('--manifest',required=True)
    p.add_argument('--windows',type=int,default=200);p.add_argument('--max-residues',type=int,default=96)
    p.add_argument('--out',required=True)
    a=p.parse_args();records={};sources=[];settings=None
    for path in a.inputs:
        d=json.loads(Path(path).read_text())
        if d.get('status')=='incomplete':raise ValueError('incomplete shard')
        if d.get('n_windows',len(d['per_window']))!=len(d['per_window']):raise ValueError('shard row count mismatch')
        if d.get('metric_version')!='empirical_w2_v2_pca_projection':raise ValueError('metric version mismatch')
        if d['manifest']!=a.manifest or d['n_samples']!=12:raise ValueError('protocol mismatch')
        if d.get('n_skipped',0):raise ValueError('skipped windows')
        config=d.get('evaluation_args')
        if config is not None:
            comparable={k:v for k,v in config.items() if k not in ('out','window_start','max_windows','device')}
            if settings is not None and comparable!=settings:raise ValueError('shard sampling settings differ')
            settings=comparable
        for row in d['per_window']:
            k=key(row)
            if k in records:raise ValueError(f'duplicate window {k}')
            records[k]=row
        sources.append(path)
    protocol=EvalProtocol(manifest=a.manifest,horizons_frames=(1,4,16,64),context_frames=8,
        max_residues=a.max_residues,max_anchors_per_trajectory=4,round_robin_proteins=True,round_robin_conditions=True)
    expected=[]
    for w in iter_windows(protocol):
        if len(expected)>=a.windows:break
        expected.append((w.protein_id,w.trajectory_id,float(w.temperature_k),int(w.anchor_frame),float(w.horizon)))
    if len(expected)!=a.windows or set(expected)!=records.keys():raise ValueError('shards do not cover exact requested protocol windows')
    result=dict(baseline='bioemu2',metric_version='empirical_w2_v2_pca_projection',manifest=a.manifest,
        n_samples=12,n_windows=len(expected),n_proteins=len({k[0] for k in expected}),sources=sources,
        evaluation_args=settings,
        per_window=[records[k] for k in expected])
    Path(a.out).write_text(json.dumps(result,indent=2))
    print(f'merged {len(expected)} windows')


if __name__=='__main__':main()
