"""Evaluate a CA oracle component on a separately sampled protein split.

Future coordinates enter only the oracle encoder, not a forecast. This evaluation
cannot establish GLE or BioEmu2 superiority.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.dataset import build_dataset
from protein_jepa_frame.data.collate import collate_clips
from protein_jepa_frame.data.protocol import validate_manifest_split
from protein_jepa_frame.training.train_lasp_calibration import Calibration
from protein_jepa_frame.training.train_utils import _move_batch


def bootstrap(rows,key,seed=17):
    groups={}
    for row in rows:groups.setdefault(row['protein'],[]).append(row[key])
    # Equal protein weighting; rows within a protein are dependent.
    values=np.array([np.mean(v) for v in groups.values()])
    rng=np.random.default_rng(seed)
    means=values[rng.integers(len(values),size=(4000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n_proteins=len(values))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',default='artifacts/mdcath_manifest_full/test.jsonl')
    p.add_argument('--windows-per-stride',type=int,default=80)
    p.add_argument('--max-residues',type=int,default=96)
    p.add_argument('--seed',type=int,default=2718)
    p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    args=SimpleNamespace(**ck['calibration_args']);args.init_from=a.checkpoint
    audit=validate_manifest_split(args.manifest,a.manifest)
    model=Calibration(args,torch.device('cuda')).to('cuda').eval()
    rows=[]
    for stride in (1,4,16):
        ds=build_dataset(DataConfig(manifest_path=a.manifest,training_mode='clip',num_frames=5,
             frame_stride=stride,max_residues=a.max_residues))
        rng=np.random.default_rng(a.seed+stride)
        # Distinct proteins first, one random clip per protein before repeats.
        records={}
        previous=0
        for entry,end in zip(ds.entries,ds.cumulative):
            protein=ds.records[entry.record_index].protein_id
            records.setdefault(protein,[]).append((previous,end));previous=end
        proteins=list(records);rng.shuffle(proteins)
        indices=[]
        for i in range(a.windows_per_stride):
            protein=proteins[i%len(proteins)]
            intervals=records[protein];lo,hi=intervals[int(rng.integers(len(intervals)))]
            indices.append(int(rng.integers(lo,hi)))
        loader=DataLoader(Subset(ds,indices),batch_size=1,collate_fn=collate_clips)
        for index,batch in zip(indices,loader):
            with torch.no_grad():terms=model(_move_batch(batch,'cuda'),controls=True)
            m={k:float(v) for k,v in terms.items()}
            rows.append(dict(protein=batch['protein_id'][0],trajectory=batch['trajectory_id'][0],
                time_ps=batch['time_ps'].tolist(),stride=stride,index=index,**m,
                persistence_improvement=m['fixed_ca_rmsd']-m['ca_rmsd'],
                shuffled_improvement=m['shuffled_ca_rmsd']-m['ca_rmsd']))
    result=dict(scope='Oracle CA component; not a forecast',checkpoint=a.checkpoint,
                args=vars(a),split_audit=audit,rows=rows,
                aggregate={k:bootstrap(rows,k) for k in ['ca_rmsd','fixed_ca_rmsd',
                    'persistence_improvement','shuffled_improvement','rmsf_corr','rmsf_mae','rmsf_ratio','ca_bond','contact','ca_angle_cos_mae','ca_clash_fraction','true_ca_clash_fraction']},
                by_stride={str(s):{k:bootstrap([r for r in rows if r['stride']==s],k)
                    for k in ['ca_rmsd','persistence_improvement','shuffled_improvement','ca_bond','rmsf_corr']} for s in (1,4,16)})
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['aggregate'],indent=2),flush=True)


if __name__=='__main__':main()
