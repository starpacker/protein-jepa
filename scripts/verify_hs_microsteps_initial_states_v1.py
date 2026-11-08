"""Supplementary read-only initial-state and verifier-source identity audit."""
import hashlib
import json
import os
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1]
BASE=REPO/'results/hs_joint_20261101/microstep_train32_v1'
VERIFIER=REPO/'scripts/verify_hs_microsteps_v1.py'
VERIFIER_SHA='ee3745bf4ef418bbd1c1f12c0cd961877217464ab4516ca054f0580fdce04df4'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU visibility required')
    torch.set_num_threads(2)
    own_sha=sha(__file__)
    assert sha(VERIFIER)==VERIFIER_SHA
    verification=json.loads((BASE/'verification.execution.json').read_text())
    assert verification['returncode']==0
    assert sha(BASE/'independent_verification.json')==verification['output_sha256']
    result_path=BASE/'diagnostic.json';result_sha=sha(result_path)
    report=json.loads(result_path.read_text());checks=[]
    assert len(report['per_window'])==32
    for row in report['per_window']:
        path=REPO/row['raw_tensors']['path']
        assert sha(path)==row['raw_tensors']['sha256']
        raw=torch.load(path,map_location='cpu',weights_only=False)
        for key in ('1','4','8'):
            assert torch.equal(raw['branches'][key]['H'][:,0],raw['state']['H0'])
            assert torch.equal(raw['branches'][key]['S'][:,0],raw['state']['S0'])
            checks.append(dict(step=row['metadata']['step'],microsteps=int(key),members=12,
                H0_bitwise_equal=True,S0_bitwise_equal=True))
        del raw
    assert len(checks)==96
    assert sha(VERIFIER)==VERIFIER_SHA and sha(__file__)==own_sha and sha(result_path)==result_sha
    output=dict(passed=True,n_windows=32,n_branches=96,checks=checks,pid=os.getpid(),
        producer_result_sha256=result_sha,verifier_sha256_before_after=VERIFIER_SHA,
        supplementary_source_sha256_before_after=own_sha,
        primary_verification_sha256=verification['output_sha256'])
    with (BASE/'initial_state_and_verifier_hash_check.json').open('x') as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,branches=96,output=str(BASE/'initial_state_and_verifier_hash_check.json'))),flush=True)


if __name__=='__main__':run()
