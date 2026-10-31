import json
import sys
from types import SimpleNamespace
import pytest
from scripts import merge_bioemu2_shards as merger


def test_shard_merge_requires_exact_unique_completed_coverage(tmp_path,monkeypatch):
    row=dict(protein='p',traj='t',temperature_k=320.,anchor_frame=7,horizon=1.,bioemu2={})
    window=SimpleNamespace(protein_id='p',trajectory_id='t',temperature_k=320.,anchor_frame=7,horizon=1)
    monkeypatch.setattr(merger,'iter_windows',lambda _:iter([window]))
    shard=tmp_path/'shard.json';out=tmp_path/'out.json'
    data=dict(metric_version='empirical_w2_v2_pca_projection',manifest='m',n_samples=12,per_window=[row])
    shard.write_text(json.dumps(data))
    def invoke(inputs):
        monkeypatch.setattr(sys,'argv',['merge',*map(str,inputs),'--manifest','m','--windows','1','--out',str(out)])
        merger.main()
    invoke([shard])
    assert json.loads(out.read_text())['n_windows']==1
    with pytest.raises(ValueError,match='duplicate'):invoke([shard,shard])
    data['status']='incomplete';shard.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='incomplete'):invoke([shard])
    data.pop('status');data['per_window']=[];shard.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='exact requested'):invoke([shard])
