import json
import pytest
from evaluation.preregister import create_registration,verify_registration


def test_registered_final_test_rejects_protocol_and_role_changes(tmp_path):
    manifests={}
    for role in ('train','development','test'):
        p=tmp_path/(role+'.jsonl')
        p.write_text(json.dumps(dict(npz_path=str(tmp_path/(role+'.npz')),protein_id=role))+'\n')
        manifests[role]=str(p)
    protocol=dict(horizons_frames=[1,4],variants=['h_markov','h_plus_s','direct_jepa','persist'],
                  split='protein_holdout',manifests=manifests)
    registered=create_registration(dict(scopes={'cross':protocol},training_budget={'steps':20}))
    path=tmp_path/'registration.json';path.write_text(json.dumps(registered))
    cfg=dict(protocol,preregistration=str(path),comparison_scope='cross',dataset_role='final_test',
             manifest=manifests['test'])
    assert verify_registration(cfg)['dataset_role']=='test'
    with pytest.raises(ValueError,match='protocol differs'):
        verify_registration(dict(cfg,horizons_frames=[1]))
    with pytest.raises(ValueError,match='registered dataset role'):
        verify_registration(dict(cfg,manifest=manifests['development']))
    with pytest.raises(ValueError,match='locked preregistration'):
        verify_registration(dict(dataset_role='final_test'))
