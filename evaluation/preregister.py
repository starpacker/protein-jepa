"""Lock an endpoint-comparison protocol before opening the final test set."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from evaluation.loaders import sha256_file


PROTOCOL_FIELDS=('horizons_frames','context_frames','context_stride','frame_stride',
    'n_rollout_reps','seed','max_proteins','max_trajectories_per_protein',
    'max_anchors_per_trajectory','anchor_stride','skip_head_frames','max_residues',
    'physical_distance_scale_angstrom','physical_distance_max_pairs','variants')
DEFAULTS=dict(context_frames=8,context_stride=1,frame_stride=1,n_rollout_reps=4,seed=0,
    max_proteins=None,max_trajectories_per_protein=None,max_anchors_per_trajectory=None,
    anchor_stride=1,skip_head_frames=0,max_residues=None,
    physical_distance_scale_angstrom=1.,physical_distance_max_pairs=512)


def protocol_values(cfg):
    return {k:cfg.get(k,DEFAULTS.get(k)) for k in PROTOCOL_FIELDS}


def create_registration(cfg):
    """cfg declares scopes, all data roles, immutable training budget and metrics."""
    if not cfg.get('scopes') or not cfg.get('training_budget'):
        raise ValueError('preregistration requires scopes and explicit training_budget')
    scopes={}
    for name,spec in cfg['scopes'].items():
        manifests=spec.get('manifests',{})
        if set(manifests)!={'train','development','test'}:
            raise ValueError('each scope must declare train/development/test manifests')
        if set(spec.get('variants',[]))!={'h_markov','h_plus_s','direct_jepa','persist'}:
            raise ValueError('comparison requires h_markov/h_plus_s/direct_jepa/persist')
        from evaluation.splits import audit_manifest_pair
        split_audits={f'{a}_to_{b}':audit_manifest_pair(manifests[a],manifests[b],spec['split'])
                      for a,b in [('train','development'),('train','test'),('development','test')]}
        scopes[name]=dict(protocol=protocol_values(spec),split=spec['split'],
            split_audits=split_audits,
            manifests={role:dict(path=str(Path(p).resolve()),sha256=sha256_file(p))
                       for role,p in manifests.items()})
    return dict(schema_version=1,created_at=datetime.now(timezone.utc).isoformat(),
        purpose='locked_comparison_protocol_not_performance_evidence',scopes=scopes,
        training_budget=cfg['training_budget'],training_seeds=cfg.get('training_seeds',[0]),
        representation=cfg.get('representation'),
        primary_endpoints=['physical_energy_score','common_anchor_CA_RMSD'],
        primary_pair=['h_plus_s','h_markov'],
        secondary_pairs=[['h_plus_s','direct_jepa'],['h_plus_s','persist'],['h_markov','persist']],
        uncertainty='equal-window mean within protein; paired protein bootstrap 2000 replicates',
        decision_rule='report both primary effects at every preregistered horizon; no favorable-horizon selection',
        test_policy='no test-driven tuning, checkpoint selection, threshold changes, seed changes or window removal',
        gate_policy='failed readout gate remains failed; diagnostic_comparison cannot become headline evidence',
        path_claim='endpoint scores only; no claim of calibrated long-time transition kinetics',
        geometry='per-sample CA bond/clash diagnostics; ensemble mean reported separately',
        noise_policy='same per-window seeds; common H noise only if checkpoint kernel declares independent noise streams',
        limitations=cfg.get('limitations',[]))


def verify_registration(cfg):
    path=cfg.get('preregistration')
    if not path:
        if cfg.get('dataset_role')=='final_test':
            raise ValueError('final_test evaluation requires a locked preregistration')
        return None
    saved=json.loads(Path(path).read_text())
    scope=cfg.get('comparison_scope')
    spec=saved.get('scopes',{}).get(scope)
    if saved.get('schema_version')!=1 or not spec:
        raise ValueError('invalid preregistration or unknown comparison_scope')
    if spec['protocol']!=protocol_values(cfg) or spec['split']!=cfg.get('split'):
        raise ValueError('evaluation protocol differs from locked preregistration')
    role='test' if cfg.get('dataset_role')=='final_test' else 'development'
    if sha256_file(cfg['manifest'])!=spec['manifests'][role]['sha256']:
        raise ValueError('evaluation manifest differs from registered dataset role')
    if role=='test':
        from evaluation.splits import audit_manifest_pair
        for prior_role in ('train','development'):
            earlier=spec['manifests'][prior_role]
            if sha256_file(earlier['path'])!=earlier['sha256']:
                raise ValueError('registered prior-role manifest changed')
            audit_manifest_pair(earlier['path'],cfg['manifest'],cfg['split'])
    return dict(path=str(Path(path).resolve()),sha256=sha256_file(path),scope=scope,
                dataset_role=role,primary_endpoints=saved['primary_endpoints'],
                primary_pair=saved['primary_pair'])


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError('preregistration is immutable; choose a new experiment path')
    cfg=yaml.safe_load(Path(args.config).read_text())
    result=create_registration(cfg)
    result['config_source_sha256']=sha256_file(args.config)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(out)


if __name__=='__main__':
    main()
