"""Registered TRAIN32 reference-conditioned bounded joint-frame oracle.

Registration reads existing JSON metadata and hashes binary inputs only. Run
deserializes the pinned TRAIN cache/head; it never samples dynamics or reads MD.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from evaluation import ramachandran_reference_v1 as rama
from run_scripts import run_backbone_transport_pilot as transport
from scripts import eval_backbone_readout_pilot as scoring
from protein_jepa_frame.models.backbone_readout_v2 import BackboneReadoutConfig, BackboneReadoutV2
from protein_jepa_frame.utils.contracts import state_sha256

VERSION = 'train_joint_frame_reference_oracle_v1'
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'results/hs_joint_20261101/train_joint_frames_v1'
CACHE = 'results/hs_joint_20261101/backbone_repair/readout_train500_v2.pt'
CACHE_SHA = 'd48faeb6e7385fa0b37a1daf440bf4708ed3472d83e2073b8a44d9073a216718'
CHECKPOINT = 'artifacts/hs_joint_20261101/backbone_transport_v1/coordinate/checkpoint_latest.pt'
CHECKPOINT_SHA = 'f4fded7c56cdbea1e3ef327b94f9ea855448f72126adae2d92aa9a75a2b75954'
SELECTION = str(Path(CACHE).with_suffix('.selection.json'))
CACHE_PROTOCOL = str(Path(CACHE).with_suffix('.protocol.json'))
CONDITIONS = ('decoded_oracle_ca', 'true_md_ca')
ROLES = tuple(c+'_'+suffix for c in CONDITIONS for suffix in ('baseline','corrected')) + ('target_md',)
SOURCES = tuple(sorted(set(transport.SOURCES + (
    'run_scripts/diagnose_train_joint_frames_v1.py',
    'tests/test_train_joint_frames_v1.py',
    'docs/TRAIN_JOINT_FRAME_ORACLE_V1_20261102.md',
    'protein_jepa_frame/models/stage_c/joint_frame_oracle_v1.py',
    'tests/test_joint_frame_oracle_v1.py',
    'evaluation/ramachandran_reference_v1.py',
    'evaluation/reference_data/ramachandran_top8000_v1/PROVENANCE.json',
    str(rama.TABLE_PATH.relative_to(ROOT)),
))))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    data = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with Path(path).open('x') as stream:
        stream.write(data)


def save_tensor(path, value):
    with Path(path).open('xb') as stream:
        torch.save(value, stream)


def cpu_guard():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'requires explicit CUDA_VISIBLE_DEVICES empty')


def selected_metadata(selection, cached_protocol):
    """Metadata-only fixed prefix, never select replacement rows."""
    require(selection.get('schema_version') == 'backbone_readout_cache_v1'
            and selection.get('mode') == 'train' and selection.get('seed') == 42
            and selection.get('n_windows') == selection.get('selected_prefix') == 500,
            'wrong TRAIN selection metadata')
    full = selection['full_selection']
    require(len(full) == 500, 'fixed full500 selection required')
    metadata = [r['metadata'] for r in full]
    require(len({r['protein'] for r in metadata}) == 500
            and [r['horizon'] for r in metadata] == [1,4,16,64]*125,
            'TRAIN selection identities/schedule differ')
    protocol = cached_protocol['protocol']
    require(protocol.get('mode') == 'train' and protocol.get('n_windows') == 500
            and protocol.get('dataset_role') == 'training_readout_calibration'
            and protocol.get('selection_file') == SELECTION
            and protocol.get('selection_sha256') == sha(ROOT/SELECTION),
            'cache protocol/selection linkage differs')
    rows = metadata[:32]
    require(len({r['protein'] for r in rows}) == 32
            and Counter(r['horizon'] for r in rows) == {1:8,4:8,16:8,64:8},
            'first32 must contain32 proteins and8 of each horizon')
    return rows


def verify(protocol):
    for relative, digest in protocol['input_hashes'].items():
        require(sha(ROOT/relative) == digest, 'input changed: '+relative)
    for relative, digest in protocol['source_hashes'].items():
        path = Path(relative)
        require(not path.is_absolute() and (ROOT/path).resolve().is_relative_to(ROOT), 'source path escapes repository')
        require(sha(ROOT/path) == digest, 'source changed: '+relative)
        require(sha(ROOT/protocol['source_snapshot']/path) == digest, 'snapshot changed: '+relative)


def register():
    cpu_guard()
    out = ROOT/OUTPUT
    out.mkdir(parents=True, exist_ok=False)
    identity = dict(pid=os.getpid(), command=sys.argv, started_utc=now())
    begin = time.monotonic()
    status = dict(identity)
    try:
        require(sha(ROOT/CACHE) == CACHE_SHA and sha(ROOT/CHECKPOINT) == CHECKPOINT_SHA,
                'pinned cache/checkpoint bytes differ')
        selection = json.loads((ROOT/SELECTION).read_text())
        historical = json.loads((ROOT/CACHE_PROTOCOL).read_text())
        rows = selected_metadata(selection, historical)
        sources = {p:sha(ROOT/p) for p in SOURCES}
        snapshot = Path(OUTPUT)/'source_snapshot'
        for relative, digest in sources.items():
            destination = ROOT/snapshot/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write((ROOT/relative).read_bytes())
            require(sha(destination) == digest, 'source changed during snapshot')
        protocol = dict(schema_version=VERSION, registered_utc=now(),
            input_hashes={CACHE:CACHE_SHA,CHECKPOINT:CHECKPOINT_SHA,
                          SELECTION:sha(ROOT/SELECTION),CACHE_PROTOCOL:sha(ROOT/CACHE_PROTOCOL)},
            source_hashes=sources, source_snapshot=str(snapshot), selected_metadata=rows,
            selected_metadata_sha256=json_sha(rows), n_windows=32, n_proteins=32,
            horizon_counts={'1':8,'4':8,'16':8,'64':8}, roles=list(ROLES),
            conditions=list(CONDITIONS), device='cpu', torch_threads=2,
            runtime=dict(torch=torch.__version__,python=sys.version),
            budget=dict(steps=200,optimizer='Adam',lr=.02,weight_decay=0.,gradient_clip=1.,
                        maximum_rotation_degrees=30.,initial_rotation_vector='zero',selection='final only'),
            objective=dict(nco_squared_euclidean_per_atom=1.,periodic_phi_psi=1.,
                c_n=1.,ca_c_n=4.,c_n_ca=4.,backbone_interresidue_clash=100.),
            scoring_mask='row.atom_mask & row.target_atom_mask & observed_mask[:,None]; fixed for all roles',
            precision='head float32 CPU; local rigid atom family and scoring float64',
            conditioning='explicit future-H and future-backbone TRAIN oracle; target torsions/coordinates enter optimization',
            inference='local constructive diagnostic; no global feasibility, H information bound, forecast repair or adoption claim',
            selection_policy='first32 immutable TRAIN cache rows; no replacements, weight search, best iterate or confidence intervals',
            cache_deserialized=False,checkpoint_deserialized=False,test_opened=False,bioemu2_rerun=False)
        verify(protocol)
        write_json(out/'protocol.json', protocol)
        status.update(status='completed',returncode=0,protocol_sha256=sha(out/'protocol.json'))
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-begin)
        write_json(out/'registration.execution.json',status)
    return protocol


def validate_cache(cache, protocol):
    p = cache['protocol']
    expected = dict(mode='train',dataset_role='training_readout_calibration',seed=42,
        n_windows=500,context_frames=8,max_residues=96,horizons=[1,4,16,64],
        n_forecast_samples=0,optimizer_steps=0,test_opened=False,bioemu2_rerun=False)
    require(cache['schema_version'] == 'backbone_readout_cache_v1'
            and all(p.get(k) == v for k,v in expected.items()), 'TRAIN cache contract differs')
    done = cache['completion']
    require(done['frozen_inputs_verified'] and done['model_sha256_before'] == done['model_sha256_after']
            == p['model_tensor_sha256'], 'TRAIN cache incomplete or changed model')
    require(len(cache['rows']) == 500 and [r['metadata'] for r in cache['rows'][:32]] == protocol['selected_metadata'],
            'cache prefix differs from preregistered identities')
    require(p == json.loads((ROOT/CACHE_PROTOCOL).read_text())['protocol'], 'embedded cache protocol differs')
    return cache['rows'][:32]


def validate_row(row):
    mask = row['observed_mask']
    require(mask.ndim == 1 and mask.dtype == torch.bool and 4 <= int(mask.sum()) <= len(mask) <= 96,
            'invalid observed support')
    n = len(mask)
    for key in ('embedding_aatype','embedding_residue_index','embedding_chain_id','chemical_aatype','geometry_row_ids'):
        require(row[key].shape == (n,) and row[key].dtype == torch.long, 'invalid '+key)
    for key in ('atom_mask','target_atom_mask'):
        require(row[key].shape == (n,4) and row[key].dtype == torch.bool, 'invalid '+key)
    shared = row['atom_mask'] & row['target_atom_mask'] & mask[:,None]
    require(bool(shared[mask].all()), 'pinned TRAIN rows require complete fixed backbone support')
    for key, shape in dict(oracle_H=(1,n,128),oracle_CA=(1,n,3),observed_R=(1,n,3,3),
                           observed_backbone=(1,n,4,3),target_backbone=(1,n,4,3)).items():
        value = row[key]
        require(value.shape == shape and value.dtype == torch.float32 and value.device.type == 'cpu', 'invalid '+key)
        valid = shared if key.endswith('backbone') else mask
        require(bool(torch.isfinite(value[:,valid]).all()), 'nonfinite valid '+key)
    require(row['metadata']['sampling_seed'] is None, 'unexpected TRAIN sampling seed')
    # Official scorer validates unique row IDs and nonbranching directed PSF topology.
    rama._topology(row['peptide_edges'],row['geometry_row_ids'],n)
    return shared


def load_head():
    checkpoint = torch.load(ROOT/CHECKPOINT,map_location='cpu',weights_only=False)
    require(checkpoint['step'] == 500 and checkpoint['arm'] == 'coordinate_transport'
            and checkpoint['frame_basis'] == transport.FRAME_BASIS
            and checkpoint['frozen_inputs_verified']
            and checkpoint['head_tensor_sha256'] == state_sha256(checkpoint['head'])
            and checkpoint['input_hashes'][CACHE] == CACHE_SHA, 'wrong frozen transport latest500 head')
    with torch.random.fork_rng(devices=[]):
        head = BackboneReadoutV2(BackboneReadoutConfig(**checkpoint['head_config']))
    head.load_state_dict(checkpoint['head'],strict=True)
    head.eval().requires_grad_(False)
    return head


@torch.no_grad()
def decode_condition(head, row, ca):
    moved = transport.transport(row,ca)
    output = head(row['oracle_H'],moved['frame_seed'],ca,row['embedding_aatype'][None],
                  row['chemical_aatype'][None],row['observed_mask'][None])
    needed = row['atom_mask'] & row['target_atom_mask'] & row['observed_mask'][:,None]
    require(bool(output['atom_mask'][:,needed].all()), 'head omitted fixed scoring support')
    require(torch.equal(output['backbone'][:,row['observed_mask'],1],ca[:,row['observed_mask']]), 'head changed fixed CA')
    return {k:v.detach().cpu().clone() for k,v in output.items()}, transport.transport_audit(moved,row['observed_mask'])


@torch.no_grad()
def explicit_edge_geometry(backbone, row, mask):
    """Actual PSF edge geometry including cis omega; strict invalids retained."""
    require(backbone.shape[0] == 1, 'one member per oracle role required')
    x = backbone[0].double()
    def angle(a,b,c):
        u,v = a-b,c-b
        nu,nv = torch.linalg.vector_norm(u),torch.linalg.vector_norm(v)
        if min(float(nu),float(nv)) <= rama.EPS_ANGSTROM:
            return None
        return float(torch.rad2deg(torch.acos(torch.clamp((u*v).sum()/(nu*nv),-1.,1.))))
    records=[]
    for i,j in row['peptide_edges'].tolist():
        record=dict(source=i,target=j)
        quads = dict(c_n=((i,2),(j,0)),ca_c_n=((i,1),(i,2),(j,0)),
                     c_n_ca=((i,2),(j,0),(j,1)),omega=((i,1),(i,2),(j,0),(j,1)))
        for name,indices in quads.items():
            if not all(bool(mask[r,a]) for r,a in indices):
                value,reason=None,'masked_or_missing_atoms'
            else:
                points=torch.stack([x[r,a] for r,a in indices])
                require(bool(torch.isfinite(points).all()), 'nonfinite scored peptide atoms')
                value = (float(torch.linalg.vector_norm(points[0]-points[1])) if name=='c_n' else
                         rama._dihedral(points) if name=='omega' else angle(*points))
                reason=None if value is not None else 'degenerate'
            record[name]=dict(value=value,excluded_reason=reason)
        value=record['omega']['value']
        record['omega']['cis']=None if value is None else -90. < value < 90.
        records.append(record)
    counts={}
    for name in ('c_n','ca_c_n','c_n_ca','omega'):
        counts[name]=dict(n_edges=len(records),n_valid=sum(r[name]['value'] is not None for r in records),
            n_masked=sum(r[name]['excluded_reason']=='masked_or_missing_atoms' for r in records),
            n_degenerate=sum(r[name]['excluded_reason']=='degenerate' for r in records))
    counts['omega']['cis_count']=sum(r['omega']['cis'] is True for r in records)
    counts['omega']['trans_count']=sum(r['omega']['cis'] is False for r in records)
    return dict(per_edge=records,counts=counts,units=dict(c_n='angstrom',angles='degrees'),
                cis_rule='-90 < omega < 90; endpoints trans; cis is reported, not equated with invalid geometry')


def score(backbone,row,role,ca):
    mask=row['atom_mask'] & row['target_atom_mask'] & row['observed_mask'][:,None]
    result=scoring.score_backbone(backbone,row,role=role,fixed_ca=ca)
    result['nco_and_geometry']=transport.stats.primary(result)
    result['ramachandran']=rama.ramachandran_summary(backbone,row['chemical_aatype'],
        residue_mask=row['observed_mask'],atom_mask=mask,peptide_edges=row['peptide_edges'],
        geometry_row_ids=row['geometry_row_ids'],include_per_residue=True)
    result['explicit_edge_geometry']=explicit_edge_geometry(backbone,row,mask)
    result['score_mask_sha256']=state_sha256({'mask':mask})
    return result


def diagnose_row(head,row,index,out):
    from protein_jepa_frame.models.stage_c.joint_frame_oracle_v1 import optimize_joint_frames
    mask=validate_row(row)
    roles={'target_md':score(row['target_backbone'].double(),row,'target_md',row['target_backbone'][...,1,:])}
    errors=[]; artifacts={}; optimization_summaries={}
    for condition in CONDITIONS:
        path=out/f'row_{index:02d}_{condition}.pt'
        attempt=dict(schema_version=VERSION,index=index,metadata=row['metadata'],condition=condition,
            row=row,score_mask=mask,conditioned_on_reference=True,status='started')
        try:
            ca=row['oracle_CA'] if condition=='decoded_oracle_ca' else row['target_backbone'][...,1,:]
            output,transport_audit=decode_condition(head,row,ca)
            attempt.update(head_output=output,transport=transport_audit,fixed_ca=ca.clone())
            initial=output['backbone'].double()
            roles[condition+'_baseline']=score(initial,row,condition+'_baseline',ca)
            optimized=optimize_joint_frames(initial,row['target_backbone'].double(),
                residue_mask=row['observed_mask'],atom_mask=mask,chemical_aatype=row['chemical_aatype'],
                geometry_row_ids=row['geometry_row_ids'],peptide_edges=row['peptide_edges'],
                initial_R=output['R'].double(),psi_logits=output['psi_logits'].clone())
            attempt['optimization']=optimized
            require(torch.equal(optimized['initial_backbone'],initial), 'optimizer changed initial witness')
            require(torch.equal(optimized['psi_logits'],output['psi_logits']), 'optimizer changed psi logits')
            require(torch.equal(optimized['final_backbone'][...,1,:],initial[...,1,:]), 'optimizer changed CA bytes')
            require([r['step'] for r in optimized['trace']]==list(range(201)), 'fixed200 trace incomplete')
            optimization_summaries[condition]=scoring._plain(dict(
                initial=optimized['trace'][0],final=optimized['trace'][-1],
                n_trace_states=len(optimized['trace']),final_losses=optimized.get('final_losses'),
                conventions=optimized.get('conventions')))
            roles[condition+'_corrected']=score(optimized['final_backbone'],row,condition+'_corrected',ca)
            attempt['status']='completed'
        except Exception as error:
            if hasattr(error,'partial'):
                attempt['optimization']=error.partial
            attempt.update(status='failed',error=f'{type(error).__name__}: {error}')
            errors.append(dict(index=index,condition=condition,error=attempt['error']))
        finally:
            save_tensor(path,attempt)
            artifacts[condition]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),status=attempt['status'])
    return dict(index=index,metadata=row['metadata'],roles=roles,errors=errors,artifacts=artifacts,
                optimization=optimization_summaries,
                score_mask_sha256=state_sha256({'mask':mask}))


def run():
    cpu_guard()
    out=ROOT/OUTPUT; protocol_path=out/'protocol.json'
    require(not any((out/name).exists() for name in ('started.json','execution.json','result.json')), 'execution already attempted')
    protocol_digest=sha(protocol_path)
    started=dict(pid=os.getpid(),command=sys.argv,started_utc=now(),protocol_sha256=protocol_digest)
    write_json(out/'started.json',started)
    begin=time.monotonic(); status=dict(started); rows=[]; errors=[]
    try:
        protocol=json.loads(protocol_path.read_text())
        require(protocol['schema_version']==VERSION, 'wrong protocol')
        registration=json.loads((out/'registration.execution.json').read_text())
        require(registration['status']=='completed' and registration['returncode']==0
                and registration['protocol_sha256']==protocol_digest, 'registration did not complete')
        verify(protocol)  # before any deserialization/model construction
        torch.set_num_threads(2)
        rng_before=torch.get_rng_state().clone()
        cache=torch.load(ROOT/CACHE,map_location='cpu',weights_only=False)
        selected=validate_cache(cache,protocol)
        head=load_head(); head_before=state_sha256(head)
        for index,row in enumerate(selected):
            try:
                record=diagnose_row(head,row,index,out)
                errors.extend(record['errors']); rows.append(record)
            except Exception as error:
                failure=dict(index=index,metadata=row.get('metadata'),error=f'{type(error).__name__}: {error}')
                errors.append(failure)
                failed_path=out/f'row_{index:02d}_failed.pt'
                save_tensor(failed_path,dict(row=row,failure=failure))
                rows.append(dict(failure,status='failed',artifact=dict(
                    path=str(failed_path.relative_to(ROOT)),sha256=sha(failed_path))))
            write_json(out/f'row_{index:02d}.json',rows[-1])
            print(json.dumps(dict(index=index,completed=len(rows),errors=len(errors))),flush=True)
        require(state_sha256(head)==head_before and all(p.grad is None for p in head.parameters()), 'frozen head changed')
        require(torch.equal(torch.get_rng_state(),rng_before), 'diagnostic consumed global CPU RNG')
        require(sha(protocol_path)==protocol_digest, 'protocol changed during execution')
        verify(protocol)
        result=dict(schema_version=VERSION,status='failed' if errors else 'completed',protocol_sha256=protocol_digest,
            per_window=rows,errors=errors,n_attempted_windows=len(rows),n_successful_windows=sum(not r.get('errors') and 'roles' in r for r in rows),
            model_before_sha256=head_before,model_after_sha256=state_sha256(head),
            cpu_rng_before_sha256=state_sha256({'rng':rng_before}),cpu_rng_after_sha256=state_sha256({'rng':torch.get_rng_state()}),
            head_optimizer_steps=0,local_optimizer_steps_per_successful_condition=200,
            test_opened=False,bioemu2_rerun=False,no_forecast_sampling=True,no_confidence_intervals=True)
        write_json(out/'result.json',result)
        status.update(result_sha256=sha(out/'result.json'),errors=errors)
        require(not errors and len(rows)==32 and all(set(r['roles'])==set(ROLES) for r in rows), 'diagnostic has preserved failed attempts')
        status.update(status='completed',returncode=0)
    except BaseException as error:
        status.update(status='failed',returncode=1,error=f'{type(error).__name__}: {error}',errors=errors)
        raise
    finally:
        status.update(completed_utc=now(),seconds=time.monotonic()-begin)
        write_json(out/'execution.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('register','run'))
    args=parser.parse_args()
    result=register() if args.mode=='register' else run()
    print(json.dumps(dict(mode=args.mode,status=result.get('status','registered'))),flush=True)


if __name__=='__main__':
    main()
