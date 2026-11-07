"""Fixed-budget, cached raw-H / frozen-CA backbone readout calibration.

This runner never loads an encoder, dynamics sampler, validation trajectory or
test split. Both arms train the same independent head on the same cached rows.
Only the geometry arm adds the preregistered explicit-PSF-edge losses.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from protein_jepa_frame.models.backbone_readout_v2 import BackboneReadoutConfig, BackboneReadoutV2, HEAD_VERSION
from protein_jepa_frame.models.stage_c.backbone_objectives_v3 import backbone_geometry_losses_v3
from protein_jepa_frame.utils.contracts import state_sha256


REPO = Path(__file__).resolve().parents[1]
VERSION = 'fixed_ca_backbone_readout_training_v1'
LEGACY = 'artifacts/stage_b1a_relpos_anchored_1000dom/checkpoint_best.pt'
LEGACY_SHA = '5c02ad80b155edd853d788798d8512e757764f09f2fb9e8e97db06ccc2d88b0a'
WEIGHTS = dict(c_n=1., ca_c_n=4., c_n_ca=4., backbone_interresidue_clash=100.)
SOURCES = (
    'docs/BACKBONE_READOUT_PILOT_V2_20261101.md',
    'run_scripts/run_backbone_readout_pilot.py',
    'protein_jepa_frame/models/backbone_readout_v2.py',
    'protein_jepa_frame/models/stage_b1a/model.py',
    'protein_jepa_frame/models/stage_c/backbone_objectives_v2.py',
    'protein_jepa_frame/models/stage_c/backbone_objectives_v3.py',
    'protein_jepa_frame/models/stage_c/structure_decoder.py',
    'protein_jepa_frame/models/stage_c/geometry.py',
    'protein_jepa_frame/models/stage_c/openfold_atom37.py',
    'protein_jepa_frame/utils/contracts.py',
    'protein_jepa_frame/vendored/openfold/np/residue_constants.py',
    'protein_jepa_frame/vendored/openfold/resources/stereo_chemical_props.txt',
    'protein_jepa_frame/vendored/openfold/utils/feats.py',
    'protein_jepa_frame/vendored/openfold/utils/rigid_utils.py',
    'protein_jepa_frame/vendored/openfold/utils/tensor_utils.py',
    'protein_jepa_frame/vendored/openfold/data/data_transforms.py',
    'protein_jepa_frame/vendored/PROVENANCE.json',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), 'paths must be inside repository')
    text = str(path.relative_to(REPO))
    require(path.name != 'test.jsonl' and 'bioemu' not in text.lower(), 'excluded input/output')
    require(not text.startswith(('results/standard_eval/stopped_best/', 'results/sota_frozen_20261024_backup/')), 'protected path')
    return path


def sha(path):
    h = hashlib.sha256()
    with local(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write_json(path, value):
    with path.open('x') as f:
        json.dump(plain(value), f, indent=2, allow_nan=False)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def track(self, path, expected=None):
        path = local(path)
        key, digest = str(path.relative_to(REPO)), sha(path)
        require(expected is None or digest == expected, f'input hash mismatch: {key}')
        require(key not in self.hashes or self.hashes[key] == digest, f'input changed: {key}')
        self.hashes[key] = digest
        return path

    def declared(self, value):
        for category in ('input_hashes', 'source_hashes'):
            require(isinstance(value.get(category), dict) and value[category], f'missing {category}')
            for path, digest in value[category].items():
                self.track(path, digest)

    def verify(self):
        for path, digest in self.hashes.items():
            require(sha(path) == digest, f'input/source changed during execution: {path}')


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_head(path, tracked):
    value = torch.load(tracked.track(path), map_location='cpu', weights_only=False)
    require(value['head_version'] == HEAD_VERSION and value['step'] == 0, 'shared zero-update head required')
    tracked.declared(value)
    with torch.random.fork_rng(devices=[]):
        head = BackboneReadoutV2(BackboneReadoutConfig(**value['head_config']))
    head.load_state_dict(value['head'], strict=True)
    head.initialization = value['initialization']
    require(all(torch.isfinite(x).all() for x in head.state_dict().values()), 'nonfinite initialization')
    return head, value


def load_training_cache(path, tracked, allow_smoke=False):
    value = torch.load(tracked.track(path), map_location='cpu', weights_only=False)
    require(value['schema_version'] == 'backbone_readout_cache_v1', 'cache schema differs')
    p = value['protocol']
    expected = dict(mode='train', dataset_role='training_readout_calibration',
                    context_frames=8, max_residues=96, seed=42, horizons=[1,4,16,64],
                    n_forecast_samples=0, test_opened=False, bioemu2_rerun=False, optimizer_steps=0)
    require(all(p.get(k) == v for k,v in expected.items()), 'training cache protocol differs')
    rows = value['rows']
    require(len(rows) == p['n_windows'], 'cache row count differs')
    require(len(rows) == 500 or (allow_smoke and len(rows) in (4,32)), 'formal train cache must have 500 rows')
    require(len({r['metadata']['protein'] for r in rows}) == len(rows), 'training proteins must be unique')
    require([r['metadata']['horizon'] for r in rows] == [[1,4,16,64][i%4] for i in range(len(rows))], 'horizon order differs')
    require(value['completion']['frozen_inputs_verified'], 'cache inputs were not verified')
    require(value['completion']['model_sha256_before'] == value['completion']['model_sha256_after']
            == p['model_tensor_sha256'], 'cache model changed')
    for row in rows:
        mask = row['observed_mask']
        length = len(mask)
        require(length <= 96, 'cache exceeds residue budget')
        require(mask.dtype == torch.bool and mask.ndim == 1 and int(mask.sum()) >= 4, 'invalid observed mask')
        shapes = dict(oracle_H=(1,length,128), oracle_CA=(1,length,3),
            observed_R=(1,length,3,3), target_backbone=(1,length,4,3))
        for k,shape in shapes.items():
            x = row[k]
            require(x.shape == shape and x.dtype == torch.float32 and x.device.type == 'cpu', f'invalid cached {k}')
            valid = mask[:,None] & row['target_atom_mask'] if k == 'target_backbone' else mask
            require(bool(torch.isfinite(x[:,valid]).all()), f'nonfinite cached {k}')
        require(bool((row['target_atom_mask'][mask] & row['atom_mask'][mask]).all()), 'registered rows require all observed backbone atoms')
    tracked.declared(value)
    return value


def actual_losses(head, row):
    device = next(head.parameters()).device
    # .to() never marks the cached tensors as trainable; neither H nor CA is a
    # parameter of this independent head. Future masks enter the loss only.
    r = {k:v.detach().to(device) if torch.is_tensor(v) else v for k,v in row.items()}
    mask = r['observed_mask']
    output = head(r['oracle_H'], r['observed_R'], r['oracle_CA'],
                  r['embedding_aatype'][None], r['chemical_aatype'][None], mask[None])
    bb = output['backbone']
    require(torch.equal(bb[:,mask,1], r['oracle_CA'][:,mask]), 'head changed frozen CA')
    atoms = r['atom_mask'] & r['target_atom_mask'] & mask[:,None]
    require(bool(output['atom_mask'][:,atoms].all()), 'head omits a target atom')
    scored = atoms.clone()
    scored[:,1] = False
    require(bool(scored.any()), 'no N/C/O labels')
    difference = torch.where(scored[None,:,:,None], bb - r['target_backbone'], 0.)
    coordinate = difference.square().sum() / scored.sum()
    geom = backbone_geometry_losses_v3(bb, chemical_aatype=r['chemical_aatype'],
        atom_mask=r['atom_mask'], residue_mask=mask, peptide_edges=r['peptide_edges'],
        geometry_row_ids=r['geometry_row_ids'])
    combined = sum(WEIGHTS[k] * geom['losses'][k] for k in WEIGHTS)
    require(bool(torch.isfinite(coordinate)) and bool(torch.isfinite(combined)), 'nonfinite objective')
    return coordinate, combined, geom


def grad_norm(loss, params, retain_graph=False):
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=False)
    require(all(torch.isfinite(g).all() for g in grads), 'nonfinite calibration gradient')
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))


def scale_from_norms(coordinate, geometry):
    require(all(math.isfinite(x) and x > 0 for x in (coordinate,geometry)), 'invalid gradient scale norms')
    raw = .5 * coordinate / geometry
    require(math.isfinite(raw), 'invalid gradient scale ratio')
    return raw, min(10.,max(.1,raw))


def validate_scale(scale, rows):
    n = min(32,len(rows))
    recorded = scale['per_row']
    require(scale['protocol']['n_calibration_rows'] == n and len(recorded) == n, 'gradient calibration budget differs')
    require([r['metadata'] for r in recorded] == [r['metadata'] for r in rows[:n]], 'calibration row order differs')
    norms = [[float(r[k]) for r in recorded] for k in ('coordinate_grad_norm','weighted_geometry_grad_norm')]
    require(all(math.isfinite(x) and x >= 0 for group in norms for x in group), 'invalid per-row gradient norm')
    c,g = [sum(group)/n for group in norms]
    raw,coefficient = scale_from_norms(c,g)
    expected = dict(mean_coordinate_grad_norm=c,mean_weighted_geometry_grad_norm=g,
                    unclipped_scale=raw,scale=coefficient)
    require(all(math.isclose(float(scale[k]),v,rel_tol=1e-12,abs_tol=1e-12) for k,v in expected.items()), 'saved gradient scale differs from registered formula')
    return coefficient


def lr_at(step, steps):
    warm = min(25, steps)
    if step <= warm:
        return 1e-4 * step / warm
    fraction = (step - warm) / (steps - warm)
    return 1e-5 + .5 * (1e-4-1e-5) * (1 + math.cos(math.pi * fraction))


def register(args, tracked, **extras):
    source_hashes = {p:sha(p) for p in SOURCES}
    for p,d in source_hashes.items():
        tracked.track(p,d)
    protocol = dict(schema_version=VERSION, mode=args.mode,
        registered_utc=datetime.now(timezone.utc).isoformat(), seed=42,
        formal=not args.allow_smoke, optimizer='AdamW', learning_rate=1e-4,
        final_learning_rate=1e-5, warmup_steps=25, weight_decay=1e-4, gradient_clip=1.,
        coordinate_objective='mean squared Euclidean N/C/O coordinate error in Angstrom^2; no per-row re-alignment; fixed CA excluded',
        geometry_weights=WEIGHTS, scale_formula='clip(0.5 * mean(coord_parameter_grad_norm) / mean(weighted_geometry_parameter_grad_norm), 0.1, 10)',
        calibration_rows='first 32 training-cache rows, eval mode, zero updates',
        checkpoint_selection='fixed latest; no validation selection',
        encoder_CA_dynamics_frozen=True, inputs='raw oracle H and frozen current oracle CA',
        validation_used_for_training=False, test_opened=False, bioemu2_rerun=False,
        parameter_precision='float32; no AMP; TF32 disabled', **extras)
    out = local(args.out)
    if args.mode == 'train':
        require(not out.exists(), 'refusing existing training directory')
        out.mkdir(parents=True)
        protocol_path, snapshot = out/'protocol.json', out/'source_snapshot'
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        protocol_path, snapshot = out.with_suffix('.protocol.json'), out.with_suffix('.source_snapshot')
        require(not any(p.exists() for p in (out,protocol_path,snapshot)), 'refusing output overwrite')
    snapshot.mkdir()
    for p,d in source_hashes.items():
        dest = snapshot/p
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write((REPO/p).read_bytes())
        require(sha(dest) == d, 'source snapshot differs')
    info = dict(protocol=protocol, input_hashes=dict(tracked.hashes), source_hashes=source_hashes)
    write_json(protocol_path, info)
    return out, info


def run(args):
    torch.set_num_threads(2)
    seed_all()
    tracked = Inputs()
    if args.mode == 'init':
        path = tracked.track(LEGACY, LEGACY_SHA)
        head = BackboneReadoutV2.from_legacy_checkpoint(path, expected_sha256=LEGACY_SHA)
        out,info = register(args, tracked, optimizer_steps=0)
        tracked.verify()
        payload = dict(head.checkpoint_dict(), **info, step=0,
            head_tensor_sha256=state_sha256(head), frozen_inputs_verified=True)
        with out.open('xb') as f:
            torch.save(payload,f)
        print(json.dumps(dict(output=str(out), sha256=sha(out), step=0)), flush=True)
        return
    require(args.cache and args.initial, '--cache and --initial required')
    cache = load_training_cache(args.cache,tracked,args.allow_smoke)
    head,initial = load_head(args.initial,tracked)
    head = head.to(args.device)
    params = [p for p in head.parameters() if p.requires_grad]
    before = state_sha256(head)
    if args.mode == 'calibrate':
        n = min(32,len(cache['rows']))
        out,info = register(args,tracked, n_calibration_rows=n, optimizer_steps=0,
            train_cache=str(local(args.cache).relative_to(REPO)), initial=str(local(args.initial).relative_to(REPO)))
        head.eval()
        rows = []
        for row in cache['rows'][:n]:
            coord,geom,details = actual_losses(head,row)
            cn = grad_norm(coord,params,retain_graph=True)
            gn = grad_norm(geom,params)
            rows.append(dict(metadata=row['metadata'], coordinate_grad_norm=cn,
                weighted_geometry_grad_norm=gn, coordinate_loss=float(coord.detach()),
                geometry_losses=plain(details['losses'])))
        c,g = (sum(r[k] for r in rows)/n for k in ('coordinate_grad_norm','weighted_geometry_grad_norm'))
        raw,scale = scale_from_norms(c,g)
        require(state_sha256(head) == before and all(p.grad is None for p in head.parameters()), 'calibration mutated head')
        tracked.verify()
        write_json(out,dict(**info, scale=scale, unclipped_scale=raw,
            mean_coordinate_grad_norm=c, mean_weighted_geometry_grad_norm=g,
            per_row=rows, head_tensor_sha256=before, frozen_inputs_verified=True))
        print(json.dumps(dict(output=str(out),scale=scale,rows=n)),flush=True)
        return
    require(args.mode == 'train' and args.scale and args.arm, 'train needs --scale and --arm')
    require(args.steps == len(cache['rows']), 'one pass only: steps must equal cache rows')
    scale = json.loads(tracked.track(args.scale).read_text())
    require(scale['frozen_inputs_verified'] and scale['head_tensor_sha256'] == before, 'gradient scale initialization differs')
    require(scale['protocol']['mode'] == 'calibrate' and scale['protocol']['geometry_weights'] == WEIGHTS, 'scale protocol differs')
    for p in (args.cache,args.initial):
        key = str(local(p).relative_to(REPO))
        require(scale['input_hashes'].get(key) == tracked.hashes[key], 'gradient scale uses different cache/initialization')
    tracked.declared(scale)
    coefficient = validate_scale(scale,cache['rows'])
    lam = coefficient if args.arm == 'geometry' else 0.
    require(math.isfinite(lam) and 0 <= lam <= 10, 'invalid geometry coefficient')
    out,info = register(args,tracked, arm=args.arm, steps=args.steps,
        geometry_lambda=lam, train_cache=str(local(args.cache).relative_to(REPO)),
        initial=str(local(args.initial).relative_to(REPO)), optimizer_steps=args.steps,
        scale_file=str(local(args.scale).relative_to(REPO)),
        scale_sha256=tracked.hashes[str(local(args.scale).relative_to(REPO))],
        train_row_order='cache order, one row per update, one pass')
    initial_tensors = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    optimizer = torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    # Reset AFTER construction/loading so both arms use the same dropout draws.
    seed_all()
    head.train()
    with (out/'metrics.jsonl').open('x') as stream:
        for step,row in enumerate(cache['rows'],1):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr'] = lr_at(step,args.steps)
            coord,geom,details = actual_losses(head,row)
            objective = coord + lam*geom
            objective.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), 'missing/nonfinite trained gradient')
            norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
            record = dict(step=step,metadata=row['metadata'],lr=optimizer.param_groups[0]['lr'],
                coordinate_loss=float(coord.detach()), weighted_geometry_loss=float(geom.detach()),
                geometry_losses=plain(details['losses']), geometry_counts=plain(details['counts']),
                objective=float(objective.detach()), gradient_norm_before_clip=float(norm))
            stream.write(json.dumps(record,allow_nan=False)+'\n')
            stream.flush()
            if step % 25 == 0 or step == args.steps:
                print(json.dumps(record,allow_nan=False),flush=True)
    require(all(torch.isfinite(v).all() for v in head.state_dict().values()), 'nonfinite final head')
    tracked.verify()
    final = state_sha256(head)
    delta = math.sqrt(sum(float((v.detach().cpu().double()-initial_tensors[k].double()).square().sum()) for k,v in head.state_dict().items()))
    require(final != before and delta > 0, 'head did not update')
    payload = dict(head.checkpoint_dict(), **info, step=args.steps, arm=args.arm,
        optimizer=optimizer.state_dict(), head_tensor_sha256=final, initial_head_tensor_sha256=before,
        parameter_delta_l2=delta, frozen_inputs_verified=True)
    checkpoint = out/'checkpoint_latest.pt'
    with checkpoint.open('xb') as f:
        torch.save(payload,f)
    write_json(out/'completion.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        step=args.steps,arm=args.arm,checkpoint_sha256=sha(checkpoint),
        head_tensor_sha256=final,parameter_delta_l2=delta,frozen_inputs_verified=True,
        source_hashes=info['source_hashes'], input_hashes=info['input_hashes']))
    print(json.dumps(dict(output=str(checkpoint),step=args.steps,sha256=sha(checkpoint))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['init','calibrate','train'],required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--cache')
    parser.add_argument('--initial')
    parser.add_argument('--scale')
    parser.add_argument('--arm',choices=['coordinate','geometry'])
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--allow-smoke',action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
