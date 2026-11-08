"""Frozen observed/oracle backbone readout features, with explicit PSF topology.

Train: a seed-42 permutation of all 5198 training proteins, first 500, one
uniform eligible trajectory and anchor per protein, cycling h=1/4/16/64.
Validation: exactly the registered standard200 keys, with 12 observed-only
endpoint forecasts. Unlike iter_windows' full-clip reader, this preparation
reads only eight observed frames, then ONE endpoint after forecast completion.
No optimizer, model selection, test manifest, or BioEmu2 execution.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluation.loaders import sha256_file
from protein_jepa_frame.config import DataConfig
from protein_jepa_frame.data.backbone_topology_v2 import load_mdcath_backbone_topology, clear_mdcath_topology_cache
from protein_jepa_frame.data.dataset import TrajectoryClipDataset
from protein_jepa_frame.data.manifest import load_manifest, TrajectoryRecord
from protein_jepa_frame.training.train_stage_b1a import _extract_backbone_frames
from protein_jepa_frame.utils.contracts import state_sha256
from scripts.diag_lasp_ca_ceiling import _window_seed
from scripts.eval_ca_backbone_diagnostic import align_target_backbone
from scripts.eval_ca_endpoint_scores import load_model
from scripts.eval_ca_ordered_paths import repository_sources, _snapshot_sources, _write_json_exclusive
from scripts.gate_ca_representation import audit_sources


REPO = Path(__file__).resolve().parents[1]
VERSION = 'backbone_readout_cache_v1'
CHECKPOINT = 'artifacts/hs_joint_20261101/spatial_probe_v1/spatial_exact.pt'
TRAIN_MANIFEST = 'artifacts/mdcath_manifest_full/train.jsonl'
VAL_MANIFEST = 'artifacts/mdcath_manifest_full/val.jsonl'
STANDARD200 = 'results/hs_joint_20261101/backbone_v2/spatial_exact_allres200_v2.json'
HORIZONS = (1, 4, 16, 64)
TEMPERATURES = (320., 348., 379., 413., 450.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def repo_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), 'cache inputs/output must be inside repository')
    require(path.name != 'test.jsonl' and 'bioemu' not in str(path).lower(), 'excluded input/output')
    return path


def approved_manifest(mode, manifest=None):
    require(mode in ('train', 'val'), 'mode must be train or val')
    approved = (REPO / (TRAIN_MANIFEST if mode == 'train' else VAL_MANIFEST)).resolve()
    resolved = approved if manifest is None else Path(manifest).resolve()
    require(resolved == approved, 'only the exact approved manifest for this mode is allowed')
    return resolved


@dataclass(frozen=True)
class WindowSpec:
    record: TrajectoryRecord
    anchor_frame: int
    horizon: int

    @property
    def protein_id(self): return self.record.protein_id

    @property
    def trajectory_id(self): return self.record.trajectory_id

    @property
    def target_frame(self): return self.anchor_frame + self.horizon

    @property
    def context_frames(self): return np.arange(self.anchor_frame-7, self.anchor_frame+1, dtype=np.int64)

    def metadata(self, mode):
        return dict(protein=self.protein_id, trajectory=self.trajectory_id,
            temperature_k=float(self.record.temperature_k), anchor_frame=self.anchor_frame,
            horizon=self.horizon, target_frame=self.target_frame,
            horizon_ps=self.horizon*float(self.record.frame_interval_ps),
            context_frames=self.context_frames.tolist(),
            sampling_seed=_window_seed(0, self) if mode == 'val' else None)

    def registration(self, mode):
        record = asdict(self.record)
        record['path'] = str(self.record.path.resolve())
        return dict(metadata=self.metadata(mode), record=record, record_sha256=json_hash(record))


def _groups(records):
    grouped = OrderedDict()
    seen = set()
    for record in records:
        require(record.trajectory_id not in seen, 'duplicate manifest trajectory identity')
        seen.add(record.trajectory_id)
        require(record.format.lower() == 'mdcath-hdf5', 'only mdCATH HDF trajectories are supported')
        require(isinstance(record.num_frames, int) and not isinstance(record.num_frames, bool)
                and record.num_frames > 0, 'explicit positive num_frames required; no implicit data scan')
        require(record.frame_interval_ps == 1000., 'fixed checkpoint requires the 1000 ps raw frame grid')
        require(record.temperature_k in TEMPERATURES, 'unexpected temperature in fixed panel manifest')
        grouped.setdefault(record.protein_id, []).append(record)
    return grouped


def select_train(records, *, population=5198, count=500):
    """All randomness is fixed before reading coordinates or applying smoke cap."""
    groups = _groups(records)
    require(len(groups) == population and 0 < count <= population, 'training protein population differs from registration')
    rng = np.random.default_rng(42)
    proteins = list(groups)
    chosen = [proteins[int(i)] for i in rng.permutation(len(proteins))[:count]]
    result = []
    for index, protein in enumerate(chosen):
        horizon = HORIZONS[index % 4]
        eligible = [record for record in groups[protein] if record.num_frames >= 8+horizon]
        require(bool(eligible), f'selected protein has no eligible trajectory: {protein}')
        record = eligible[int(rng.integers(len(eligible)))]
        anchor = int(rng.integers(7, record.num_frames-horizon))
        result.append(WindowSpec(record, anchor, horizon))
    return result


def select_val(records, reference_rows, *, count=200):
    """Metadata-only standard200 first-round schedule, checked against frozen keys.

    This is the first 50 protein groups of iter_windows' seed-0 Latin-square
    schedule: protein i selects condition i modulo its condition count; anchor
    7; one complete horizon group. Fail rather than skip an ineligible record.
    The historical artifact is used ONLY for keys, never scores or selection.
    """
    require(count == 200, 'the validation selection is the fixed full standard200 panel')
    groups = _groups(records)
    require(len(groups) >= 50 and len(reference_rows) == 200, 'standard200 reference population differs')
    result = []
    for index, records_for_protein in enumerate(list(groups.values())[:50]):
        record = records_for_protein[index % len(records_for_protein)]
        require(record.num_frames >= 72, 'registered validation record lacks the complete horizon group')
        result.extend(WindowSpec(record, 7, h) for h in HORIZONS)
    keys = ('protein','trajectory','temperature_k','anchor_frame','horizon','target_frame')
    actual = [tuple(window.metadata('val')[key] for key in keys) for window in result]
    expected = [tuple(row[key] for key in keys) for row in reference_rows]
    require(actual == expected, 'metadata-only schedule differs from frozen standard200 keys')
    require(Counter(w.record.temperature_k for w in result) == Counter({t:40 for t in TEMPERATURES}),
            'standard200 temperature allocation differs')
    return result


def budget(mode, value):
    complete = 500 if mode == 'train' else 200
    count = complete if value is None else value
    require(isinstance(count, int) and not isinstance(count, bool) and 4 <= count <= complete
            and count % 4 == 0, 'budget must be a complete four-window prefix within the fixed panel')
    return count


def file_stat(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                inode=stat.st_ino, device=stat.st_dev)


def verify_clip(clip, topology, spec, *, endpoint=False):
    frames = np.array([spec.target_frame]) if endpoint else spec.context_frames
    length = topology.n_residues
    require(clip['coords'].shape == (len(frames), length, 4, 3), 'legacy reader crop/backbone shape differs')
    require(clip['coords'].dtype == torch.float32, 'legacy reader coordinates must be float32')
    for name, old_name in (('embedding_aatype','aatype'), ('embedding_residue_index','residue_index'),
                           ('embedding_chain_id','chain_id')):
        require(torch.equal(getattr(topology, name), clip[old_name]), f'legacy {old_name} differs from topology')
    require(clip['atom_mask'].dtype == torch.bool and clip['atom_mask'].shape == (len(frames), length, 4),
            'legacy atom mask shape/dtype differs')
    # The static PDB mask is not allowed to conceal nonfinite MD atoms.
    require(torch.equal(clip['atom_mask'], topology.atom_mask[None].expand(len(frames), -1, -1)),
            'MD atom mask differs from verified PDB topology; no dropping/replacing window')
    require(torch.equal(clip['residue_mask'], clip['atom_mask'].all(-1)), 'legacy residue mask differs')
    require(bool(torch.isfinite(clip['coords'][clip['atom_mask']]).all()), 'nonfinite valid backbone atoms')
    expected_times = torch.from_numpy((spec.record.first_frame_time_ps
        + frames.astype(np.float32)*float(spec.record.frame_interval_ps)).astype(np.float32))
    require(torch.equal(clip['time_ps'], expected_times), 'actual physical timestamps differ from registered window')


def clip_fingerprint(clip):
    return state_sha256({key:clip[key] for key in ('coords','atom_mask','residue_mask',
                                                'aatype','residue_index','chain_id','time_ps')})


@torch.no_grad()
def prepare_row(model, spec, topology, reader, *, mode, device='cpu'):
    """reader(frames) is called with observation first, endpoint after sampling."""
    require(not model.training and model.prediction_kind == 'endpoint', 'strict eval-mode endpoint model required')
    require(float(model.kernel.cfg.step_ps) == float(spec.record.frame_interval_ps), 'model/data physical step differs')
    context = reader(spec.context_frames)
    verify_clip(context, topology, spec)
    observed_mask = context['residue_mask'].all(0)
    require(int(observed_mask.sum()) >= 4, 'fewer than four observed-valid residues')
    observed = context['coords'].to(device)
    mask = observed_mask[None].to(device)
    aa, ri, ci = [context[key][None].to(device) for key in ('aatype','residue_index','chain_id')]
    rotation, anchor = _extract_backbone_frames(observed[-1:])
    forecast = {}
    if mode == 'val':
        seed = _window_seed(0, spec)
        forecast_ca, forecast_h = model.sample(observed[None], mask, aa, ri, ci,
            torch.tensor([spec.record.temperature_k], dtype=observed.dtype, device=device),
            spec.horizon, 12, torch.Generator(device=device).manual_seed(seed))
        require(forecast_ca.shape == (1,12,len(observed_mask),3)
                and forecast_h.shape == (1,12,len(observed_mask),128)
                and model.last_observation_steps == [spec.horizon]*12, 'forecast endpoint contract differs')
        forecast = dict(forecast_H=forecast_h[0].detach().cpu().clone(),
                        forecast_CA=forecast_ca[0].detach().cpu().clone())
    else:
        require(mode == 'train', 'unknown cache mode')
    # No endpoint coordinate, endpoint mask, or future H is available above.
    anchor_h = model.bundle.encode(observed[-1:], mask, aa, ri, ci)
    target = reader(np.array([spec.target_frame], dtype=np.int64))
    verify_clip(target, topology, spec, endpoint=True)
    require(bool(target['atom_mask'][0,observed_mask].all()),
            'target lacks observed-valid complete backbone; no sample dropping')
    endpoint = target['coords'].to(device)
    oracle_h = model.bundle.encode(endpoint, mask, aa, ri, ci)
    oracle_ca = model.bundle.decode_ca(oracle_h, anchor_h, aa, ri, ci, rotation, anchor, mask)
    aligned = align_target_backbone(endpoint[0], anchor[0], mask[0])[None]
    tensors = dict(observed_R=rotation, observed_backbone=observed[-1:], target_backbone=aligned,
                   anchor_H=anchor_h, oracle_H=oracle_h, oracle_CA=oracle_ca, **forecast)
    expected = dict(observed_R=(1,len(observed_mask),3,3), observed_backbone=(1,len(observed_mask),4,3),
                    target_backbone=(1,len(observed_mask),4,3), anchor_H=(1,len(observed_mask),128),
                    oracle_H=(1,len(observed_mask),128), oracle_CA=(1,len(observed_mask),3))
    if mode == 'val': expected.update(forecast_H=(12,len(observed_mask),128), forecast_CA=(12,len(observed_mask),3))
    for name, value in tensors.items():
        require(tuple(value.shape) == expected[name] and value.dtype == torch.float32, f'{name} shape/dtype differs')
        require(bool(torch.isfinite(value[:,observed_mask.to(value.device)]).all()), f'nonfinite valid {name}')
    row = dict(metadata=spec.metadata(mode), observed_mask=observed_mask.clone(),
        **{key:getattr(topology,key).clone() for key in ('embedding_aatype','embedding_residue_index',
            'embedding_chain_id','chemical_aatype','geometry_row_ids','peptide_edges','atom_mask')},
        target_atom_mask=target['atom_mask'][0].clone(),
        **{key:value.detach().cpu().clone() for key,value in tensors.items()},
        topology=dict(sha256=topology.topology_sha256, **topology.as_metadata()),
        fingerprints=dict(observation=clip_fingerprint(context), raw_label=clip_fingerprint(target),
            aligned_label=state_sha256({'backbone':aligned}),
            oracle_features=state_sha256({'H':oracle_h,'CA':oracle_ca}),
            forecast=None if mode == 'train' else state_sha256(forecast)),
        boundary=dict(forecast_before_target_read=True, observed_frames=spec.context_frames.tolist(),
            label_frames=[spec.target_frame], future_reencoding_for_forecast=0,
            oracle_label_role='future endpoint for supervised readout calibration; never forecast input'))
    return row


def run(args):
    require(Path.cwd().resolve() == REPO, 'run from repository root for stable source registration')
    mode = args.mode
    manifest = approved_manifest(mode, args.manifest)
    require(repo_path(args.checkpoint) == (REPO/CHECKPOINT).resolve(), 'fixed spatial_exact checkpoint required')
    count = budget(mode, args.max_windows)
    require(args.torch_threads > 0, 'positive thread count required')
    out = repo_path(args.out)
    require(out.suffix == '.pt', 'cache output must have .pt extension')
    selection_file, protocol_file = out.with_suffix('.selection.json'), out.with_suffix('.protocol.json')
    snapshot = out.with_suffix('.source_snapshot')
    require(not any(p.exists() for p in (out, selection_file, protocol_file, snapshot)), 'refusing to overwrite cache evidence')
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(0)
    input_hashes = {}
    def track(path):
        path = repo_path(path)
        relative = str(path.relative_to(REPO))
        digest = sha256_file(path)
        require(relative not in input_hashes or input_hashes[relative] == digest, f'input changed: {relative}')
        input_hashes[relative] = digest
        return path
    track(manifest); track(args.checkpoint)
    records = load_manifest(manifest)
    if mode == 'train':
        full_selection = select_train(records)
    else:
        reference = json.loads(track(STANDARD200).read_text())
        full_selection = select_val(records, reference['per_window'])
    specs = full_selection[:count]
    registration = dict(schema_version=VERSION, registered_utc=datetime.now(timezone.utc).isoformat(),
        mode=mode, seed=42 if mode == 'train' else 0, n_windows=count,
        full_selection=[w.registration(mode) for w in full_selection],
        selected_prefix=count, input_hashes=dict(input_hashes),
        coordinates_read=False, model_loaded=False,
        selection='seed42 NumPy default_rng permutation in manifest first-protein order; uniform eligible record then anchor'
                  if mode == 'train' else 'metadata-only iter_windows first-round standard200 keys; checked against frozen full200 artifact')
    _write_json_exclusive(selection_file, registration)
    track(selection_file)
    topologies, data_stats = [], {}
    for spec in specs:
        path = str(spec.record.path.resolve())
        before = file_stat(path)
        topology = load_mdcath_backbone_topology(spec.record)
        topology = topology.subset(torch.arange(min(96,topology.n_residues)))
        require(file_stat(path) == before, 'data file changed during topology preflight')
        require(path not in data_stats or data_stats[path] == before, 'data file changed across preflight rows')
        data_stats[path] = before
        topologies.append(topology)
    checkpoint = torch.load(repo_path(args.checkpoint), map_location='cpu', weights_only=False)
    model = load_model(str(repo_path(args.checkpoint)), checkpoint, args.device).eval().requires_grad_(False)
    # Only manifest identities are opened for inherited split provenance.
    provenance = audit_sources(checkpoint, checkpoint['component_checkpoint'], TRAIN_MANIFEST, VAL_MANIFEST)
    for audit in provenance['training_splits']: track(audit['manifest'])
    track(VAL_MANIFEST)
    for item in provenance['upstream_checkpoints'].values(): track(item['path'])
    sources = repository_sources() + ['protein_jepa_frame/vendored/PROVENANCE.json',
        'protein_jepa_frame/vendored/openfold/resources/stereo_chemical_props.txt']
    source_hashes = {path:sha256_file(REPO/path) for path in sorted(set(sources))}
    model_hash = state_sha256(model)
    protocol = dict(schema_version=VERSION, mode=mode,
        dataset_role='training_readout_calibration' if mode == 'train' else 'development_validation',
        checkpoint=CHECKPOINT, manifest=str(manifest.relative_to(REPO)),
        seed=42 if mode == 'train' else 0, n_windows=count,
        full_panel=count == len(full_selection), complete_panel_windows=len(full_selection),
        horizons=list(HORIZONS), max_residues=96, context_frames=8, n_forecast_samples=12 if mode == 'val' else 0,
        forecast_observed_only=True, before_target_read=True,
        optimizer_steps=0, test_opened=False, bioemu2_rerun=False,
        selection_file=str(selection_file.relative_to(REPO)), selection_sha256=sha256_file(selection_file),
        frame_interval_ps=1000., model_tensor_sha256=model_hash,
        label_gauge='raw future CA fitted to observed anchor CA once; same R,t applied to all future backbone atoms',
        oracle_H='encode raw future backbone with observed mask and unchanged legacy embedding IDs',
        oracle_CA='existing bundle.decode_ca(oracle_H, anchor_H, observed frames and legacy metadata)',
        chemistry='explicit PSF/PDB topology v2 chemical IDs; never substituted into learned embedding IDs',
        data_hash_scope='PDB/PSF metadata hashes plus exact consumed tensor fingerprints; full HDF byte hash not computed; file stat checked before/after',
        mask_policy='context all eight frames; topology/reader aa,ri,ci,atommask exact; target complete at observed-valid residues or fail',
        registered_utc=datetime.now(timezone.utc).isoformat())
    provenance.update(data_file_stats=data_stats,
        topology_metadata=[dict(metadata=spec.metadata(mode), topology_sha256=topology.topology_sha256,
                                metadata_hashes=topology.metadata_hashes, source_identity=topology.source_identity)
                           for spec,topology in zip(specs,topologies)],
        boundary='only train/val manifests; inherited split audits read manifest identities, never validation coordinates in train mode')
    _snapshot_sources(snapshot, source_hashes)
    _write_json_exclusive(protocol_file, dict(protocol=protocol, source_hashes=source_hashes,
        input_hashes=input_hashes, provenance=provenance))
    rows = []
    for index, (spec,topology) in enumerate(zip(specs,topologies)):
        reader = TrajectoryClipDataset([spec.record], DataConfig(num_frames=8,max_residues=96))
        rows.append(prepare_row(model, spec, topology, lambda frames:reader._read_clip(spec.record,frames),
                                mode=mode, device=args.device))
        if (index+1) % 4 == 0:
            print(f'backbone-cache {mode}: {index+1}/{count}', flush=True)
    require(len(rows) == count and len({tuple(row['metadata'][k] for k in
        ('protein','trajectory','anchor_frame','horizon')) for row in rows}) == count, 'incomplete/duplicate cache rows')
    require(state_sha256(model) == model_hash and all(p.grad is None for p in model.parameters()), 'model changed or accumulated gradients')
    for path, expected in input_hashes.items(): require(sha256_file(REPO/path) == expected, f'input changed: {path}')
    for path, expected in source_hashes.items(): require(sha256_file(REPO/path) == expected, f'source changed: {path}')
    for path, expected in data_stats.items(): require(file_stat(path) == expected, f'data file changed: {path}')
    clear_mdcath_topology_cache()
    for spec, topology in zip(specs,topologies):
        reread = load_mdcath_backbone_topology(spec.record).subset(torch.arange(topology.n_residues))
        require(reread.topology_sha256 == topology.topology_sha256, 'input PDB/PSF metadata changed')
    result = dict(schema_version=VERSION, protocol=protocol, source_hashes=source_hashes,
        input_hashes=input_hashes, provenance=provenance, rows=rows,
        completion=dict(completed_utc=datetime.now(timezone.utc).isoformat(), frozen_inputs_verified=True,
                        model_sha256_before=model_hash, model_sha256_after=state_sha256(model)))
    with out.open('xb') as handle: torch.save(result, handle)
    print(json.dumps(dict(output=str(out), n_windows=len(rows), sha256=sha256_file(out))), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train','val'), required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-windows', type=int)
    parser.add_argument('--torch-threads', type=int, default=2)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
