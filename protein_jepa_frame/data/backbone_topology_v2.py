"""Explicit mdCATH backbone identity and PSF connectivity, independent of model IDs.

Only PDB/PSF metadata and the protein atom-count metadata are read. Coordinates
never determine connectivity. Existing dataset embedding aa/ri/ci and backbone
atom selection are checked against the old parser and preserved exactly.
Insertion-coded residues retain complete UIDs and unique geometry_row_ids;
only explicit PSF C_i--N_j bonds generate directed peptide_edges[i,j].

Chemical IDs use OpenFold order. The six explicit histidine protonation aliases
HSD/HSE/HSP/HID/HIE/HIP map to chemical HIS while legacy embedding IDs remain
unchanged. MSE retains the existing backbone-only MET interpretation. Other
unrecognized residue names remain chemical UNK, never inferred from geometry.

Cached entries are keyed by file identity/stat and metadata paths, not trajectory
coordinates. Each public load returns independent tensor/dict copies. Replacing
a metadata file changes the cache key; clear_mdcath_topology_cache() explicitly
invalidates entries. Hashes identify the actual metadata bytes and parser code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch

from protein_jepa_frame.data.dataset import TrajectoryClipDataset, THREE_TO_ONE
from protein_jepa_frame.data.manifest import TrajectoryRecord
from protein_jepa_frame.vendored.openfold.np import residue_constants as rc


SCHEMA_VERSION = 'mdcath_backbone_topology_v2.0'
ATOM_NAMES = ('N','CA','C','O')
HISTIDINE_ALIASES = ('HSD','HSE','HSP','HID','HIE','HIP')
CHEMICAL_ALIAS_POLICY = {'histidine_aliases': HISTIDINE_ALIASES,
                         'MSE': 'MET_backbone_only', 'other_unknown': 'UNK20'}


def _hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _json_hash(value):
    return _hash_bytes(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def _text(value, name):
    if isinstance(value,bytes):
        return value.decode('utf-8',errors='strict')
    if isinstance(value,str):
        return value
    raise ValueError(f'{name} must be a scalar UTF-8 text dataset')


@dataclass(frozen=True)
class ResidueUID:
    source_path: str
    domain_id: str
    segment_id: str
    author_chain_id: str
    author_residue_number: int
    insertion_code: str


@dataclass(frozen=True)
class BackboneTopologyV2:
    residue_uids: tuple[ResidueUID,...]
    residue_names: tuple[str,...]
    embedding_aatype: torch.Tensor
    embedding_residue_index: torch.Tensor
    embedding_chain_id: torch.Tensor
    chemical_aatype: torch.Tensor
    atom_indices: torch.Tensor
    atom_mask: torch.Tensor
    geometry_row_ids: torch.Tensor
    peptide_edges: torch.Tensor
    peptide_atom_indices: torch.Tensor
    metadata_hashes: dict[str,str]
    source_identity: dict
    schema_version: str = SCHEMA_VERSION

    @property
    def n_residues(self):
        return len(self.residue_uids)

    def subset(self, indices):
        """Keep requested local rows in order; explicitly drop/remap crop edges.

        atom_indices stay in the ORIGINAL HDF protein atom coordinate axis.
        geometry_row_ids retain their original identities, never define edges.
        Model embedding IDs are sliced, not renumbered. Duplicates are rejected.
        """
        indices = (torch.empty(0,dtype=torch.long) if isinstance(indices,(tuple,list)) and not indices
                   else torch.as_tensor(indices,device='cpu'))
        if indices.ndim != 1 or indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
            raise ValueError('subset indices must be a one-dimensional integer sequence')
        indices = indices.long()
        if bool(((indices < 0) | (indices >= self.n_residues)).any()) or len(indices.unique()) != len(indices):
            raise ValueError('subset indices must be unique and within current local rows')
        mapping = torch.full((self.n_residues,),-1,dtype=torch.long)
        mapping[indices] = torch.arange(len(indices))
        mapped = mapping[self.peptide_edges]
        keep = (mapped >= 0).all(-1)
        kwargs = {name:getattr(self,name)[indices].clone() for name in (
            'embedding_aatype','embedding_residue_index','embedding_chain_id','chemical_aatype',
            'atom_indices','atom_mask','geometry_row_ids')}
        return BackboneTopologyV2(
            residue_uids=tuple(self.residue_uids[i] for i in indices.tolist()),
            residue_names=tuple(self.residue_names[i] for i in indices.tolist()),
            **kwargs, peptide_edges=mapped[keep].clone(),
            peptide_atom_indices=self.peptide_atom_indices[keep].clone(),
            metadata_hashes=dict(self.metadata_hashes),source_identity=copy.deepcopy(self.source_identity))

    @property
    def topology_sha256(self):
        """Content hash of this view, including ordering, typing and explicit edges."""
        return _json_hash(self.as_metadata())

    def as_metadata(self):
        return dict(schema_version=self.schema_version,source_identity=copy.deepcopy(self.source_identity),
            metadata_hashes=dict(self.metadata_hashes),
            residue_uids=[asdict(uid) for uid in self.residue_uids],residue_names=list(self.residue_names),
            **{name:getattr(self,name).tolist() for name in (
                'embedding_aatype','embedding_residue_index','embedding_chain_id','chemical_aatype',
                'atom_indices','atom_mask','geometry_row_ids','peptide_edges','peptide_atom_indices')})


def _psf_records(psf_text):
    lines = psf_text.splitlines()
    if not lines or not lines[0].strip().startswith('PSF'):
        raise ValueError('missing PSF file header')
    def section(tag):
        found = [i for i,line in enumerate(lines) if tag in line]
        if len(found) != 1:
            raise ValueError(f'PSF requires exactly one {tag} section')
        index = found[0]
        try:
            count = int(lines[index].split()[0])
        except (IndexError,ValueError) as error:
            raise ValueError(f'invalid PSF {tag} count') from error
        if count < 0:
            raise ValueError(f'negative PSF {tag} count')
        return index,count
    atom_start,n_atoms = section('!NATOM')
    atoms = []
    for line in lines[atom_start+1:atom_start+1+n_atoms]:
        fields = line.split()
        if len(fields) < 6:
            raise ValueError('malformed or truncated PSF atom record')
        try:
            atom_id = int(fields[0])
        except ValueError as error:
            raise ValueError('invalid PSF atom ID') from error
        atoms.append((atom_id,*fields[1:5]))
    ids = [row[0] for row in atoms]
    if len(atoms) != n_atoms or len(set(ids)) != n_atoms or any(i <= 0 for i in ids):
        raise ValueError('PSF atom IDs must be complete, positive and unique')
    bond_start,n_bonds = section('!NBOND')
    tokens = []
    try:
        for line in lines[bond_start+1:]:
            if '!' in line:
                break
            tokens.extend(int(value) for value in line.split())
    except ValueError as error:
        raise ValueError('invalid PSF bond record') from error
    if len(tokens) != 2*n_bonds:
        raise ValueError('PSF bond count differs from records')
    id_to_ordinal = {atom_id:i for i,atom_id in enumerate(ids)}
    bonds = set()
    for a,b in zip(tokens[::2],tokens[1::2]):
        if a not in id_to_ordinal or b not in id_to_ordinal or a == b:
            raise ValueError('PSF bond has unknown atom or self edge')
        pair = tuple(sorted((id_to_ordinal[a],id_to_ordinal[b])))
        if pair in bonds:
            raise ValueError('duplicate undirected PSF bond')
        bonds.add(pair)
    return atoms,bonds


def parse_mdcath_backbone_topology(pdb_text, psf_text, *, source_path, domain_id, expected_atom_count):
    """Parse text metadata with strict protein-prefix identity and legacy checks."""
    pdb_text,psf_text = _text(pdb_text,'PDB'),_text(psf_text,'PSF')
    if isinstance(expected_atom_count,bool) or not isinstance(expected_atom_count,int) or expected_atom_count < 1:
        raise ValueError('expected protein atom count must be a positive integer')
    pdb_atoms = [line for line in pdb_text.splitlines() if line.startswith(('ATOM  ','HETATM'))]
    if len(pdb_atoms) != expected_atom_count:
        raise ValueError('PDB atom count differs from protein metadata')
    psf_atoms,bonds = _psf_records(psf_text)
    if len(psf_atoms) < len(pdb_atoms):
        raise ValueError('PSF has fewer atoms than the protein PDB')
    residues = {}
    legacy_keys = {}
    for ordinal,line in enumerate(pdb_atoms):
        if len(line) < 76:
            raise ValueError('PDB atom record lacks required segment/identity columns')
        try:
            serial,number = int(line[6:11]),int(line[22:26])
        except ValueError as error:
            raise ValueError('PDB atom/residue number is malformed') from error
        atom_name,altloc,name = line[12:16].strip(),line[16:17].strip(),line[17:20].strip().upper()
        chain,icode,segment = line[21:22].strip() or '_',line[26:27].strip(),line[72:76].strip()
        if (serial,segment,str(number)+icode,name,atom_name) != psf_atoms[ordinal]:
            raise ValueError(f'protein PDB/PSF prefix identity mismatch at atom ordinal {ordinal}')
        uid = ResidueUID(str(source_path),str(domain_id),segment,chain,number,icode)
        legacy_key = (chain,number,icode)
        if legacy_key in legacy_keys and legacy_keys[legacy_key] != uid:
            raise ValueError('full residue UID cannot map one-to-one to legacy loader rows')
        legacy_keys[legacy_key] = uid
        row = residues.setdefault(uid,dict(name=name,atoms={}))
        if row['name'] != name:
            raise ValueError('conflicting residue names for the same UID')
        if atom_name in ATOM_NAMES and altloc in {'','A'}:
            row['atoms'].setdefault(atom_name,ordinal)
    uids = tuple(residues)
    names = tuple(row['name'] for row in residues.values())
    indices = torch.tensor([[row['atoms'].get(atom,-1) for atom in ATOM_NAMES]
                            for row in residues.values()],dtype=torch.long)
    mask = indices >= 0
    legacy = TrajectoryClipDataset._parse_pdb_backbone(pdb_text,expected_atom_count)
    if (not torch.equal(indices,torch.from_numpy(legacy.atom_indices))
            or not torch.equal(mask,torch.from_numpy(legacy.atom_mask))):
        raise ValueError('backbone atom mapping differs from the legacy loader')
    chemical = torch.tensor([rc.restype_order.get(
        'H' if name in HISTIDINE_ALIASES else THREE_TO_ONE.get(name,'X'),20) for name in names],dtype=torch.long)
    carbon_rows = {int(index):row for row,index in enumerate(indices[:,2]) if index >= 0}
    nitrogen_rows = {int(index):row for row,index in enumerate(indices[:,0]) if index >= 0}
    edges = set()
    for a,b in bonds:
        for c,n in ((a,b),(b,a)):
            if c in carbon_rows and n in nitrogen_rows:
                edge = (carbon_rows[c],nitrogen_rows[n])
                if edge[0] == edge[1]:
                    raise ValueError('unexpected same-residue carbonyl C--N PSF bond')
                edges.add(edge)
    peptide = torch.tensor(sorted(edges),dtype=torch.long).reshape(-1,2)
    peptide_atoms = torch.stack((indices[peptide[:,0],2],indices[peptide[:,1],0]),dim=-1)
    source = Path(__file__).resolve()
    from protein_jepa_frame.data import dataset as legacy_module
    hashes = dict(pdb_text_sha256=_hash_bytes(pdb_text.encode()),psf_text_sha256=_hash_bytes(psf_text.encode()),
        topology_parser_sha256=_hash_bytes(source.read_bytes()),
        legacy_dataset_parser_sha256=_hash_bytes(Path(legacy_module.__file__).read_bytes()),
        openfold_constants_sha256=_hash_bytes(Path(rc.__file__).read_bytes()),
        chemical_alias_policy_sha256=_json_hash(CHEMICAL_ALIAS_POLICY))
    identity = dict(path=str(source_path),domain_id=str(domain_id),protein_atom_count=len(pdb_atoms),
        psf_atom_count=len(psf_atoms),psf_bond_count=len(bonds),protein_prefix_identity_verified=True,
        chemical_alias_policy=copy.deepcopy(CHEMICAL_ALIAS_POLICY))
    return BackboneTopologyV2(uids,names,torch.from_numpy(legacy.aatype).clone(),
        torch.from_numpy(legacy.residue_index).clone(),torch.from_numpy(legacy.chain_id).clone(),
        chemical,indices,mask,torch.arange(len(uids)),peptide,peptide_atoms,hashes,identity)


@lru_cache(maxsize=128)
def _cached_topology(path, inode, size, mtime_ns, ctime_ns, domain_id, pdb_dataset, psf_dataset, atom_count):
    import h5py
    with h5py.File(path,'r') as handle:
        if pdb_dataset not in handle or psf_dataset not in handle:
            raise ValueError('mdCATH topology requires both protein PDB and PSF metadata')
        if atom_count is None:
            residue_atoms = handle[f'{domain_id}/resid']
            if len(residue_atoms.shape) != 1:
                raise ValueError('mdCATH resid metadata must be one-dimensional')
            expected = len(residue_atoms)
        else:
            expected = atom_count
        result = parse_mdcath_backbone_topology(handle[pdb_dataset][()],handle[psf_dataset][()],
            source_path=path,domain_id=domain_id,expected_atom_count=expected)
    result.source_identity.update(pdb_dataset=pdb_dataset,psf_dataset=psf_dataset)
    status = Path(path).stat()
    if (status.st_ino,status.st_size,status.st_mtime_ns,status.st_ctime_ns) != (inode,size,mtime_ns,ctime_ns):
        raise RuntimeError('mdCATH metadata file changed while parsing')
    return result


def clear_mdcath_topology_cache():
    _cached_topology.cache_clear()


def load_mdcath_backbone_topology(record: TrajectoryRecord) -> BackboneTopologyV2:
    """Read metadata only; return a safe copy independent of the internal cache."""
    if record.format.lower() != 'mdcath-hdf5':
        raise ValueError('this topology adapter only supports mdCATH HDF5 records')
    path = record.path.resolve()
    status = path.stat()
    domain = record.domain_id or record.protein_id
    metadata = record.metadata or {}
    pdb = str(metadata.get('protein_pdb_dataset') or f'{domain}/pdbProteinAtoms').lstrip('/')
    psf = str(metadata.get('protein_psf_dataset') or f'{domain}/psf').lstrip('/')
    count = metadata.get('atom_count')
    if count is not None and (isinstance(count,bool) or not isinstance(count,int) or count < 1):
        raise ValueError('record atom_count must be a positive integer')
    result = _cached_topology(str(path),status.st_ino,status.st_size,status.st_mtime_ns,status.st_ctime_ns,
                              domain,pdb,psf,count)
    # Inputs with the same physical metadata share the cache across replicas.
    # Tensor/dict copies prevent one training caller from corrupting another.
    copy = result.subset(torch.arange(result.n_residues))
    copy.source_identity.update(protein_id=record.protein_id)
    return copy
