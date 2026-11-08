"""Synthetic CPU metadata tests; no real MD coordinates or split manifests."""
from dataclasses import replace
import os

import h5py
import pytest
import torch

from protein_jepa_frame.data.backbone_topology_v2 import (
    HISTIDINE_ALIASES, clear_mdcath_topology_cache, load_mdcath_backbone_topology,
    parse_mdcath_backbone_topology,
)
from protein_jepa_frame.data.manifest import TrajectoryRecord
from protein_jepa_frame.vendored.openfold.np import residue_constants as rc


def text_metadata(residues=None, edges=None, missing=()):
    residues = residues or [('GLY','0',1,'A','P0'),('SER','0',1,'B','P0'),
                            ('HSD','0',1,'C','P0'),('ARG','0',2,'','P0')]
    if edges is None: edges = [(i,i+1) for i in range(len(residues)-1)]
    pdb,psf_atoms,indices = [],[],{}
    for row,(name,chain,number,icode,segment) in enumerate(residues):
        for atom in ('N','CA','C','O'):
            if (row,atom) in missing: continue
            serial = len(pdb)+1
            line = [' ']*80
            for start,stop,value in ((0,6,'ATOM  '),(6,11,f'{serial:5d}'),(12,16,f'{atom:>4}'),
                    (17,20,f'{name:>3}'),(21,22,chain),(22,26,f'{number:4d}'),(26,27,icode or ' '),
                    (30,38,'   0.000'),(38,46,'   0.000'),(46,54,'   0.000'),
                    (72,76,f'{segment:<4}'),(76,78,f'{atom[0]:>2}')):
                line[start:stop] = value
            pdb.append(''.join(line))
            psf_atoms.append(f'{serial:8d} {segment} {number}{icode} {name} {atom} TYPE 0.0 12.0 0')
            indices[row,atom] = serial
    bonds = [(indices[i,'C'],indices[j,'N']) for i,j in edges]
    psf = ('PSF\n\n'+f'{len(psf_atoms):8d} !NATOM\n'+'\n'.join(psf_atoms)+'\n\n'
           +f'{len(bonds):8d} !NBOND: bonds\n'
           +'\n'.join(f'{a:8d}{b:8d}' for a,b in bonds)+'\n\n       0 !NTHETA: angles\n')
    return '\n'.join(pdb)+'\n',psf,len(pdb)


def parse(pdb=None,psf=None,count=None):
    if pdb is None:pdb,psf,count=text_metadata()
    return parse_mdcath_backbone_topology(pdb,psf,source_path='synthetic.h5',domain_id='domain',expected_atom_count=count)


def write_hdf(tmp_path, *, with_psf=True):
    pdb,psf,count = text_metadata()
    path = tmp_path/'metadata.h5'
    with h5py.File(path,'w') as handle:
        domain = handle.create_group('domain')
        domain.create_dataset('pdbProteinAtoms',data=pdb.encode())
        if with_psf:domain.create_dataset('psf',data=psf.encode())
        domain.create_dataset('resid',data=[1]*count)
        domain.create_dataset('coords',shape=(2,count,3),dtype='f4')
    return TrajectoryRecord(path=path,protein_id='p',trajectory_id='p_T320_R0',domain_id='domain',format='mdcath-hdf5')


def test_insertion_codes_keep_full_UID_and_original_embedding_ids():
    topology = parse()
    assert topology.embedding_residue_index.tolist() == [1,1,1,2]
    assert topology.embedding_chain_id.tolist() == [0,0,0,0]
    assert topology.geometry_row_ids.tolist() == [0,1,2,3]
    assert len(set(topology.residue_uids)) == 4
    assert [uid.insertion_code for uid in topology.residue_uids] == ['A','B','C','']
    assert topology.embedding_aatype[2] == 20
    assert topology.chemical_aatype[2] == rc.restype_order['H']
    assert topology.peptide_edges.tolist() == [[0,1],[1,2],[2,3]]
    assert topology.peptide_atom_indices.tolist() == [[2,4],[6,8],[10,12]]
    assert topology.source_identity['protein_prefix_identity_verified']
    assert all(len(value)==64 for value in topology.metadata_hashes.values())


def test_histidine_aliases_change_chemical_ids_only_and_unknown_remains_unknown():
    names = [*HISTIDINE_ALIASES,'HIS','MSE','ZZZ']
    pdb,psf,count = text_metadata([(name,'A',i+1,'','PRO') for i,name in enumerate(names)])
    topology = parse(pdb,psf,count)
    assert topology.embedding_aatype[:6].tolist() == [20]*6
    assert topology.chemical_aatype[:7].tolist() == [rc.restype_order['H']]*7
    assert topology.chemical_aatype[-2:].tolist() == [rc.restype_order['M'],20]
    assert topology.atom_mask.all()


def test_explicit_edges_allow_cross_chain_number_gaps_and_do_not_invent_neighbors():
    residues = [('ALA','A',-2,'','P0'),('ALA','B',47,'','P1'),('ALA','B',48,'','P1')]
    topology = parse(*text_metadata(residues,edges=[(0,1),(2,0)]))
    assert topology.embedding_residue_index.tolist() == [-2,47,48]
    assert topology.embedding_chain_id.tolist() == [0,1,1]
    assert topology.peptide_edges.tolist() == [[0,1],[2,0]]
    assert [1,2] not in topology.peptide_edges.tolist()
    disconnected = parse(*text_metadata(residues,edges=[]))
    assert disconnected.peptide_edges.shape == (0,2)


def test_subset_preserves_order_identity_atom_axis_and_remaps_only_retained_edges():
    full = parse()
    sub = full.subset([3,1,2])
    assert sub.geometry_row_ids.tolist() == [3,1,2]
    assert sub.embedding_residue_index.tolist() == [2,1,1]
    assert sub.peptide_edges.tolist() == [[1,2],[2,0]]
    assert sub.peptide_atom_indices.tolist() == [[6,8],[10,12]]
    assert torch.equal(sub.atom_indices,full.atom_indices[[3,1,2]])
    assert sub.metadata_hashes == full.metadata_hashes
    assert sub.topology_sha256 != full.topology_sha256
    again = sub.subset([2,0])
    assert again.geometry_row_ids.tolist() == [2,3]
    assert again.peptide_edges.tolist() == [[0,1]]
    empty = full.subset([])
    assert empty.atom_mask.shape == (0,4) and empty.peptide_edges.shape == (0,2)
    for invalid in ([0,0],[-1],[4],[1.],torch.tensor([True,False])):
        with pytest.raises(ValueError):full.subset(invalid)


def test_missing_O_preserves_other_atoms_and_explicit_connection():
    topology = parse(*text_metadata(missing=[(1,'O')]))
    assert topology.atom_indices[1,3] == -1
    assert topology.atom_mask[1].tolist() == [True,True,True,False]
    assert topology.peptide_edges.tolist() == [[0,1],[1,2],[2,3]]


@pytest.mark.parametrize('fault', ['missing_psf','prefix_name','prefix_order','bond_count',
                                 'bond_atom','duplicate_bond','same_residue_bond','legacy_UID_merge'])
def test_malformed_or_mismatched_metadata_fail_closed(fault):
    pdb,psf,count = text_metadata()
    if fault == 'missing_psf':psf=''
    elif fault == 'prefix_name':psf=psf.replace('GLY N TYPE','ALA N TYPE',1)
    elif fault == 'prefix_order':
        lines=psf.splitlines();idx=next(i for i,l in enumerate(lines) if '!NATOM' in l)
        lines[idx+1],lines[idx+2]=lines[idx+2],lines[idx+1];psf='\n'.join(lines)
    elif fault == 'bond_count':psf=psf.replace('       3 !NBOND','       4 !NBOND')
    elif fault == 'bond_atom':psf=psf.replace('       3       5','     999       5')
    elif fault == 'duplicate_bond':psf=psf.replace('       7       9','       3       5')
    elif fault == 'same_residue_bond':psf=psf.replace('       3       5','       3       1')
    else:pdb,psf,count=text_metadata([('ALA','A',1,'','P0'),('ALA','A',1,'','P1')])
    with pytest.raises(ValueError):parse(pdb,psf,count)


def test_loader_requires_psf_and_does_not_read_coordinates(tmp_path,monkeypatch):
    clear_mdcath_topology_cache()
    record = write_hdf(tmp_path)
    original = h5py.Dataset.__getitem__
    reads=[]
    def guard(dataset,key):
        reads.append(dataset.name)
        assert not dataset.name.endswith('/coords')
        return original(dataset,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',guard)
    result = load_mdcath_backbone_topology(record)
    assert result.n_residues == 4
    assert reads == ['/domain/pdbProteinAtoms','/domain/psf']
    assert result.source_identity['path'] == str(record.path.resolve())
    assert result.source_identity['domain_id'] == 'domain'
    assert result.source_identity['protein_id'] == 'p'
    with pytest.raises(ValueError,match='supports'):
        load_mdcath_backbone_topology(replace(record,format='npz'))
    clear_mdcath_topology_cache()
    with h5py.File(record.path,'a') as handle:del handle['domain/psf']
    with pytest.raises(ValueError,match='both'):
        load_mdcath_backbone_topology(record)


def test_cache_returns_independent_copies_and_file_stat_change_invalidates(tmp_path,monkeypatch):
    from protein_jepa_frame.data import backbone_topology_v2 as module
    clear_mdcath_topology_cache()
    record = write_hdf(tmp_path)
    original = module.parse_mdcath_backbone_topology
    calls=[]
    def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(module,'parse_mdcath_backbone_topology',counted)
    first = load_mdcath_backbone_topology(record)
    expected = first.topology_sha256
    first.embedding_residue_index[0]=999
    first.metadata_hashes['pdb_text_sha256']='changed'
    first.source_identity['chemical_alias_policy']['other_unknown']='changed'
    second=load_mdcath_backbone_topology(replace(record,trajectory_id='p_T450_R4'))
    assert len(calls)==1 and second.topology_sha256==expected
    status=record.path.stat()
    os.utime(record.path,ns=(status.st_atime_ns,status.st_mtime_ns+1000000))
    third=load_mdcath_backbone_topology(record)
    assert len(calls)==2 and third.topology_sha256==expected
