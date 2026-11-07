"""Synthetic geometry/topology tests; official numerical oracle is separate."""
import math

import pytest
import torch

from evaluation import ramachandran_reference_v1 as rama


def place(a,b,c,torsion,length=1.4,angle=115.):
    # Independent internal-coordinate construction with explicit torsion sign.
    u=(c-b)/torch.linalg.vector_norm(c-b)
    v=a-b
    v=v-(v*u).sum()*u
    v=v/torch.linalg.vector_norm(v)
    w=torch.linalg.cross(u,v)
    t,q=math.radians(torsion),math.radians(angle)
    return c+length*(-math.cos(q)*u+math.sin(q)*(math.cos(t)*v+math.sin(t)*w))


def backbone(omega=180.,phi=-60.,psi=-40.,next_omega=180.):
    x=torch.zeros(3,4,3,dtype=torch.float64)
    x[0,1]=torch.tensor([-1.,0.,0.])
    x[0,2]=torch.tensor([0.,0.,0.])
    x[1,0]=torch.tensor([0.,1.,0.])
    x[1,1]=place(x[0,1],x[0,2],x[1,0],omega)
    x[1,2]=place(x[0,2],x[1,0],x[1,1],phi)
    x[2,0]=place(x[1,0],x[1,1],x[1,2],psi)
    x[2,1]=place(x[1,1],x[1,2],x[2,0],next_omega)
    x[2,2]=place(x[1,2],x[2,0],x[2,1],-70.)
    x[0,0]=torch.tensor([-1.,-.7,.3])
    x[:,3]=x[:,2]+torch.tensor([.2,.6,.5])
    return x


def analyze(x=None,aa=None,**kwargs):
    if x is None:x=backbone()
    if aa is None:aa=torch.zeros(3,dtype=torch.long)
    kwargs.setdefault('peptide_edges',torch.tensor([[0,1],[1,2]]))
    kwargs.setdefault('geometry_row_ids',torch.tensor([101,104,999]))
    return rama.ramachandran_summary(x,aa,**kwargs)


def middle(result):
    return result['per_sample'][0]['per_residue'][1]


@pytest.mark.parametrize('current,next_aa,omega,next_omega,expected',[
    (0,0,180.,0.,'general'),
    (rama.GLY,rama.PRO,180.,0.,'glycine'),
    (rama.PRO,rama.PRO,0.,180.,'cis-proline'),
    (rama.PRO,rama.PRO,180.,0.,'trans-proline'),
    (rama.ILE,rama.PRO,180.,180.,'pre-proline'),
    (rama.VAL,0,180.,180.,'isoleucine or valine'),
    (rama.ILE,0,180.,180.,'isoleucine or valine'),
])
def test_real_geometry_six_classes_priority_and_previous_omega(current,next_aa,omega,next_omega,expected):
    result=analyze(backbone(omega=omega,next_omega=next_omega),torch.tensor([0,current,next_aa]))
    row=middle(result)
    assert row['rama_class']==expected
    assert row['phi_degrees']==pytest.approx(-60.,abs=1e-12)
    assert row['psi_degrees']==pytest.approx(-40.,abs=1e-12)
    if current==rama.PRO:
        assert abs(row['previous_omega_degrees'])==pytest.approx(abs(omega),abs=1e-12)
    direct=rama.score_angles(-60.,-40.,expected)
    assert row['score']==pytest.approx(float(direct['score']),abs=1e-15)
    assert result['totals']['n_scored']==1
    assert result['per_sample'][0]['exclusion_counts']['topology_incomplete']==2
    assert set(result['per_sample'][0]['per_class'])==set(rama.CLASSES)


@pytest.mark.parametrize('omega,expected',[(-90.,3),(90.,3),(-89.999,2),(89.999,2),(-180.,3),(180.,3)])
def test_cis_boundaries_are_strict_and_torsion_sign(omega,expected):
    assert rama._class_of(rama.PRO,0,omega)==expected
    row=middle(analyze(backbone(omega=omega,phi=64.,psi=-123.),torch.tensor([0,rama.PRO,0])))
    assert row['phi_degrees']==pytest.approx(64.,abs=1e-12)
    assert row['psi_degrees']==pytest.approx(-123.,abs=1e-12)
    assert row['rama_class']==rama.CLASSES[expected]


def test_topology_local_indices_and_explicit_permutation():
    x=backbone(); aa=torch.tensor([0,rama.GLY,0])
    original=analyze(x,aa)
    # Same geometry under arbitrary row storage order and nonconsecutive identities.
    order=torch.tensor([2,0,1])
    permuted=analyze(x[order],aa[order],geometry_row_ids=torch.tensor([999,101,104]),
                     peptide_edges=torch.tensor([[1,2],[2,0]]))
    row=permuted['per_sample'][0]['per_residue'][2]
    assert row['score']==middle(original)['score'] and row['geometry_row_id']==104
    assert row['previous_row']==1 and row['next_row']==0
    with pytest.raises(ValueError,match='outside local'):
        analyze(x,aa,peptide_edges=torch.tensor([[101,104],[104,999]]))


def test_topology_gap_empty_and_masked_neighbor_never_bridge():
    gap=analyze(peptide_edges=torch.tensor([[0,1]]))
    assert gap['totals']['n_scored']==0 and gap['mean']['non_outlier_fraction'] is None
    empty=analyze(peptide_edges=torch.empty(0,2,dtype=torch.long))
    assert empty['per_sample'][0]['issue_counts']['missing_previous']==3
    assert empty['per_sample'][0]['issue_counts']['missing_next']==3
    masked=analyze(residue_mask=torch.tensor([False,True,True]))
    assert middle(masked)['excluded_reason']=='neighbor_masked'
    assert masked['totals']['n_scored']==0


@pytest.mark.parametrize('edges,ids,pattern',[
    ([[0,0]],[1,2,3],'self'),
    ([[0,1],[0,1]],[1,2,3],'duplicate'),
    ([[0,1],[0,2]],[1,2,3],'multiple incoming/outgoing'),
    ([[0,2],[1,2]],[1,2,3],'multiple incoming/outgoing'),
    ([[0,3]],[1,2,3],'outside local'),
    ([[-1,1]],[1,2,3],'outside local'),
    ([[0,1]],[1,1,3],'unique'),
])
def test_invalid_topology_rejected(edges,ids,pattern):
    with pytest.raises(ValueError,match=pattern):
        analyze(peptide_edges=torch.tensor(edges),geometry_row_ids=torch.tensor(ids))


def test_missing_previous_ca_excludes_pro_not_general_and_o_not_required():
    mask=torch.ones(3,4,dtype=torch.bool); mask[0,1]=False; mask[:,3]=False
    x=backbone(omega=0.); x[~mask]=float('nan')
    pro=analyze(x,torch.tensor([0,rama.PRO,0]),atom_mask=mask)
    assert middle(pro)['excluded_reason']=='missing_atoms'
    assert pro['per_sample'][0]['issue_counts']['pro_omega_atoms_missing']==1
    assert pro['totals']['n_scored']==0
    assert analyze(x,atom_mask=mask)['totals']['n_scored']==1
    assert pro['conventions']['pro_missing_omega'].startswith('excluded')


def test_missing_atoms_and_degenerate_torsions_are_counted():
    am=torch.ones(3,4,dtype=torch.bool); am[1,1]=False
    missing=analyze(atom_mask=am)['per_sample'][0]
    assert missing['exclusion_counts']['missing_atoms']==1
    assert missing['issue_counts']['phi_atoms_missing']==1 and missing['issue_counts']['psi_atoms_missing']==1
    x=backbone(); x[1,2]=x[1,1]
    degenerate=analyze(x)['per_sample'][0]
    assert degenerate['exclusion_counts']['degenerate']==1
    assert degenerate['issue_counts']['phi_degenerate']==1 and degenerate['issue_counts']['psi_degenerate']==1
    x=backbone(); x[0,1]=x[0,2]
    pro=analyze(x,torch.tensor([0,rama.PRO,0]))['per_sample'][0]
    assert pro['issue_counts']['pro_omega_degenerate']==1
    assert pro['per_residue'][1]['category'] is None


def test_unknown_chemical_identity_is_not_general_or_gly_fallback():
    assert middle(analyze(aa=torch.tensor([0,20,0])))['excluded_reason']=='unknown_current'
    assert middle(analyze(aa=torch.tensor([0,0,20])))['excluded_reason']=='unknown_next'
    assert middle(analyze(aa=torch.tensor([20,rama.GLY,20])))['rama_class']=='glycine'
    assert middle(analyze(aa=torch.tensor([20,rama.PRO,20])))['rama_class']=='trans-proline'
    with pytest.raises(ValueError,match='OpenFold'):
        analyze(aa=torch.tensor([0,21,0]))


def test_masked_nan_valid_nonfinite_and_strict_mask_dtype():
    x=backbone(); x[0]=float('nan')
    result=analyze(x,aa=torch.tensor([-99,0,0]),residue_mask=torch.tensor([False,True,True]))
    assert result['totals']['n_scored']==0
    assert result['per_sample'][0]['exclusion_counts']['masked']==1
    with pytest.raises(ValueError,match='nonfinite'):
        analyze(x)
    with pytest.raises(ValueError,match='boolean'):
        analyze(residue_mask=torch.ones(3,dtype=torch.long))


def test_rigid_invariance_no_grad_and_input_unchanged():
    x=backbone(omega=12.,phi=-74.,psi=133.).requires_grad_()
    before=x.detach().clone()
    q,_=torch.linalg.qr(torch.tensor([[1.,2.,3.],[3.,-1.,2.],[1.,4.,-2.]],dtype=torch.float64))
    q[:,0]*=torch.linalg.det(q)
    moved=x@q.T+torch.tensor([22.,-14.,7.])
    first=middle(analyze(x,torch.tensor([0,rama.PRO,0])))
    second=middle(analyze(moved,torch.tensor([0,rama.PRO,0])))
    for key in ('phi_degrees','psi_degrees','previous_omega_degrees','score'):
        assert first[key]==pytest.approx(second[key],abs=1e-12)
    assert torch.equal(x.detach(),before)
    output=rama.score_angles(torch.tensor(-60.,requires_grad=True),-40.,0)
    assert not output['score'].requires_grad and output['score'].dtype==torch.float64


def test_four_categories_denominators_and_batched_empty_samples():
    counts=rama._counts([0,1,2,2])
    assert counts['favored_fraction']==.5 and counts['allowed_only_fraction']==.25
    assert counts['non_outlier_fraction']==.75 and counts['outlier_fraction']==.25
    x=backbone()[None].expand(2,-1,-1,-1).clone()
    result=analyze(x,residue_mask=torch.tensor([[True,True,True],[False,False,False]]))
    assert result['n_samples']==2 and result['totals']['n_scored']==1
    assert result['per_sample'][1]['non_outlier_fraction'] is None
    assert result['mean']['non_outlier_fraction']==result['per_sample'][0]['non_outlier_fraction']
    assert result['totals']['n_scored']+result['totals']['n_excluded']==result['totals']['n_residues']


def test_batch_broadcast_ambiguity_and_explicit_axes():
    x=backbone()[None,None].expand(2,2,-1,-1,-1)
    with pytest.raises(ValueError,match='ambiguous'):
        analyze(x,aa=torch.zeros(2,3,dtype=torch.long))
    result=analyze(x,aa=torch.zeros(2,1,3,dtype=torch.long),include_per_residue=False)
    assert result['n_samples']==4 and result['totals']['n_scored']==4
    assert all('per_residue' not in r for r in result['per_sample'])


def test_score_rejects_unknown_classes_nonfinite_and_bad_ids():
    for cls in ('not-a-class',6,-1,True,1.0):
        with pytest.raises(ValueError):rama.score_angles(-60.,-40.,cls)
    with pytest.raises(ValueError,match='finite'):
        rama.score_angles(float('nan'),0.,0)
    with pytest.raises(ValueError,match='integers'):
        analyze(geometry_row_ids=torch.tensor([1.,2.,3.]))
    with pytest.raises(ValueError,match='integers'):
        analyze(aa=torch.zeros(3))
