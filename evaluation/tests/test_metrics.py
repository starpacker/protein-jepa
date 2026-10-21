import math
import numpy as np
import pytest
from evaluation.metrics import WindowResult, aggregate, paired_bootstrap_ci,paired_variant_comparisons


def row(protein, anchor, rmsd, persist, variant="model"):
    return WindowResult(protein, protein+"_rep0", 320., 1, variant, rmsd, persist,
                        anchor_frame=anchor, target_frame=anchor+1)


def test_persistence_self_pairing_and_equal_protein_weight():
    rows = [row("a", 0, 1., 1., "persist"), row("a", 1, 3., 3., "persist"),
            row("b", 0, 8., 8., "persist")]
    result = aggregate(rows)[320.][1]["persist"]
    assert result["delta_ci"] == [0., 0., 0.]
    assert result["n_proteins"] == 2
    assert result["n_windows"] == 3
    assert result["rmsd_mean"] == 5.  # (mean([1,3])+8)/2


def test_delta_uses_each_original_anchor():
    rows = [row("a", 0, 1., 1.), row("a", 1, 3., 3.)]
    assert aggregate(rows)[320.][1]["model"]["delta_ci"] == [0., 0., 0.]
    assert not aggregate(rows)[320.][1]["model"]["delta_significant"]


def test_duplicate_window_and_bad_persistence_rejected():
    a = row("a", 0, 1., 1.)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([a, a])
    with pytest.raises(ValueError, match="persistence mismatch"):
        aggregate([a, row("a", 0, 2., 2., "persist")])


def test_nan_pairs_dropped_jointly_and_lengths_checked():
    assert paired_bootstrap_ci([1., np.nan, 4.], [1., 99., 4.]) == (0., 0., 0.)
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.], [1., 2.])


def test_model_comparisons_pair_exact_windows_before_equal_protein_weight():
    rows=[]
    for protein,anchors,effect in [('a',range(5),-1.),('b',range(1),3.)]:
        for anchor in anchors:
            rows.extend([row(protein,anchor,4.+effect,9.,'h_plus_s'),
                         row(protein,anchor,4.,9.,'h_markov')])
    comparison=next(r for r in paired_variant_comparisons(rows) if r['metric']=='rmsd')
    assert comparison['delta_a_minus_b']==1. # equal proteins, not six windows
    with pytest.raises(ValueError,match='different window sets'):
        paired_variant_comparisons(rows[:-1])
