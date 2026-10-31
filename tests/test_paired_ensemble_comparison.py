import json
import pytest
from scripts.compare_paired_ensembles import index_rows


def test_window_identity_includes_anchor_and_rejects_duplicates(tmp_path):
    p = tmp_path / 'rows.json'
    row = dict(protein='p', traj='r', temperature_k=320., anchor_frame=3, horizon=4.)
    p.write_text(json.dumps([row, dict(row, anchor_frame=4)]))
    assert len(index_rows(p)) == 2
    p.write_text(json.dumps([row, row]))
    with pytest.raises(ValueError, match='duplicate'):
        index_rows(p)
