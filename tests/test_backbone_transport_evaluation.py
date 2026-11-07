"""Serialization-only evaluator tests; synthetic data, no cache/model loading."""
import ast
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval_backbone_transport_pilot as evaluator


def test_nested_tuple_list_conventions_accept_without_changing_numeric_values():
    live = dict(conventions=dict(bond=(1.329, .014), box=((-180., -30.), (-90., 45.))),
                per_sample=[dict(loss=.017, count=12)], backbone_sha256='a'*64)
    saved = json.loads(json.dumps(live))
    assert live != saved
    evaluator.require_canonical_json_equal(live, saved, 'must serialize identically')
    assert live['conventions']['bond'] == (1.329, .014)
    assert evaluator.canonical_json(live) == saved


@pytest.mark.parametrize('value', [0., .017, 1.329, -180.])
def test_one_ulp_numeric_changes_are_rejected(value):
    live = {'legacy_geometry': {'mean': {'loss': value}}}
    saved = {'legacy_geometry': {'mean': {'loss': math.nextafter(value, math.inf)}}}
    with pytest.raises(ValueError, match='numeric differs'):
        evaluator.require_canonical_json_equal(live, saved, 'numeric differs')


@pytest.mark.parametrize('field', ['backbone_sha256', 'fixed_ca_sha256'])
def test_hash_changes_are_rejected(field):
    with pytest.raises(ValueError, match='hash differs'):
        evaluator.require_canonical_json_equal({field: 'a'*64}, {field: 'a'*63+'b'}, 'hash differs')


@pytest.mark.parametrize('invalid', [math.nan, math.inf, -math.inf])
def test_nonfinite_values_fail_json_boundary(invalid):
    with pytest.raises(ValueError):
        evaluator.require_canonical_json_equal({'x': invalid}, {'x': invalid}, 'nonfinite')


def test_evaluate_ast_matches_frozen_logic_except_single_json_comparison():
    old = ast.parse(inspect.getsource(evaluator.frozen.evaluate))
    new = ast.parse(inspect.getsource(evaluator.evaluate))
    changes = 0
    class Replace(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal changes
            if (isinstance(node.func, ast.Name) and node.func.id == 'require'
                    and isinstance(node.args[0], ast.Compare)
                    and ast.unparse(node.args[0]) == 'a[field] == b[field]'):
                changes += 1
                return ast.Call(func=ast.Name(id='require_canonical_json_equal', ctx=ast.Load()),
                    args=[node.args[0].left, node.args[0].comparators[0], node.args[1]], keywords=[])
            return self.generic_visit(node)
    expected = Replace().visit(old)
    assert changes == 1
    assert ast.dump(expected, include_attributes=False) == ast.dump(new, include_attributes=False)
    # These remain the actual frozen helpers, not reimplementations.
    for name in ('load_checkpoint', 'validate_match', 'transport', 'transport_audit', 'decode', 'describe'):
        assert getattr(evaluator, name) is getattr(evaluator.frozen, name)


def test_registration_snapshots_new_sources_hashes_and_refuses_overwrite(tmp_path, monkeypatch):
    source = tmp_path/'new.py'; source.write_text('registered synthetic source\n')
    document = tmp_path/'note.md'; document.write_text('serialization only\n')
    monkeypatch.setattr(evaluator, 'REPO', tmp_path)
    monkeypatch.setattr(evaluator, 'SOURCES', ('new.py', 'note.md'))
    monkeypatch.setattr(evaluator.base, 'local', lambda path: Path(path))
    class Tracked:
        def __init__(self): self.hashes = {}
        def track(self, path):
            result = tmp_path/path; self.hashes[path] = evaluator.base.sha(result); return result
    output = tmp_path/'evaluation200_v2.json'
    args = SimpleNamespace(mode='evaluate', out=str(output), device='cpu')
    _, info = evaluator.register(args, Tracked(), n_windows=200, n_forecast_samples=12)
    saved = json.loads(output.with_suffix('.protocol.json').read_text())
    assert saved['protocol']['schema_version'] == evaluator.VERSION
    assert saved['protocol']['serialization_correction']['numerical_tolerance'] is None
    assert saved['protocol']['serialization_correction']['scientific_protocol_version'] == evaluator.frozen.VERSION
    for name, digest in info['source_hashes'].items():
        assert evaluator.base.sha(output.with_suffix('.source_snapshot')/name) == digest
    with pytest.raises(ValueError, match='refusing existing output'):
        evaluator.register(args, Tracked())


def test_sources_extend_frozen_set_and_statistics_contract_is_identical():
    assert set(evaluator.frozen.SOURCES) <= set(evaluator.SOURCES)
    assert evaluator.DOCUMENT in evaluator.SOURCES
    assert 'scripts/eval_backbone_transport_pilot.py' in evaluator.SOURCES
    assert evaluator.stats is evaluator.frozen.stats
    assert evaluator.CONTRASTS == evaluator.frozen.CONTRASTS
    assert evaluator.PINNED is evaluator.frozen.PINNED
    assert len(evaluator.stats.GROUPS)*len(evaluator.CONTRASTS)*2*len(evaluator.stats.METRICS) == 150
