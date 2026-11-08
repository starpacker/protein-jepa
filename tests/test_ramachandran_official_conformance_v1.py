"""Independent compiled upstream reference, not a second interpolation formula.

Only scitbx storage and assertion dependencies are shimmed. The C++ scorer,
compiled numeric tables, interpolation arithmetic and classification thresholds
are the original pinned, unmodified files. No molecular data or models are read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from evaluation import ramachandran_reference_v1 as implementation


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'evaluation/reference_data/ramachandran_top8000_v1/cctbx'
PINNED = {
    'mmtbx/validation/ramachandran/rama_eval.h':
        '9e0776759f8c37b39f919056f43e140000ef676cfd86170a5b88afc2ddcb98b1',
    'mmtbx/validation/ramachandran/rama8000_tables.h':
        '314621665c0be2801f12a5a34288e6f1cb52a79ecee23e9d93b9508c42ec1acd',
    'scitbx/math/linear_interpolation.h':
        '947c9b1437475833e6c908aa2855d5c0089ad35eafa1306908f7972be1118122',
}

# This has no numerical scoring operations: it only stores a pointer and the
# row stride used by the upstream C++ table accessor.
STORAGE_SHIM = r'''
#pragma once
#include <cstddef>
#include <string>
#include <stdexcept>
namespace scitbx { namespace af {
  template<int N> struct c_grid {};
  template<class T, class Grid> class const_ref {
    const T* pointer_ = nullptr;
    std::size_t rows_ = 0, columns_ = 0;
  public:
    const_ref() = default;
    const_ref(const T* p, std::size_t rows, std::size_t columns)
        : pointer_(p), rows_(rows), columns_(columns) {}
    T operator()(int i, int j) const {
      if (i < 0 || j < 0 || std::size_t(i) >= rows_ || std::size_t(j) >= columns_)
        throw std::out_of_range("upstream table index");
      return pointer_[std::size_t(i) * columns_ + std::size_t(j)];
    }
  };
}}
'''
ASSERT_SHIM = r'''
#pragma once
#include <stdexcept>
#define SCITBX_ASSERT(condition) do { if (!(condition)) \
  throw std::runtime_error("upstream SCITBX_ASSERT: " #condition); } while (0)
'''
DRIVER = r'''
#include <iomanip>
#include <iostream>
#include <mmtbx/validation/ramachandran/rama_eval.h>
int main() {
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout << std::setprecision(17);
  mmtbx::validation::ramachandran::rama_eval oracle;
  char mode;
  int category;
  double x, y;
  try {
    while (std::cin >> mode >> category >> x) {
      if (category < 0 || category > 5) return 3;
      if (mode == 'A') {
        if (!(std::cin >> y)) return 4;
        const double score = oracle.get_score(category, x, y);
        std::cout << score << ' ' << oracle.evaluate_score(category, score) << '\n';
      } else if (mode == 'S') {
        std::cout << x << ' ' << oracle.evaluate_score(category, x) << '\n';
      } else return 5;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 6;
  }
  return std::cin.eof() ? 0 : 7;
}
'''


@pytest.fixture(scope='module')
def compiled_official(tmp_path_factory):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('official C++ conformance requires g++; no Python fallback')
    for name, expected in PINNED.items():
        assert hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() == expected
    folder = tmp_path_factory.mktemp('official_rama_cpp')
    shim = folder / 'shim'
    for relative, content in {
        'scitbx/array_family/shared.h': STORAGE_SHIM,
        'scitbx/array_family/versa.h': '#pragma once\n#include <scitbx/array_family/shared.h>\n',
        'scitbx/array_family/accessors/c_grid.h': '#pragma once\n#include <scitbx/array_family/shared.h>\n',
        'scitbx/error.h': ASSERT_SHIM,
    }.items():
        path = shim / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    driver = folder / 'reference.cpp'
    driver.write_text(DRIVER)
    binary = folder / 'reference'
    command = [compiler, '-std=c++11', '-O2', '-ffp-contract=off', '-fno-fast-math',
               '-I', str(shim), '-I', str(REFERENCE), str(driver), '-o', str(binary)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    return binary


def upstream_angles(binary, category, phi, psi):
    phi, psi = np.broadcast_arrays(np.asarray(phi, dtype=np.float64), np.asarray(psi, dtype=np.float64))
    query = ''.join(f'A {category} {x:.17g} {y:.17g}\n' for x, y in zip(phi.flat, psi.flat))
    run = subprocess.run([str(binary)], input=query, check=True, capture_output=True, text=True, timeout=60)
    result = np.fromstring(run.stdout, sep=' ').reshape(-1, 2)
    assert result.shape == (phi.size, 2)
    return result[:, 0].reshape(phi.shape), result[:, 1].astype(np.int64).reshape(phi.shape)


def check_angles(binary, category, phi, psi, *, exact=False):
    expected_score, expected_category = upstream_angles(binary, category, phi, psi)
    actual = implementation.score_angles(torch.tensor(phi, dtype=torch.float64),
                                         torch.tensor(psi, dtype=torch.float64), category)
    assert actual['score'].dtype == torch.float64 and actual['score'].device.type == 'cpu'
    assert actual['category'].dtype == torch.int64
    score = actual['score'].detach().numpy()
    if exact:
        np.testing.assert_array_equal(score, expected_score)
    else:
        np.testing.assert_allclose(score, expected_score, atol=2e-14, rtol=2e-13)
    np.testing.assert_array_equal(actual['category'].numpy(), expected_category)


@pytest.mark.parametrize('category', range(6))
def test_all_official_table_centers_exact(compiled_official, category):
    phi, psi = np.meshgrid(np.arange(-179., 180., 2.), np.arange(-179., 180., 2.), indexing='ij')
    check_angles(compiled_official, category, phi, psi, exact=True)


@pytest.mark.parametrize('category', range(6))
def test_complete_integer_degree_grid_matches_compiled_interpolation(compiled_official, category):
    phi, psi = np.meshgrid(np.arange(-180., 181.), np.arange(-180., 181.), indexing='ij')
    check_angles(compiled_official, category, phi, psi)


@pytest.mark.parametrize('category', range(6))
def test_periodic_seams_bin_edges_and_neighboring_floats(compiled_official, category):
    knots = np.array([-540., -360., -181., -180., -179., -1., 0., 1., 179., 180., 181., 360., 540.])
    edges = np.concatenate([knots, np.nextafter(knots, -np.inf), np.nextafter(knots, np.inf)])
    phi, psi = np.meshgrid(edges, edges, indexing='ij')
    check_angles(compiled_official, category, phi, psi)


@pytest.mark.parametrize('category', range(6))
def test_fixed_random_off_grid_angles(compiled_official, category):
    points = np.random.default_rng(20261101 + category).uniform(-1080., 1080., size=(20000, 2))
    check_angles(compiled_official, category, points[:, 0], points[:, 1])


def test_threshold_equalities_and_adjacent_float_labels_against_official_cpp(compiled_official):
    levels = np.array([0., .0005, .001, .002, .02, 1.], dtype=np.float64)
    scores = np.unique(np.concatenate([levels, np.nextafter(levels, -np.inf), np.nextafter(levels, np.inf)]))
    scores = scores[(scores >= 0) & (scores <= 1)]
    types = np.repeat(np.arange(6), len(scores))
    repeated = np.tile(scores, 6)
    query = ''.join(f'S {category} {score:.17g}\n' for category, score in zip(types, repeated))
    run = subprocess.run([str(compiled_official)], input=query, check=True, capture_output=True, text=True, timeout=60)
    expected = np.fromstring(run.stdout, sep=' ').reshape(-1, 2)[:, 1].astype(np.int64)
    actual = implementation._classify_scores(torch.tensor(repeated), torch.tensor(types))
    np.testing.assert_array_equal(actual.numpy(), expected)
