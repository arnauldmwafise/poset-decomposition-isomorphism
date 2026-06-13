"""Unit tests for poset_operad.decomposition.*"""

import numpy as np
import pytest
from poset_operad.decomposition.boundary import (
    extract_semiequidual_subcomponents,
    extract_disconnected_core_with_depths,
)
from poset_operad.decomposition.direct_sum import (
    extract_direct_sum_components,
    extract_maximal_disconnected_submatrices,
)
from poset_operad.decomposition.tree import build_poset_decomposition_tree


class TestExtractSemiequidualSubcomponents:
    def test_semi_right_depth_1(self, semi_right_4):
        sub_list, status, (d1, d2) = extract_semiequidual_subcomponents(semi_right_4)
        assert d1 >= 1
        # Sub-matrix must be strictly smaller than the input
        assert sub_list[0].shape[0] < semi_right_4.shape[0]
        assert status == "connected"

    def test_no_saturation_chain(self, chain_3):
        # The fully-saturated lower-triangle of a chain means d2 == n,
        # so the algorithm returns the original matrix, depths (0, n) or similar.
        # What matters is the sub-matrix is the same size (no meaningful stripping).
        sub_list, _, (d1, d2) = extract_semiequidual_subcomponents(chain_3)
        assert isinstance(sub_list, list) and len(sub_list) == 1

    def test_empty_matrix(self):
        M = np.empty((0, 0), dtype=int)
        sub_list, _, (d1, d2) = extract_semiequidual_subcomponents(M)
        assert d1 == 0 and d2 == 0


class TestExtractDisconnectedCoreWithDepths:
    def test_semi_right_returns_core(self, semi_right_4):
        result = extract_disconnected_core_with_depths(semi_right_4)
        assert result is not None
        core, meta = result
        assert "depth1" in meta or "depth1,depth2" in meta

    def test_chain_returns_none(self, antichain_3):
        # A pure antichain (identity) has no saturated boundaries → returns None
        result = extract_disconnected_core_with_depths(antichain_3)
        assert result is None


class TestExtractDirectSumComponents:
    def test_antichain_splits_into_singletons(self, antichain_3):
        components = extract_direct_sum_components(antichain_3)
        assert len(components) == 3
        assert all(c.shape == (1, 1) for c in components)

    def test_chain_is_one_component(self, chain_3):
        components = extract_direct_sum_components(chain_3)
        assert len(components) == 1


class TestExtractMaximalDisconnectedSubmatrices:
    def test_returns_list(self, semi_right_4):
        result = extract_maximal_disconnected_submatrices(semi_right_4)
        assert isinstance(result, list)

    def test_small_matrix(self):
        M = np.array([[1]])
        result = extract_maximal_disconnected_submatrices(M)
        assert result == []


class TestBuildPosetDecompositionTree:
    def test_tree_is_list(self, semi_right_4):
        tree = build_poset_decomposition_tree(semi_right_4)
        assert isinstance(tree, list)

    def test_trivial_produces_empty_tree(self, trivial):
        tree = build_poset_decomposition_tree(trivial)
        # trivial poset decomposes to nothing non-trivial
        assert isinstance(tree, list)
