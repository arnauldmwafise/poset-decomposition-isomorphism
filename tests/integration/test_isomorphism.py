"""
Integration tests for isomorphism verification across all three tiers.

These tests use the canonical isobag fixtures to verify that the algorithms
return correct results on known isomorphic / non-isomorphic pairs.
"""

from __future__ import annotations

from typing import Any, Generator
import numpy as np
import pytest

from poset_operad.isomorphism.hierarchical import (
    clear_cache,
    verify_poset_isomorphism_hierarchical,
)
from poset_operad.isomorphism.saturation import (
    check_all_isomorphisms,
    verify_isomorphism_via_maximal_disconnection_and_saturation,
)
from poset_operad.isomorphism.direct_sum import (
    verify_isomorphism_via_direct_sum_decomposition,
)


@pytest.fixture(autouse=True)
def reset_cache() -> Generator[None, None, None]:
    """Ensure the memoization cache is clean before every test."""
    clear_cache()
    yield
    clear_cache()


class TestHierarchicalIsomorphism:
    def test_isobag1_pair_is_isomorphic(self, isobag1: Any) -> None:
        a, b = isobag1
        assert verify_poset_isomorphism_hierarchical(a, b) is True

    def test_self_isomorphic(self, semi_right_4: Any) -> None:
        assert verify_poset_isomorphism_hierarchical(semi_right_4, semi_right_4) is True

    def test_chain_vs_antichain_not_isomorphic(self, chain_3: Any, antichain_3: Any) -> None:
        assert verify_poset_isomorphism_hierarchical(chain_3, antichain_3) is False

    def test_different_sizes_not_isomorphic(self, chain_3: Any, chain_4: Any) -> None:
        assert verify_poset_isomorphism_hierarchical(chain_3, chain_4) is False

    def test_isobag2_all_mutual(self, isobag2: Any) -> None:
        """All matrices in isobag2 should be mutually isomorphic."""
        for i in range(len(isobag2)):
            for j in range(len(isobag2)):
                result = verify_poset_isomorphism_hierarchical(isobag2[i], isobag2[j])
                assert result is True, f"Expected isomorphic: pair ({i}, {j})"


class TestSaturationIsomorphism:
    def test_self_isomorphic(self, semi_right_4: Any) -> None:
        result = verify_isomorphism_via_maximal_disconnection_and_saturation(
            semi_right_4, semi_right_4
        )
        assert result is True

    def test_chain_vs_antichain(self, chain_3: Any, antichain_3: Any) -> None:
        result = verify_isomorphism_via_maximal_disconnection_and_saturation(
            chain_3, antichain_3
        )
        assert result is False


class TestCheckAllIsomorphisms:
    def test_returns_complete_grid(self, isobag1: Any) -> None:
        results = check_all_isomorphisms(
            isobag1, verify_poset_isomorphism_hierarchical
        )
        n = len(isobag1)
        assert len(results) == n * n

    def test_diagonal_is_true(self, isobag1: Any) -> None:
        results = check_all_isomorphisms(
            isobag1, verify_poset_isomorphism_hierarchical
        )
        for i in range(len(isobag1)):
            assert results[(i, i)] is True


class TestNonSemiEquidualSuiteResolution:
    """Validates the Tier 2 Direct Sum Resolution over the T_NSE Test Suite.
    
    Uses check_all_isomorphisms to verify that the 6x6 non-semi-equidual collection
    evaluates correctly under direct-sum partitioning, but fails under standard
    hierarchical and maximal saturation checks across all combinations.
    """
    @pytest.fixture
    def t_nse_collection(self) -> list[np.ndarray]:
        def _parse(s: str) -> np.ndarray:
            return np.fromstring(s, dtype=np.int32, sep=' ').reshape(6, 6)

        m0 = _parse("1 0 0 0 0 0  0 1 0 0 0 0  1 1 1 0 0 0  1 1 1 1 0 0  1 1 1 0 1 0  1 1 0 0 0 1")
        m1 = _parse("1 0 0 0 0 0  0 1 0 0 0 0  0 1 1 0 0 0  0 1 0 1 0 0  1 1 1 1 1 0  1 1 1 1 0 1")
        m2 = _parse("1 0 0 0 0 0  1 1 0 0 0 0  1 0 1 0 0 0  0 0 0 1 0 0  1 1 1 1 1 0  1 1 1 1 0 1")
        m3 = _parse("1 0 0 0 0 0  0 1 0 0 0 0  1 1 1 0 0 0  0 0 0 1 0 0  1 1 1 1 1 0  1 1 1 1 0 1")
        m4 = _parse("1 0 0 0 0 0  0 1 0 0 0 0  1 1 1 0 0 0  1 1 0 1 0 0  1 1 0 1 1 0  1 1 0 1 0 1")
        m5 = _parse("1 0 0 0 0 0  0 1 0 0 0 0  0 0 1 0 0 0  0 1 1 1 0 0  1 1 1 1 1 0  1 1 1 1 0 1")
        m6 = _parse("1 0 0 0 0 0  0 1 0 0 0 0  1 1 1 0 0 0  1 1 0 1 0 0  1 1 0 0 1 0  1 1 0 1 1 1")
        m7 = _parse("1 0 0 0 0 0  0 1 0 0 0 0  1 1 1 0 0 0  1 1 0 1 0 0  1 1 1 1 1 0  1 1 0 0 0 1")

        return [m0, m1, m2, m3, m4, m5, m6, m7]

    def test_t_nse_grid_resolution_across_all_prescribed_engines(
        self, t_nse_collection: list[np.ndarray]
    ) -> None:
        """Verifies full cross-check grids via check_all_isomorphisms."""
        n = len(t_nse_collection)
        
        # ── 1. ENGINE RUN: Direct Sum Decomposition (Tier 2 Resolution) ──
        # Since all matrices represent independent self-isomorphic pairs or unique variants,
        # running them through check_all_isomorphisms yields a complete complete results map.
        ds_results = check_all_isomorphisms(t_nse_collection, verify_isomorphism_via_direct_sum_decomposition)
        assert len(ds_results) == n * n
        
        # Assert that diagonal self-isomorphisms evaluate as TRUE under Direct Sum
        for i in range(n):
            assert ds_results[(i, i)] is True, f"M_{i} failed self-isomorphism on direct_sum grid"

        # ── 2. ENGINE RUN: Hierarchical Tree Decomposition (Strict Boundary Isolation) ──
        # Because these matrices are structurally non-semi-equidual (no valid boundaries can be stripped),
        # your research specifies that the hierarchical tree engine must return FALSE across the entire grid.
        hier_results = check_all_isomorphisms(t_nse_collection, verify_poset_isomorphism_hierarchical)
        assert len(hier_results) == n * n
        for (i, j), res in hier_results.items():
            assert res is False, f"Hierarchical engine incorrectly returned True for pair ({i}, {j})"

        # ── 3. ENGINE RUN: Maximal Saturation Disconnection (Tier 3 Core Check) ──
        # Similarly, these entries do not support standard boundary saturation layers.
        # The saturation engine must fail across all 64 grid pairs.
        sat_results = check_all_isomorphisms(t_nse_collection, verify_isomorphism_via_maximal_disconnection_and_saturation)
        assert len(sat_results) == n * n
        for (i, j), res in sat_results.items():
            assert res is False, f"Saturation engine incorrectly returned True for pair ({i}, {j})"
