"""
Integration tests for isomorphism verification across all three tiers.

These tests use the canonical isobag fixtures to verify that the algorithms
return correct results on known isomorphic / non-isomorphic pairs.
"""

from __future__ import annotations

from typing import Any, Generator
import pytest

from poset_operad.isomorphism.hierarchical import (
    clear_cache,
    verify_poset_isomorphism_hierarchical,
)
from poset_operad.isomorphism.saturation import (
    check_all_isomorphisms,
    verify_isomorphism_via_maximal_disconnection_and_saturation,
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
