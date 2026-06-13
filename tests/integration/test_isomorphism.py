"""
Integration tests for isomorphism verification across all three tiers.

These tests use the canonical isobag fixtures to verify that the algorithms
return correct results on known isomorphic / non-isomorphic pairs.
"""

import numpy as np
import pytest
from poset_operad.isomorphism.hierarchical import (
    verify_poset_isomorphism_hierarchical,
    clear_cache,
)
from poset_operad.isomorphism.saturation import (
    verify_isomorphism_via_maximal_disconnection_and_saturation,
    check_all_isomorphisms,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure the memoization cache is clean before every test."""
    clear_cache()
    yield
    clear_cache()


class TestHierarchicalIsomorphism:
    def test_isobag1_pair_is_isomorphic(self, isobag1):
        a, b = isobag1
        assert verify_poset_isomorphism_hierarchical(a, b) is True

    def test_self_isomorphic(self, semi_right_4):
        assert verify_poset_isomorphism_hierarchical(semi_right_4, semi_right_4) is True

    def test_chain_vs_antichain_not_isomorphic(self, chain_3, antichain_3):
        assert verify_poset_isomorphism_hierarchical(chain_3, antichain_3) is False

    def test_different_sizes_not_isomorphic(self, chain_3, chain_4):
        assert verify_poset_isomorphism_hierarchical(chain_3, chain_4) is False

    def test_isobag2_all_mutual(self, isobag2):
        """All matrices in isobag2 should be mutually isomorphic."""
        for i in range(len(isobag2)):
            for j in range(len(isobag2)):
                result = verify_poset_isomorphism_hierarchical(isobag2[i], isobag2[j])
                assert result is True, f"Expected isomorphic: pair ({i}, {j})"


class TestSaturationIsomorphism:
    def test_self_isomorphic(self, semi_right_4):
        result = verify_isomorphism_via_maximal_disconnection_and_saturation(
            semi_right_4, semi_right_4
        )
        assert result is True

    def test_chain_vs_antichain(self, chain_3, antichain_3):
        result = verify_isomorphism_via_maximal_disconnection_and_saturation(
            chain_3, antichain_3
        )
        assert result is False


class TestCheckAllIsomorphisms:
    def test_returns_complete_grid(self, isobag1):
        results = check_all_isomorphisms(
            isobag1, verify_poset_isomorphism_hierarchical
        )
        n = len(isobag1)
        assert len(results) == n * n

    def test_diagonal_is_true(self, isobag1):
        results = check_all_isomorphisms(
            isobag1, verify_poset_isomorphism_hierarchical
        )
        for i in range(len(isobag1)):
            assert results[(i, i)] is True
