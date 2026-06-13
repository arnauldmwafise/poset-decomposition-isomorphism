"""Unit tests for poset_operad.core.predicates."""

import numpy as np
import pytest
from poset_operad.core.predicates import (
    is_trivial_poset,
    is_chain_or_antichain,
    is_non_trivial_poset,
    check_poset_connectivity,
    is_disconnected_poset,
    is_partial_semi_equidualizable,
    is_non_partial_semi_equidualizable,
)


class TestIsTrivialPoset:
    def test_trivial(self, trivial):
        assert is_trivial_poset(trivial) is True

    def test_2x2_not_trivial(self):
        assert is_trivial_poset(np.array([[1, 0], [1, 1]])) is False

    def test_zero_element(self):
        assert is_trivial_poset(np.array([[0]])) is False

    def test_empty_not_trivial(self):
        assert is_trivial_poset(np.empty((0, 0), dtype=int)) is False


class TestIsChainOrAntichain:
    def test_antichain_3(self, antichain_3):
        assert is_chain_or_antichain(antichain_3) is True

    def test_chain_3(self, chain_3):
        assert is_chain_or_antichain(chain_3) is True

    def test_non_trivial_is_neither(self, semi_right_4):
        assert is_chain_or_antichain(semi_right_4) is False

    def test_trivial_is_antichain(self, trivial):
        assert is_chain_or_antichain(trivial) is True


class TestIsNonTrivialPoset:
    def test_non_trivial(self, semi_right_4):
        assert is_non_trivial_poset(semi_right_4) is True

    def test_chain_is_trivial(self, chain_3):
        assert is_non_trivial_poset(chain_3) is False

    def test_antichain_is_trivial(self, antichain_3):
        assert is_non_trivial_poset(antichain_3) is False

    def test_empty(self):
        assert is_non_trivial_poset(np.empty((0, 0), dtype=int)) is False


class TestCheckPosetConnectivity:
    def test_connected_chain(self, chain_3):
        assert check_poset_connectivity(chain_3) is True

    def test_disconnected_antichain(self, antichain_3):
        # 3×3 identity: three isolated nodes
        assert check_poset_connectivity(antichain_3) is False

    def test_trivial_connected(self, trivial):
        assert check_poset_connectivity(trivial) is True

    def test_empty_not_connected(self):
        assert check_poset_connectivity(np.empty((0, 0), dtype=int)) is False

    def test_semi_right_connected(self, semi_right_4):
        # The full semi-right matrix IS connected (col-0 links all nodes)
        assert check_poset_connectivity(semi_right_4) is True


class TestIsDisconnectedPoset:
    def test_antichain_disconnected(self, antichain_3):
        assert is_disconnected_poset(antichain_3) is True

    def test_chain_not_disconnected(self, chain_3):
        assert is_disconnected_poset(chain_3) is False


class TestIsPartialSemiEquidualizable:
    def test_semi_right_is_pse(self, semi_right_4):
        assert is_partial_semi_equidualizable(semi_right_4) is True

    def test_semi_left_is_pse(self, semi_left_4):
        assert is_partial_semi_equidualizable(semi_left_4) is True

    def test_chain_is_not_pse(self, chain_4):
        assert is_partial_semi_equidualizable(chain_4) is False

    def test_antichain_not_pse(self, antichain_3):
        assert is_partial_semi_equidualizable(antichain_3) is False

    def test_trivial_not_pse(self, trivial):
        assert is_partial_semi_equidualizable(trivial) is False


class TestIsNonPartialSemiEquidualizable:
    def test_negation_of_pse(self, semi_right_4, chain_4):
        assert is_non_partial_semi_equidualizable(semi_right_4) is False
        assert is_non_partial_semi_equidualizable(chain_4) is True
