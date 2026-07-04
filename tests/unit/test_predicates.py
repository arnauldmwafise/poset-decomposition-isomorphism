"""Unit tests for poset_operad.core.predicates."""

from __future__ import annotations

from typing import Any
import pytest

from poset_operad.core.backend import xp
from poset_operad.core.predicates import (
    check_poset_connectivity,
    is_chain_or_antichain,
    is_disconnected_poset,
    is_non_partial_semi_equidualizable,
    is_non_trivial_poset,
    is_partial_semi_equidualizable,
    is_trivial_poset,
)


class TestIsTrivialPoset:
    def test_trivial(self, trivial: Any) -> None:
        assert is_trivial_poset(trivial) is True

    def test_2x2_not_trivial(self) -> None:
        assert is_trivial_poset(xp.asarray([[1, 0], [1, 1]], dtype=xp.int32)) is False

    def test_zero_element(self) -> None:
        assert is_trivial_poset(xp.asarray([[0]], dtype=xp.int32)) is False

    def test_empty_not_trivial(self) -> None:
        assert is_trivial_poset(xp.empty((0, 0), dtype=xp.int32)) is False


class TestIsChainOrAntichain:
    def test_antichain_3(self, antichain_3: Any) -> None:
        assert is_chain_or_antichain(antichain_3) is True

    def test_chain_3(self, chain_3: Any) -> None:
        assert is_chain_or_antichain(chain_3) is True

    def test_non_trivial_is_neither(self, semi_right_4: Any) -> None:
        assert is_chain_or_antichain(semi_right_4) is False

    def test_trivial_is_antichain(self, trivial: Any) -> None:
        assert is_chain_or_antichain(trivial) is True


class TestIsNonTrivialPoset:
    def test_non_trivial(self, semi_right_4: Any) -> None:
        assert is_non_trivial_poset(semi_right_4) is True

    def test_chain_is_trivial(self, chain_3: Any) -> None:
        assert is_non_trivial_poset(chain_3) is False

    def test_antichain_is_trivial(self, antichain_3: Any) -> None:
        assert is_non_trivial_poset(antichain_3) is False

    def test_empty(self) -> None:
        assert is_non_trivial_poset(xp.empty((0, 0), dtype=xp.int32)) is False


class TestCheckPosetConnectivity:
    def test_connected_chain(self, chain_3: Any) -> None:
        assert check_poset_connectivity(chain_3) is True

    def test_disconnected_antichain(self, antichain_3: Any) -> None:
        # 3×3 identity: three isolated nodes
        assert check_poset_connectivity(antichain_3) is False

    def test_trivial_connected(self, trivial: Any) -> None:
        assert check_poset_connectivity(trivial) is True

    def test_empty_not_connected(self) -> None:
        assert check_poset_connectivity(xp.empty((0, 0), dtype=xp.int32)) is False

    def test_semi_right_connected(self, semi_right_4: Any) -> None:
        # The full semi-right matrix IS connected (col-0 links all nodes)
        assert check_poset_connectivity(semi_right_4) is True


class TestIsDisconnectedPoset:
    def test_antichain_disconnected(self, antichain_3: Any) -> None:
        assert is_disconnected_poset(antichain_3) is True

    def test_chain_not_disconnected(self, chain_3: Any) -> None:
        assert is_disconnected_poset(chain_3) is False


class TestIsPartialSemiEquidualizable:
    def test_semi_right_is_pse(self, semi_right_4: Any) -> None:
        assert is_partial_semi_equidualizable(semi_right_4) is True

    def test_semi_left_is_pse(self, semi_left_4: Any) -> None:
        assert is_partial_semi_equidualizable(semi_left_4) is True

    def test_chain_is_not_pse(self, chain_4: Any) -> None:
        assert is_partial_semi_equidualizable(chain_4) is False

    def test_antichain_not_pse(self, antichain_3: Any) -> None:
        assert is_partial_semi_equidualizable(antichain_3) is False

    def test_trivial_not_pse(self, trivial: Any) -> None:
        assert is_partial_semi_equidualizable(trivial) is False


class TestIsNonPartialSemiEquidualizable:
    def test_negation_of_pse(self, semi_right_4: Any, chain_4: Any) -> None:
        assert is_non_partial_semi_equidualizable(semi_right_4) is False
        assert is_non_partial_semi_equidualizable(chain_4) is True
