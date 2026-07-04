"""Core atomic predicates and matrix primitive helpers."""

from __future__ import annotations

from poset_operad.core.backend import get_connected_components_count
from poset_operad.core.predicates import (
    check_poset_connectivity,
    is_chain_or_antichain,
    is_disconnected_poset,
    is_non_partial_semi_equidualizable,
    is_non_trivial_poset,
    is_partial_semi_equidualizable,
    is_trivial_poset,
)
from poset_operad.core.submatrix import (
    get_maximal_elements,
    get_minimal_elements,
    get_principal_submatrix,
)

__all__ = [
    # backend primitives
    "get_connected_components_count",
    # predicates
    "is_trivial_poset",
    "is_non_trivial_poset",
    "is_chain_or_antichain",
    "is_partial_semi_equidualizable",
    "is_non_partial_semi_equidualizable",
    "is_disconnected_poset",
    "check_poset_connectivity",
    # submatrix operations
    "get_principal_submatrix",
    "get_maximal_elements",
    "get_minimal_elements",
]
