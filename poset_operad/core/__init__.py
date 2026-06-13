"""Core atomic predicates and matrix primitive helpers."""

from poset_operad.core.predicates import (
    is_trivial_poset,
    is_non_trivial_poset,
    is_chain_or_antichain,
    is_partial_semi_equidualizable,
    is_non_partial_semi_equidualizable,
    is_disconnected_poset,
    check_poset_connectivity,
)
from poset_operad.core.submatrix import (
    get_principal_submatrix,
    get_maximal_elements,
    get_minimal_elements,
)

__all__ = [
    "is_trivial_poset",
    "is_non_trivial_poset",
    "is_chain_or_antichain",
    "is_partial_semi_equidualizable",
    "is_non_partial_semi_equidualizable",
    "is_disconnected_poset",
    "check_poset_connectivity",
    "get_principal_submatrix",
    "get_maximal_elements",
    "get_minimal_elements",
]
