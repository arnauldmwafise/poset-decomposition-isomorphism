"""Structural decomposition algorithms for poset matrices."""

from __future__ import annotations

from poset_operad.decomposition.boundary import (
    extract_disconnected_core_with_depths,
    extract_semiequidual_subcomponents,
)
from poset_operad.decomposition.direct_sum import (
    extract_direct_sum_components,
    extract_maximal_disconnected_submatrices,
    extract_poset_direct_sum_components,
)
from poset_operad.decomposition.tree import (
    build_poset_decomposition_tree,
    decompose_dual_core_into_components,
    update_nested_posets,
)

__all__ = [
    "extract_semiequidual_subcomponents",
    "extract_disconnected_core_with_depths",
    "extract_direct_sum_components",
    "extract_poset_direct_sum_components",
    "extract_maximal_disconnected_submatrices",
    "build_poset_decomposition_tree",
    "decompose_dual_core_into_components",
    "update_nested_posets",
]
