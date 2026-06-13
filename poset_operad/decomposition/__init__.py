"""Structural decomposition algorithms for poset matrices."""

from poset_operad.decomposition.boundary import (
    extract_semiequidual_subcomponents,
    extract_disconnected_core_with_depths,
)
from poset_operad.decomposition.direct_sum import (
    extract_direct_sum_components,
    extract_poset_direct_sum_components,
    extract_maximal_disconnected_submatrices,
)
from poset_operad.decomposition.tree import (
    build_poset_decomposition_tree,
    decompose_dual_core_into_components,
)

__all__ = [
    "extract_semiequidual_subcomponents",
    "extract_disconnected_core_with_depths",
    "extract_direct_sum_components",
    "extract_poset_direct_sum_components",
    "extract_maximal_disconnected_submatrices",
    "build_poset_decomposition_tree",
    "decompose_dual_core_into_components",
]
