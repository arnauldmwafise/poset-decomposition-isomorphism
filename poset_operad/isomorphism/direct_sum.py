"""
poset_operad.isomorphism.direct_sum
======================================

Isomorphism verification via direct-sum decomposition.

Reduces the global comparison to a set of independent local comparisons on
connected parallel sub-components, exploiting the block-diagonal structure of
direct sums.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from poset_operad.core.predicates import is_non_partial_semi_equidualizable
from poset_operad.decomposition.direct_sum import extract_poset_direct_sum_components
from poset_operad.decomposition.tree import (
    build_poset_decomposition_tree,
    update_nested_posets,
)
from poset_operad.utils.equality import are_poset_structures_strictly_equal
from poset_operad.utils.collections import get_satisfying_posets


def verify_isomorphism_via_direct_sum_decomposition(
    M1: NDArray[np.int_],
    M2: NDArray[np.int_],
) -> bool:
    """Verify isomorphism by comparing direct-sum component hierarchies.

    Steps
    -----
    1. Extract direct-sum components for both matrices.
    2. Build recursive decomposition trees for every component.
    3. Compare tree hierarchies with :func:`are_poset_structures_strictly_equal`.
    4. As a final pass, compare lower-triangular entry counts (relation density)
       of non-reducible leaf nodes.

    Parameters
    ----------
    M1, M2:
        n×n binary poset adjacency matrices.

    Returns
    -------
    bool

    Complexity
    ----------
    Time O(K · N³), Space O(K · N²).
    """
    ds1 = extract_poset_direct_sum_components(M1)
    ds2 = extract_poset_direct_sum_components(M2)

    if (ds1 is None) != (ds2 is None):
        return False
    if ds1 is None:
        return False

    tree1 = update_nested_posets(ds1, build_poset_decomposition_tree)
    tree2 = update_nested_posets(ds2, build_poset_decomposition_tree)

    if are_poset_structures_strictly_equal(tree1, tree2):
        list1 = get_satisfying_posets(tree1, is_non_partial_semi_equidualizable)
        list2 = get_satisfying_posets(tree2, is_non_partial_semi_equidualizable)
        counts1 = sorted(int(np.sum(np.tril(m))) for m in list1)
        counts2 = sorted(int(np.sum(np.tril(m))) for m in list2)
        return counts1 == counts2

    return False
