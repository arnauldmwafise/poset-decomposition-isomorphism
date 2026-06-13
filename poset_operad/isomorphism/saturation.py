"""
poset_operad.isomorphism.saturation
======================================

Saturation-metric and maximal-disconnection isomorphism verification.

Targets posets with irregular or non-obvious direct-sum structures by using
triangular saturation counts as a fast global invariant and maximal
disconnected sub-matrices as the structural sieve.
"""

from __future__ import annotations

import itertools
import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable

from poset_operad.decomposition.direct_sum import extract_maximal_disconnected_submatrices
from poset_operad.utils.equality import are_poset_structures_strictly_equal
from poset_operad.utils.metrics import compute_triangular_saturation_metrics


def verify_isomorphism_via_maximal_disconnection_and_saturation(
    M1: NDArray[np.int_],
    M2: NDArray[np.int_],
) -> bool:
    """Verify isomorphism via saturation metrics and maximal disconnected sub-matrices.

    Steps
    -----
    1. Compute and compare row/column triangular saturation counts.
    2. Partition both matrices into maximal disconnected principal submatrices.
    3. Compare the resulting collections with
       :func:`are_poset_structures_strictly_equal`.

    Parameters
    ----------
    M1, M2:
        n×n binary poset adjacency matrices.

    Returns
    -------
    bool

    Complexity
    ----------
    Time O(N⁴), Space O(N²).
    """
    metric1 = compute_triangular_saturation_metrics(M1)
    metric2 = compute_triangular_saturation_metrics(M2)
    if sorted(metric1) != sorted(metric2):
        return False

    output1 = extract_maximal_disconnected_submatrices(M1)
    output2 = extract_maximal_disconnected_submatrices(M2)
    return are_poset_structures_strictly_equal(output1, output2)


def check_all_isomorphisms(
    testbag: list[NDArray[np.int_]],
    predicate: Callable[[NDArray[np.int_], NDArray[np.int_]], bool],
) -> dict[tuple[int, int], bool]:
    """Exhaustive pairwise isomorphism test across every ordered pair in *testbag*.

    Parameters
    ----------
    testbag:
        Collection of 2-D ndarray poset matrices.
    predicate:
        Binary isomorphism function returning ``bool``.

    Returns
    -------
    dict[tuple[int, int], bool]
        Mapping ``(i, j) → predicate(testbag[i], testbag[j])``.

    Complexity
    ----------
    Time O(n² · T), Space O(n²).
    """
    n = len(testbag)
    return {
        (x, y): predicate(testbag[x], testbag[y])
        for x, y in itertools.product(range(n), repeat=2)
    }
