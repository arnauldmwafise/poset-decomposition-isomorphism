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
from collections.abc import Callable
from typing import Any, Dict, Tuple

from poset_operad.core.backend import xp, logger
from poset_operad.decomposition.direct_sum import extract_maximal_disconnected_submatrices
from poset_operad.utils.equality import are_poset_structures_strictly_equal
from poset_operad.utils.metrics import compute_triangular_saturation_metrics


def verify_isomorphism_via_maximal_disconnection_and_saturation(
    M1: Any,
    M2: Any,
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
    M1 = xp.asarray(M1)
    M2 = xp.asarray(M2)

    metric1 = compute_triangular_saturation_metrics(M1)
    metric2 = compute_triangular_saturation_metrics(M2)
    
    # Fast global invariant triage step
    if sorted(list(metric1)) != sorted(list(metric2)):
        logger.info("Saturation isomorphism check: Fast structural mismatch caught on triangular metrics.")
        return False

    logger.debug("Triangular metrics matched. Extracting maximal disconnected submatrices for deep verification.")
    output1 = extract_maximal_disconnected_submatrices(M1)
    output2 = extract_maximal_disconnected_submatrices(M2)
    
    isomorphic = are_poset_structures_strictly_equal(output1, output2)
    logger.info(f"Maximal disconnection isomorphism check complete. Result Isomorphic: {isomorphic}")
    return isomorphic


def check_all_isomorphisms(
    testbag: list[Any],
    predicate: Callable[[Any, Any], bool],
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
    logger.info(f"Launching exhaustive pairwise evaluation sweep over a bag of {n} poset matrices ({n * n} checks).")
    
    # Pre-cast all array vectors onto the optimized target hardware device layer
    device_bag = [xp.asarray(matrix) for matrix in testbag]
    
    results = {
        (x, y): bool(predicate(device_bag[x], device_bag[y]))
        for x, y in itertools.product(range(n), repeat=2)
    }
    
    logger.info("Pairwise isomorphism matrix evaluation completed successfully.")
    return results
