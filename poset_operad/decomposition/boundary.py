"""
poset_operad.decomposition.boundary
=====================================

Boundary-depth scanning and disconnected-core extraction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from poset_operad.core.backend import xp, logger
from poset_operad.core.predicates import check_poset_connectivity
from poset_operad.core.submatrix import get_maximal_elements, get_minimal_elements


# ── Public API ────────────────────────────────────────────────────────────────

def extract_semiequidual_subcomponents(
    matrix: Any,
) -> tuple[list[Any], str, tuple[int, int]]:
    """Strip saturated boundary layers and return the inner sub-matrix."""
    matrix = xp.asarray(matrix)
    n = matrix.shape[0]
    if n == 0:
        return [matrix], "connected", (0, 0)

    i = j = 0
    while i < n and bool(xp.all(matrix[i:, i] == 1)):
        i += 1
    while j < n and bool(xp.all(matrix[n - j - 1, : n - j] == 1)):
        j += 1

    d1, d2 = i, j

    if d1 == 0 and d2 == 0:
        return [matrix], "connected", (0, 0)

    logger.info(f"Isolated semi-equidual components. Shape: {matrix.shape} | Identified depths: d1={d1}, d2={d2}")

    if d1 > 0 and d2 > 0 and d1 == d2:
        sub = matrix[d1 : n - d2, d1 : n - d2]
    elif d1 > 0:
        sub = matrix[d1:, d1:]
    else:
        sub = matrix[: n - d2, : n - d2]

    return [sub], "connected", (d1, d2)


def extract_disconnected_core_with_depths(
    poset_matrix: Any,
) -> tuple[Any, dict[str, tuple[int, ...]]] | None:
    """Extract the internal disconnected sub-matrix and report boundary depths."""
    poset_matrix = xp.asarray(poset_matrix)
    n = poset_matrix.shape[0]
    if n == 0:
        return None

    depth1 = depth2 = 0
    while depth1 < n and bool(xp.all(poset_matrix[depth1:, depth1:] == 1 if depth1 == n-1 else poset_matrix[depth1:, depth1])):
        depth1 += 1
    
    # Precise iterative scan for row depths
    for idx in range(n):
        if bool(xp.all(poset_matrix[n - idx - 1, : n - idx])):
            depth2 += 1
        else:
            break

    # Case 1: Double Dualizable
    if depth1 > 0 and depth1 == depth2:
        if len(get_maximal_elements(poset_matrix)) == len(
            get_minimal_elements(poset_matrix)
        ):
            core = poset_matrix[depth1 : n - depth2, depth1 : n - depth2]
            return core, {"depth1,depth2": (depth1, depth2)}

    # Case 2: Semi-Right Dualizable
    if depth1 > 0:
        sub = poset_matrix[depth1:, depth1:]
        if not check_poset_connectivity(sub):
            return sub, {"depth1": (depth1,)}

    # Case 3: Semi-Left Dualizable
    if depth2 > 0:
        sub = poset_matrix[: n - depth2, : n - depth2]
        if not check_poset_connectivity(sub):
            return sub, {"depth2": (depth2,)}

    return None
