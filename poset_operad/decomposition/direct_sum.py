"""
poset_operad.decomposition.direct_sum
========================================

Direct-sum decomposition routines.

A **direct sum** of posets corresponds to a block-diagonal structure in the
adjacency matrix: no element of one block relates to any element of another.
These functions identify such blocks, including maximal disconnected intervals.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from numpy.typing import NDArray

from poset_operad.core.predicates import is_disconnected_poset
from poset_operad.core.submatrix import get_principal_submatrix


# ── Public API ────────────────────────────────────────────────────────────────

def extract_direct_sum_components(
    poset_matrix: NDArray[np.int_],
) -> list[NDArray[np.int_]]:
    """Return a list of principal submatrices representing each direct-sum component.

    Uses ``scipy.sparse.csgraph.connected_components`` with
    ``connection='weak'`` so directed edges are treated as undirected paths.

    Parameters
    ----------
    poset_matrix:
        n×n binary (0,1)-matrix.

    Returns
    -------
    list[np.ndarray]
        Connected principal submatrices, one per component.

    Complexity
    ----------
    Time O(n + E), Space O(n²).
    """
    n_components, labels = connected_components(
        poset_matrix, directed=True, connection="weak"
    )
    components: list[NDArray[np.int_]] = []
    for i in range(n_components):
        indices = np.where(labels == i)[0]
        components.append(poset_matrix[np.ix_(indices, indices)])
    return components


def extract_poset_direct_sum_components(
    matrix: NDArray[np.int_],
) -> list[list[NDArray[np.int_]]] | None:
    """Identify transition points in saturated boundaries and extract direct-sum groups.

    Performs a vectorized **forward** (column) scan and **backward** (row) scan
    to locate the first index where boundary saturation breaks.  When a
    disconnected inner region is found, :func:`extract_direct_sum_components`
    decomposes it further.

    Parameters
    ----------
    matrix:
        N×N binary poset adjacency matrix.

    Returns
    -------
    list[list[np.ndarray]] or ``None``
        Groups of connected submatrices if disconnection is found; else ``None``.

    Complexity
    ----------
    Time O(N³), Space O(N²).
    """
    dim = matrix.shape[0]
    if dim == 0:
        return None

    extracted: list[list[NDArray[np.int_]]] = []

    # Forward: column saturation scan
    col_bounds = np.array([bool(np.all(matrix[i:, i])) for i in range(dim)])
    transitions_f = np.where(col_bounds[:-1] & ~col_bounds[1:])[0]
    for t_idx in transitions_f:
        depth = int(t_idx) + 1
        inner = matrix[depth:, depth:]
        if inner.size > 0 and is_disconnected_poset(inner):
            extracted.append(extract_direct_sum_components(inner))
            break

    # Backward: row saturation scan
    row_bounds = np.array(
        [bool(np.all(matrix[dim - j - 1, : dim - j])) for j in range(dim)]
    )
    transitions_b = np.where(row_bounds[:-1] & ~row_bounds[1:])[0]
    for t_idx in transitions_b:
        depth = int(t_idx) + 1
        inner = matrix[:-depth, :-depth]
        if inner.size > 0 and is_disconnected_poset(inner):
            extracted.append(extract_direct_sum_components(inner))
            break

    return extracted if extracted else None


def extract_maximal_disconnected_submatrices(
    matrix: NDArray[np.int_],
) -> list[NDArray[np.int_]]:
    """Return all maximal principal submatrices that are graph-theoretically disconnected.

    A submatrix is *maximal* disconnected if it is disconnected and not
    contained within any larger disconnected principal submatrix.

    Algorithm
    ---------
    1. Enumerate all contiguous index intervals ``[i, j]`` and test each for
       disconnection (undirected sense).
    2. Remove any interval whose index-set is a strict subset of a larger
       disconnected interval.

    Parameters
    ----------
    matrix:
        N×N binary poset adjacency matrix.

    Returns
    -------
    list[np.ndarray]
        Maximal disconnected principal submatrices, sorted by start index.

    Complexity
    ----------
    Time O(N⁴), Space O(N²).
    """
    n = matrix.shape[0]
    if n < 2:
        return []

    def _is_disconnected(sub_mat: NDArray[np.int_]) -> bool:
        if sub_mat.shape[0] < 2:
            return False
        undirected = sub_mat.astype(bool) | sub_mat.astype(bool).T
        n_comp, _ = connected_components(csr_matrix(undirected), directed=False)
        return n_comp > 1

    # 1. Collect all disconnected intervals
    disconnected_intervals: list[set[int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            indices = list(range(i, j + 1))
            sub = matrix[np.ix_(indices, indices)]
            if _is_disconnected(sub):
                disconnected_intervals.append(set(indices))

    # 2. Filter to maximal sets
    disconnected_intervals.sort(key=len, reverse=True)
    maximal: list[set[int]] = []
    for current in disconnected_intervals:
        if not any(current.issubset(existing) for existing in maximal):
            maximal.append(current)

    # 3. Extract submatrices
    maximal.sort(key=lambda s: min(s))
    results: list[NDArray[np.int_]] = []
    for idx_set in maximal:
        sorted_idx = sorted(idx_set)
        results.append(matrix[np.ix_(sorted_idx, sorted_idx)].astype(int))
    return results
