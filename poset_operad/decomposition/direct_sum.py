"""
poset_operad.decomposition.direct_sum
========================================

Direct-sum decomposition routines.

A **direct sum** of posets corresponds to a block-diagonal structure in the
adjacency matrix: no element of one block relates to any element of another.
These functions identify such blocks, including maximal disconnected intervals.
"""

from __future__ import annotations

from typing import Any, List, Union

from poset_operad.core.backend import xp, GPU_AVAILABLE, logger
from poset_operad.core.predicates import is_disconnected_poset
from poset_operad.core.submatrix import get_principal_submatrix


# ── Public API ────────────────────────────────────────────────────────────────

def extract_direct_sum_components(
    poset_matrix: Any,
) -> list[Any]:
    """Return a list of principal submatrices representing each direct-sum component.

    Uses a parallel pointer-jumping relaxation solver natively inside VRAM if a 
    GPU is active, otherwise falls back to an optimized compiled SciPy graph routine.

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
    poset_matrix = xp.asarray(poset_matrix)
    n = poset_matrix.shape[0]
    if n == 0:
        return []

    if GPU_AVAILABLE:
        import cupy as cp
        M_undirected = ((poset_matrix != 0) | (poset_matrix.T != 0)).astype(cp.int32)
        cp.fill_diagonal(M_undirected, 1)
        
        labels = cp.arange(n, dtype=cp.int32)
        old_labels = cp.zeros(n, dtype=cp.int32)
        edges = cp.argwhere(M_undirected > 0)
        src, dst = edges[:, 0], edges[:, 1]
        
        while not cp.all(labels == old_labels):
            old_labels = labels.copy()
            cp.minimum.at(labels, src, old_labels[dst])
            cp.minimum.at(labels, dst, old_labels[src])
            labels = labels[labels]
            
        unique_labels = cp.unique(labels).tolist()
        components_indices = [
            cp.argwhere(labels == ul).flatten().tolist() for ul in unique_labels
        ]
    else:
        # Optimized CPU Fallback using compiled SciPy graph routines
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        import numpy as np
        
        n_components, labels = connected_components(
            csr_matrix(poset_matrix), directed=True, connection="weak"
        )
        components_indices = [
            np.where(labels == i)[0].tolist() for i in range(n_components)
        ]

    return [poset_matrix[xp.ix_(idx_list, idx_list)] for idx_list in components_indices]


def extract_poset_direct_sum_components(
    matrix: Any,
) -> list[list[Any]] | None:
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
    matrix = xp.asarray(matrix)
    dim = matrix.shape[0]
    if dim == 0:
        return None

    extracted: list[list[Any]] = []

    grid_y, grid_x = xp.meshgrid(xp.arange(dim), xp.arange(dim), indexing='ij')

    col_mask = (matrix != 0) | (grid_y < grid_x)
    col_bounds = xp.all(col_mask, axis=0)
    transitions_f = xp.where(col_bounds[:-1] & ~col_bounds[1:])[0]
    
    for t_idx in transitions_f.tolist():
        depth = int(t_idx) + 1
        inner = matrix[depth:, depth:]
        if inner.size > 0 and is_disconnected_poset(inner):
            logger.info(f"Transition point captured via forward boundary scan at column depth: {depth}")
            extracted.append(extract_direct_sum_components(inner))
            break

    backward_mask = (matrix != 0) | (grid_y > grid_y[::-1, :][:, xp.newaxis][grid_x])
    row_bounds = xp.all(backward_mask, axis=1)[::-1]
    transitions_b = xp.where(row_bounds[:-1] & ~row_bounds[1:])[0]
    
    for t_idx in transitions_b.tolist():
        depth = int(t_idx) + 1
        inner = matrix[:-depth, :-depth]
        if inner.size > 0 and is_disconnected_poset(inner):
            logger.info(f"Transition point captured via backward boundary scan at row depth: {depth}")
            extracted.append(extract_direct_sum_components(inner))
            break

    return extracted if extracted else None


def extract_maximal_disconnected_submatrices(
    matrix: Any,
) -> list[Any]:
    """Return all maximal principal submatrices that are graph-theoretically disconnected.

    A submatrix is *maximal* disconnected if it is disconnected and not
    contained within any larger disconnected principal submatrix.

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
    matrix = xp.asarray(matrix)
    n = matrix.shape[0]
    if n < 2:
        return []

    def _is_disconnected(sub_mat: Any) -> bool:
        if sub_mat.shape[0] < 2:
            return False
        undirected = sub_mat.astype(bool) | sub_mat.astype(bool).T
        
        if GPU_AVAILABLE:
            import cupy as cp
            dim = undirected.shape[0]
            labels = cp.arange(dim, dtype=cp.int32)
            old_labels = cp.zeros(dim, dtype=cp.int32)
            edges = cp.argwhere(undirected > 0)
            src, dst = edges[:, 0], edges[:, 1]
            
            while not cp.all(labels == old_labels):
                old_labels = labels.copy()
                cp.minimum.at(labels, src, old_labels[dst])
                cp.minimum.at(labels, dst, old_labels[src])
                labels = labels[labels]
            return int(cp.unique(labels).size) > 1
        else:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            n_comp, _ = connected_components(csr_matrix(undirected), directed=False)
            return n_comp > 1

    disconnected_intervals: list[set[int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            indices = list(range(i, j + 1))
            sub = matrix[xp.ix_(indices, indices)]
            if _is_disconnected(sub):
                disconnected_intervals.append(set(indices))

    disconnected_intervals.sort(key=len, reverse=True)
    maximal: list[set[int]] = []
    for current in disconnected_intervals:
        if not any(current.issubset(existing) for existing in maximal):
            maximal.append(current)

    maximal.sort(key=lambda s: min(s))
    results: list[Any] = []
    for idx_set in maximal:
        sorted_idx = sorted(idx_set)
        results.append(matrix[xp.ix_(sorted_idx, sorted_idx)].astype(int))
        
    logger.debug(f"Partitioned source into {len(results)} maximal independent parallel submatrices.")
    return results
