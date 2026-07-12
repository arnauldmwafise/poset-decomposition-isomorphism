"""
poset_operad.decomposition.direct_sum
========================================

Direct-sum decomposition routines.
"""

from __future__ import annotations

from typing import Any, List, Union

import numpy as np
from poset_operad.core.backend import xp, GPU_AVAILABLE, logger
from poset_operad.core.predicates import is_disconnected_poset
from poset_operad.core.submatrix import get_principal_submatrix


def extract_direct_sum_components(
    poset_matrix: Any,
) -> list[Any]:
    """Return a list of principal submatrices representing each direct-sum component."""
    poset_matrix = xp.asarray(poset_matrix)
    n = poset_matrix.shape
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
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        
        poset_matrix_np = poset_matrix.get() if hasattr(poset_matrix, 'get') else np.asarray(poset_matrix)
        
        n_components, labels = connected_components(
            csr_matrix(poset_matrix_np), directed=True, connection="weak"
        )
        # FIX: Extract the 0th element from np.where to unpack the tuple array before calling .tolist()
        components_indices = [
            np.where(labels == i)[0].tolist() for i in range(n_components)
        ]

    return [poset_matrix[xp.ix_(idx_list, idx_list)] for idx_list in components_indices]


def extract_poset_direct_sum_components(
    matrix: Any,
) -> list[list[Any]] | None:
    """Identify transition points in saturated boundaries and extract direct-sum groups."""
    matrix = xp.asarray(matrix)
    dim = matrix.shape
    if dim == 0:
        return None

    extracted: list[list[Any]] = []
    grid_y, grid_x = xp.meshgrid(xp.arange(dim), xp.arange(dim), indexing='ij')

    col_mask = (matrix != 0) | (grid_y < grid_x)
    col_bounds = xp.all(col_mask, axis=0)
    transitions_f = xp.where(col_bounds[:-1] & ~col_bounds[1:])
    
    for t_idx in transitions_f.tolist():
        depth = int(t_idx) + 1
        inner = matrix[depth:, depth:]
        if inner.size > 0 and is_disconnected_poset(inner):
            logger.info(f"Transition point captured via forward boundary scan at column depth: {depth}")
            extracted.append(extract_direct_sum_components(inner))
            break

    backward_mask = (matrix != 0) | (grid_y > grid_y[::-1, :][:, xp.newaxis][grid_x])
    row_bounds = xp.all(backward_mask, axis=1)[::-1]
    transitions_b = xp.where(row_bounds[:-1] & ~row_bounds[1:])
    
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
    """Return all maximal principal submatrices that are graph-theoretically disconnected."""
    matrix = xp.asarray(matrix)
    n = matrix.shape
    if n < 2:
        return []

    def _is_disconnected(sub_mat: Any) -> bool:
        if sub_mat.shape < 2:
            return False
        undirected = sub_mat.astype(bool) | sub_mat.astype(bool).T
        
        if GPU_AVAILABLE:
            import cupy as cp
            dim = undirected.shape
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
            
            sub_mat_np = undirected.get() if hasattr(undirected, 'get') else np.asarray(undirected)
            n_comp, _ = connected_components(csr_matrix(sub_mat_np), directed=False)
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
