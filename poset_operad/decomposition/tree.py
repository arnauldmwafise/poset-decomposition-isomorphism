"""
poset_operad.decomposition.tree
=================================

Recursive poset decomposition tree.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Union, Callable
import numpy as np

from poset_operad.core.backend import xp, GPU_AVAILABLE, logger
from poset_operad.core.predicates import is_non_trivial_poset
from poset_operad.core.submatrix import get_principal_submatrix
from poset_operad.decomposition.boundary import extract_disconnected_core_with_depths


def decompose_dual_core_into_components(
    poset_matrix: Any,
) -> tuple[list[tuple[Any, Any]], list[Any]]:
    """Decompose the disconnected core of a dualizable poset into connected components."""
    poset_matrix = xp.asarray(poset_matrix)
    core_extraction = extract_disconnected_core_with_depths(poset_matrix)
    
    # If the current subposet has no saturated boundaries, check for a direct sum split
    if core_extraction is None:
        from poset_operad.decomposition.direct_sum import extract_direct_sum_components
        from poset_operad.core.predicates import is_disconnected_poset
        if is_disconnected_poset(poset_matrix):
            submatrices = extract_direct_sum_components(poset_matrix)
            # Filter out singletons to match the structural non-trivial descent criteria
            submatrices = [sub for sub in submatrices if sub.shape[0] > 1]
            return [(poset_matrix, sub) for sub in submatrices], submatrices
        return [], []

    disconn_core, _metadata = core_extraction
    n = disconn_core.shape[0]
    if n == 0:
        return [], []

    if GPU_AVAILABLE:
        import cupy as cp
        M_undirected = ((disconn_core != 0) | (disconn_core.T != 0)).astype(cp.int32)
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
        
        disconn_core_np = disconn_core.get() if hasattr(disconn_core, 'get') else np.asarray(disconn_core)
        _, labels = connected_components(csr_matrix(disconn_core_np), directed=False)
        unique_labels = np.unique(labels)
        components_indices = [
            np.where(labels == label)[0].tolist() for label in unique_labels
        ]

    submatrices = [
        get_principal_submatrix(disconn_core, idx_list)
        for idx_list in components_indices
    ]
    paired_results = [(poset_matrix, sub) for sub in submatrices]
    return paired_results, submatrices


def build_poset_decomposition_tree(
    root_matrix: Any,
) -> list[list[Any]]:
    """Recursively decompose *root_matrix* into a hierarchy of non-trivial components."""
    root_matrix = xp.asarray(root_matrix)
    hierarchy: list[list[Any]] = []
    current_pool = [root_matrix]
    level = 0

    while current_pool:
        level_components: list[Any] = []
        for matrix in current_pool:
            _, components = decompose_dual_core_into_components(matrix)
            level_components.extend(components)

        if not level_components:
            break

        logger.info(f"Decomposition Tree Layer {level} completed. Discovered {len(level_components)} sub-components.")
        hierarchy.append(level_components)
        current_pool = [m for m in level_components if is_non_trivial_poset(m)]
        level += 1

    return hierarchy


def update_nested_posets(
    collection: list | tuple | Any,
    func: Callable,
) -> list | tuple | Any:
    """Recursively apply *func* to every 2-D ndarray found in *collection*."""
    if isinstance(collection, xp.ndarray) and collection.ndim == 2:
        result = func(collection)
        if isinstance(result, list) and len(result) > 0:
            return result
        return collection
    if isinstance(collection, list):
        return [update_nested_posets(item, func) for item in collection]
    if isinstance(collection, tuple):
        return tuple(update_nested_posets(item, func) for item in collection)
    return collection
