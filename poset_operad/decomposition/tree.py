"""
poset_operad.decomposition.tree
=================================

Recursive poset decomposition tree.

The *decomposition tree* captures the genealogy of a poset: it records how a
complex order is built from smaller, connected sub-orders by iteratively
stripping boundary layers and splitting disconnected cores.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from numpy.typing import NDArray

from poset_operad.core.predicates import is_non_trivial_poset
from poset_operad.core.submatrix import get_principal_submatrix
from poset_operad.decomposition.boundary import extract_disconnected_core_with_depths


# ── Public API ────────────────────────────────────────────────────────────────

def decompose_dual_core_into_components(
    poset_matrix: NDArray[np.int_],
) -> tuple[list[tuple[NDArray[np.int_], NDArray[np.int_]]], list[NDArray[np.int_]]]:
    """Decompose the disconnected core of a dualizable poset into connected components.

    Steps
    -----
    1. Extract the disconnected core via
       :func:`~poset_operad.decomposition.boundary.extract_disconnected_core_with_depths`.
    2. Convert the core to an undirected NetworkX graph.
    3. Identify connected components (disjoint index sets).
    4. Extract each as a principal submatrix.

    Parameters
    ----------
    poset_matrix:
        n×n dualizable poset adjacency matrix.

    Returns
    -------
    paired_results : list[tuple[np.ndarray, np.ndarray]]
        ``(original_matrix, component_submatrix)`` pairs for lineage tracking.
    components : list[np.ndarray]
        Standalone connected-component submatrices.

    Complexity
    ----------
    Time O(n²)–O(n³), Space O(n²).
    """
    core_extraction = extract_disconnected_core_with_depths(poset_matrix)
    if core_extraction is None:
        return [], []

    disconn_core, _metadata = core_extraction

    graph = nx.from_numpy_array(disconn_core)
    components_indices = [sorted(c) for c in nx.connected_components(graph)]

    submatrices = [
        get_principal_submatrix(disconn_core, idx_list)
        for idx_list in components_indices
    ]
    paired_results = [(poset_matrix, sub) for sub in submatrices]
    return paired_results, submatrices


def build_poset_decomposition_tree(
    root_matrix: NDArray[np.int_],
) -> list[list[NDArray[np.int_]]]:
    """Recursively decompose *root_matrix* into a hierarchy of non-trivial components.

    The tree is built breadth-first:

    * Each *level* holds the components discovered at that recursion depth.
    * Only non-trivial components (neither chain nor antichain) are processed
      further at the next level.
    * Terminates when no new components are produced or all are trivial.

    Parameters
    ----------
    root_matrix:
        The initial n×n poset matrix to decompose.

    Returns
    -------
    list[list[np.ndarray]]
        Nested hierarchy; ``result[k]`` contains the components at depth ``k``.

    Complexity
    ----------
    Time O(K · n²), Space O(K · n²), where K is the total number of
    sub-matrices discovered across all levels.
    """
    hierarchy: list[list[NDArray[np.int_]]] = []
    current_pool = [root_matrix]

    while current_pool:
        level_components: list[NDArray[np.int_]] = []
        for matrix in current_pool:
            _, components = decompose_dual_core_into_components(matrix)
            level_components.extend(components)

        if not level_components:
            break

        hierarchy.append(level_components)
        current_pool = [m for m in level_components if is_non_trivial_poset(m)]

    return hierarchy


def update_nested_posets(
    collection: list | tuple | NDArray[np.int_],
    func: object,
) -> list | tuple | NDArray[np.int_]:
    """Recursively apply *func* to every 2-D ndarray found in *collection*.

    The original container types (``list`` / ``tuple``) are preserved.
    If *func* returns a non-empty list for an array, that list replaces the
    array; otherwise the array is kept unchanged.

    Parameters
    ----------
    collection:
        Nested structure potentially containing 2-D ndarrays.
    func:
        Transformation callable: ``np.ndarray → list``.

    Returns
    -------
    Same structure as input, with 2-D arrays replaced by ``func``'s output.

    Complexity
    ----------
    Time O(N · T), Space O(D + M) where D = max recursion depth.
    """
    if isinstance(collection, np.ndarray) and collection.ndim == 2:
        result = func(collection)  # type: ignore[operator]
        if isinstance(result, list) and len(result) > 0:
            return result
        return collection
    if isinstance(collection, list):
        return [update_nested_posets(item, func) for item in collection]
    if isinstance(collection, tuple):
        return tuple(update_nested_posets(item, func) for item in collection)
    return collection
