"""
poset_operad.core.submatrix
============================

Primitive operations for extracting sub-structures from poset matrices.
All functions are stateless and return new arrays (views where safe).
"""

from __future__ import annotations

from typing import Any, List, Union

from poset_operad.core.backend import xp, logger


def get_principal_submatrix(
    matrix: Any,
    index_set: list[int] | set[int] | Any,
) -> Any:
    """Return the principal submatrix formed by *index_set* rows and columns.

    Uses ``xp.ix_`` for O(k²) cross-product indexing, where *k* is the size
    of *index_set*.

    Parameters
    ----------
    matrix:
        Source n×n square matrix.
    index_set:
        Row/column indices to preserve (order is respected).

    Returns
    -------
    np.ndarray
        A new k×k array containing the selected principal sub-block.

    Complexity
    ----------
    Time O(k²), Space O(k²).
    """
    matrix = xp.asarray(matrix)
    # Convert index set cleanly matching active device context layout
    indices = list(index_set) if isinstance(index_set, (list, set)) else xp.asarray(index_set).tolist()
    
    return matrix[xp.ix_(indices, indices)]


def get_maximal_elements(posetmatrix: Any) -> list[int]:
    r"""Return column indices of all maximal elements of *posetmatrix*.

    An element ``xⱼ`` is *maximal* if no other element strictly exceeds it,
    which in the adjacency matrix means column ``j`` has sum exactly 1
    (only the reflexive entry ``M[j, j] = 1``).

    Parameters
    ----------
    posetmatrix:
        n×n binary ndarray where ``M[i, j] = 1`` ↔ ``xᵢ ≤ xⱼ``.

    Returns
    -------
    list[int]
        Sorted list of column indices.

    Complexity
    ----------
    Time O(n²), Space O(n).
    """
    posetmatrix = xp.asarray(posetmatrix)
    if posetmatrix.size == 0:
        return []
        
    # Parallel column sum reduction map executed on active core device
    col_sums = xp.sum(posetmatrix, axis=0)
    max_indices = xp.where(col_sums == 1)[0]
    
    return max_indices.tolist()


def get_minimal_elements(posetmatrix: Any) -> list[int]:
    r"""Return row indices of all minimal elements of *posetmatrix*.

    An element ``xᵢ`` is *minimal* if no other element strictly precedes it,
    which in the adjacency matrix means row ``i`` has sum exactly 1 (only
    ``M[i, i] = 1``).

    Parameters
    ----------
    posetmatrix:
        n×n binary ndarray where ``M[i, j] = 1`` ↔ ``xᵢ ≤ xⱼ``.

    Returns
    -------
    list[int]
        Sorted list of row indices.

    Complexity
    ----------
    Time O(n²), Space O(n).
    """
    posetmatrix = xp.asarray(posetmatrix)
    if posetmatrix.size == 0:
        return []
        
    # Parallel row sum reduction map executed on active core device
    row_sums = xp.sum(posetmatrix, axis=1)
    min_indices = xp.where(row_sums == 1)[0]
    
    return min_indices.tolist()
