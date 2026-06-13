"""
poset_operad.core.submatrix
============================

Primitive operations for extracting sub-structures from poset matrices.
All functions are stateless and return new arrays (views where safe).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def get_principal_submatrix(
    matrix: NDArray[np.int_],
    index_set: list[int] | set[int] | NDArray[np.int_],
) -> NDArray[np.int_]:
    """Return the principal submatrix formed by *index_set* rows and columns.

    Uses ``np.ix_`` for O(k²) cross-product indexing, where *k* is the size
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

    Examples
    --------
    >>> import numpy as np
    >>> M = np.arange(16).reshape(4, 4)
    >>> get_principal_submatrix(M, [0, 2])
    array([[ 0,  2],
           [ 8, 10]])
    """
    indices = list(index_set)
    return matrix[np.ix_(indices, indices)]


def get_maximal_elements(posetmatrix: NDArray[np.int_]) -> list[int]:
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
    if posetmatrix.size == 0:
        return []
    col_sums = np.sum(posetmatrix, axis=0)
    return np.where(col_sums == 1)[0].tolist()


def get_minimal_elements(posetmatrix: NDArray[np.int_]) -> list[int]:
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
    if posetmatrix.size == 0:
        return []
    row_sums = np.sum(posetmatrix, axis=1)
    return np.where(row_sums == 1)[0].tolist()
