"""
poset_operad.utils.signatures
================================

Permutation-invariant structural hash functions for poset matrices.

Both functions create a hash that is identical for any two matrices that are
related by a simultaneous row-and-column permutation (same structure, different
element labelling).  They are **heuristics**; collisions are possible but
rare in practice for posets of modest size.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def get_signature(matrix: NDArray[np.int_]) -> int:
    """Return a permutation-invariant integer hash for *matrix*.

    The hash combines:

    * The matrix shape.
    * Sorted row sums.
    * Sorted column sums.

    Parameters
    ----------
    matrix:
        Numerical ndarray of any shape.

    Returns
    -------
    int

    Complexity
    ----------
    Time O(n² + n log n), Space O(n).
    """
    if matrix.size == 0:
        return hash(None)
    rs = tuple(sorted(map(int, np.sum(matrix, axis=1))))
    cs = tuple(sorted(map(int, np.sum(matrix, axis=0))))
    return hash((matrix.shape, rs, cs))


def get_poset_signature(matrix: NDArray[np.int_]) -> int:
    """Alias of :func:`get_signature` — preferred name in the isomorphism layer.

    Parameters
    ----------
    matrix:
        n×n binary poset adjacency matrix.

    Returns
    -------
    int
    """
    return get_signature(matrix)
