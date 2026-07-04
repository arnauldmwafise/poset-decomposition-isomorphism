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

import hashlib
from typing import Any

from poset_operad.core.backend import xp


def get_signature(matrix: Any) -> int:
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
    matrix = xp.asarray(matrix)
    if matrix.size == 0:
        return 0

    # 1. Hardware-accelerated row and column vector sum reductions
    row_sums = xp.sort(xp.sum(matrix, axis=1)).tolist()
    col_sums = xp.sort(xp.sum(matrix, axis=0)).tolist()

    # 2. Construct a cross-platform deterministic cryptographic token
    # This prevents randomized salting behaviors during cluster node routing transitions
    signature_string = f"{matrix.shape}-{row_sums}-{col_sums}"
    sha256_hash = hashlib.sha256(signature_string.encode('utf-8')).hexdigest()
    
    # Slice the token to return a clean, standard 64-bit signed integer representation
    return int(sha256_hash[:16], 16)


def get_poset_signature(matrix: Any) -> int:
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
