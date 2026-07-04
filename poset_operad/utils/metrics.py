"""
poset_operad.utils.metrics
============================

Scalar invariants derived from the triangular structure of poset matrices.
"""

from __future__ import annotations

from typing import Any, Tuple

from poset_operad.core.backend import xp


def compute_triangular_saturation_metrics(
    matrix: Any,
) -> tuple[int, int]:
    """Return ``(saturated_rows, saturated_cols)`` for the lower-triangular mask.

    A row (or column) is *saturated* if every element required by the lower-
    triangular mask to be 1 actually is 1.  Formally, it checks:

        ``(M == 1) | ~lower_tri``  is all-True for that row / column.

    Parameters
    ----------
    matrix:
        n×n binary poset adjacency matrix.

    Returns
    -------
    tuple[int, int]
        ``(row_saturation_count, col_saturation_count)``.

    Complexity
    ----------
    Time O(n²), Space O(n²).

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[1,0,0],[1,1,0],[1,1,1]])
    >>> compute_triangular_saturation_metrics(M)
    (3, 3)
    """
    matrix = xp.asarray(matrix)
    n = matrix.shape[0]
    if n == 0:
        return 0, 0

    # 1. Create an architecture-aware Lower Triangular Mask natively
    lower_tri = xp.tril(xp.ones((n, n), dtype=bool))
    
    # 2. Vectorized verification without device-to-host memory syncs
    cond = (matrix == 1) | (~lower_tri)

    # 3. Parallel aggregation across available execution cores
    rows = int(xp.sum(xp.all(cond, axis=1)))
    cols = int(xp.sum(xp.all(cond, axis=0)))
    
    return rows, cols
