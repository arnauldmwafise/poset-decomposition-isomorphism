"""
poset_operad.utils.metrics
============================

Scalar invariants derived from the triangular structure of poset matrices.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_triangular_saturation_metrics(
    matrix: NDArray[np.int_],
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
    n = matrix.shape[0]
    if n == 0:
        return 0, 0

    lower_tri: NDArray[np.bool_] = np.tril(np.ones((n, n), dtype=bool))
    cond: NDArray[np.bool_] = (matrix == 1) | (~lower_tri)

    rows = int(np.count_nonzero(np.all(cond, axis=1)))
    cols = int(np.count_nonzero(np.all(cond, axis=0)))
    return rows, cols
