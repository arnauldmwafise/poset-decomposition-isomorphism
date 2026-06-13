"""
poset_operad.decomposition.boundary
=====================================

Boundary-depth scanning and disconnected-core extraction.

These routines identify the *semi-left* and *semi-right* depth layers of a
poset matrix and extract the inner kernel that remains after those layers are
stripped.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from poset_operad.core.predicates import check_poset_connectivity
from poset_operad.core.submatrix import get_maximal_elements, get_minimal_elements


# ── Public API ────────────────────────────────────────────────────────────────

def extract_semiequidual_subcomponents(
    matrix: NDArray[np.int_],
) -> tuple[list[NDArray[np.int_]], str, tuple[int, int]]:
    """Strip saturated boundary layers and return the inner sub-matrix.

    Computes the *semi-right depth* ``d1`` (leading saturated columns) and
    the *semi-left depth* ``d2`` (trailing saturated rows), then slices the
    matrix to expose the non-trivial inner kernel.

    Parameters
    ----------
    matrix:
        n×n numerical or boolean ndarray.

    Returns
    -------
    submatrices : list[np.ndarray]
        A one-element list ``[sub_matrix]``.
    connectivity : str
        Defaults to ``"connected"`` (informational).
    depths : tuple[int, int]
        The computed ``(d1, d2)`` boundary depths.

    Complexity
    ----------
    Time O(n²), Space O(n²) for mask creation; O(1) for the output view.

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[1,0,0],[1,1,0],[1,0,1]])
    >>> sub, status, (d1, d2) = extract_semiequidual_subcomponents(M)
    >>> d1
    1
    """
    n = matrix.shape[0]
    if n == 0:
        return [matrix], "connected", (0, 0)

    # Semi-right depth: leading columns fully saturated
    d1_mask = (matrix != 0) | (np.arange(n)[:, None] < np.arange(n))
    d1_valid = np.all(d1_mask, axis=0)
    d1 = int(np.argmin(d1_valid)) if not np.all(d1_valid) else n

    # Semi-left depth: trailing rows fully saturated
    d2_mask = (matrix != 0) | (np.arange(n) > np.arange(n)[::-1][:, None])
    d2_valid = np.all(d2_mask, axis=1)[::-1]
    d2 = int(np.argmin(d2_valid)) if not np.all(d2_valid) else n

    if d1 == 0 and d2 == 0:
        return [matrix], "connected", (0, 0)

    if d1 > 0 and d2 > 0 and d1 == d2:
        sub = matrix[d1 : n - d2, d1 : n - d2]
    elif d1 > 0:
        sub = matrix[d1:, d1:]
    else:
        sub = matrix[: n - d2, : n - d2]

    return [sub], "connected", (d1, d2)


def extract_disconnected_core_with_depths(
    poset_matrix: NDArray[np.int_],
) -> tuple[NDArray[np.int_], dict[str, tuple[int, ...]]] | None:
    """Extract the internal disconnected sub-matrix and report boundary depths.

    Identifies *semi-depths* and returns the disconnected core, together
    with metadata describing which dualizability case was satisfied:

    * ``"depth1,depth2"`` – double-dualizable (symmetric equal depths)
    * ``"depth1"``        – semi-right dualizable
    * ``"depth2"``        – semi-left dualizable

    Parameters
    ----------
    poset_matrix:
        n×n binary ndarray.

    Returns
    -------
    (submatrix, metadata) or ``None``
        ``submatrix`` is the principal disconnected core.
        ``metadata`` is a dict mapping the case name to its depth tuple.
        Returns ``None`` if no disconnected core is found.

    Complexity
    ----------
    Time O(n²), Space O(n²).
    """
    n = poset_matrix.shape[0]
    depth1 = depth2 = 0

    for i in range(n):
        if np.all(poset_matrix[i:, i]):
            depth1 += 1
        else:
            break

    for j in range(n):
        row_idx = n - j - 1
        if np.all(poset_matrix[row_idx, : row_idx + 1]):
            depth2 += 1
        else:
            break

    # Case 1: Double Dualizable
    if depth1 > 0 and depth1 == depth2:
        if len(get_maximal_elements(poset_matrix)) == len(
            get_minimal_elements(poset_matrix)
        ):
            core = poset_matrix[depth1 : n - depth2, depth1 : n - depth2]
            return core, {"depth1,depth2": (depth1, depth2)}

    # Case 2: Semi-Right Dualizable
    if depth1 > 0:
        sub = poset_matrix[depth1:, depth1:]
        if not check_poset_connectivity(sub):
            return sub, {"depth1": depth1}

    # Case 3: Semi-Left Dualizable
    if depth2 > 0:
        sub = poset_matrix[: n - depth2, : n - depth2]
        if not check_poset_connectivity(sub):
            return sub, {"depth2": depth2}

    return None
