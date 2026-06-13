"""
poset_operad.core.predicates
============================

Atomic boolean predicates for poset adjacency matrices.

Every function here is **pure** (no mutation, no global state) and operates
in O(n²) time or better.  They form the lowest layer of the library and are
depended upon by all higher modules.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from numpy.typing import NDArray


# ── Public API ────────────────────────────────────────────────────────────────

def is_trivial_poset(poset_matrix: NDArray[np.int_]) -> bool:
    """Return ``True`` iff *poset_matrix* is the 1×1 identity element ``[[1]]``.

    In the Poset Operad the **trivial poset** is the unique poset on a single
    element: it contains only the reflexive relation ``x ≤ x`` and acts as the
    operadic identity.

    Parameters
    ----------
    poset_matrix:
        Square binary ndarray.

    Returns
    -------
    bool
        ``True`` if shape is ``(1, 1)`` and the sole entry equals 1.

    Complexity
    ----------
    Time O(1), Space O(1).

    Examples
    --------
    >>> import numpy as np
    >>> is_trivial_poset(np.array([[1]]))
    True
    >>> is_trivial_poset(np.array([[1, 0], [1, 1]]))
    False
    """
    return poset_matrix.shape == (1, 1) and poset_matrix.item() == 1


def is_chain_or_antichain(matrix: NDArray[np.int_]) -> bool:
    """Return ``True`` iff *matrix* encodes a total order (chain) or identity relation (antichain).

    Classification is based on the row-sum distribution:

    * **Antichain** – every row sums to exactly 1 (reflexive relation only).
    * **Chain** – sorted row sums equal ``[1, 2, …, n]``, i.e. each element
      has a unique "depth" in the total order.

    Parameters
    ----------
    matrix:
        Square binary ndarray.

    Returns
    -------
    bool

    Complexity
    ----------
    Time O(n²), Space O(n).

    Examples
    --------
    >>> import numpy as np
    >>> chain = np.array([[1,0,0],[1,1,0],[1,1,1]])
    >>> is_chain_or_antichain(chain)
    True
    >>> is_chain_or_antichain(np.eye(3, dtype=int))
    True
    """
    n = matrix.shape[0]
    row_sums = np.sum(matrix, axis=1)

    if np.all(row_sums == 1):          # antichain / identity
        return True
    if np.array_equal(np.sort(row_sums), np.arange(1, n + 1)):  # chain
        return True
    return False


def is_non_trivial_poset(matrix: NDArray[np.int_]) -> bool:
    """Return ``True`` iff *matrix* is neither a chain nor an antichain.

    Non-trivial posets contain elements that are both comparable and
    incomparable (beyond reflexivity) and are the primary objects of
    interest for structural decomposition.

    Parameters
    ----------
    matrix:
        Square binary ndarray.

    Returns
    -------
    bool

    Complexity
    ----------
    Time O(n²), Space O(n).
    """
    n = matrix.shape[0]
    if n == 0:
        return False
    return not is_chain_or_antichain(matrix)


def check_poset_connectivity(poset_matrix: NDArray[np.int_]) -> bool:
    """Return ``True`` iff the underlying undirected graph of *poset_matrix* is connected.

    Connectivity is tested by treating directed edges as undirected (via
    ``networkx.from_numpy_array``) and applying BFS/DFS to count components.

    Parameters
    ----------
    poset_matrix:
        Square binary ndarray.

    Returns
    -------
    bool
        ``True`` when there is exactly one connected component.

    Complexity
    ----------
    Time O(n + E), Space O(n + E).

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[1,0],[1,1]])
    >>> check_poset_connectivity(M)
    True
    """
    n = poset_matrix.shape[0]
    if n == 0:
        return False
    if n == 1:
        return is_trivial_poset(poset_matrix)
    graph = nx.from_numpy_array(poset_matrix)
    return nx.is_connected(graph)


def is_disconnected_poset(poset_matrix: NDArray[np.int_]) -> bool:
    """Return ``True`` iff the poset is a direct sum of two or more components.

    This is the logical negation of :func:`check_poset_connectivity`.

    Parameters
    ----------
    poset_matrix:
        Square binary ndarray.

    Returns
    -------
    bool

    Complexity
    ----------
    Time O(n + E), Space O(n + E).
    """
    return not check_poset_connectivity(poset_matrix)


def is_partial_semi_equidualizable(M: NDArray[np.int_]) -> bool:
    """Return ``True`` iff *M* has saturated boundary layers whose removal yields a disconnected core.

    Algorithm
    ---------
    1. **Boundary Scan** – scan leading columns for *semi-right depth* ``i``
       and trailing rows for *semi-left depth* ``j``.
    2. **Guard** – if both depths are zero the matrix has no detectable
       boundary structure.
    3. **Core Extraction** – slice ``M[i:n-j, i:n-j]`` (or the appropriate
       one-sided slice).
    4. **Disconnectivity Check** – the core must be graph-theoretically
       disconnected.

    Parameters
    ----------
    M:
        Square binary ndarray representing a poset adjacency matrix.

    Returns
    -------
    bool

    Complexity
    ----------
    Time O(n²), Space O(n²).

    References
    ----------
    See the docstring in the original research notebook for the scientific
    context within the Poset Operad.
    """
    n = M.shape[0]
    i = j = 0

    while i < n and np.all(M[i:, i] == 1):
        i += 1
    while j < n and np.all(M[n - j - 1, : n - j] == 1):
        j += 1

    if i == 0 and j == 0:
        return False

    if i > 0 and j > 0:
        if i >= n - j:
            return False
        m_core = M[i : n - j, i : n - j]
    elif i > 0:
        m_core = M[i:, i:]
    else:
        m_core = M[: n - j, : n - j]

    if m_core.size < 2:
        return False

    return not nx.is_connected(nx.from_numpy_array(m_core))


def is_non_partial_semi_equidualizable(poset_matrix: NDArray[np.int_]) -> bool:
    """Return ``True`` iff *poset_matrix* is **not** partially semi-equidualizable.

    Identifies *irreducible* matrices that have no detectable boundary
    saturation and cannot be simplified by triangular boundary reduction.

    Parameters
    ----------
    poset_matrix:
        Square binary ndarray.

    Returns
    -------
    bool

    Complexity
    ----------
    Time O(n²), Space O(n²) — inherited from
    :func:`is_partial_semi_equidualizable`.
    """
    return not is_partial_semi_equidualizable(poset_matrix)
