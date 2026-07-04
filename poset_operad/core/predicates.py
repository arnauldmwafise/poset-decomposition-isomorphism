"""
poset_operad.core.predicates
============================

Atomic boolean predicates for poset adjacency matrices.

Every function here is **pure** (no mutation, no global state) and operates
in O(n²) time or better.  They form the lowest layer of the library and are
depended upon by all higher modules.
"""

from __future__ import annotations

from typing import Any

from poset_operad.core.backend import xp, logger, get_connected_components_count


# ── Public API ────────────────────────────────────────────────────────────────

def is_trivial_poset(poset_matrix: Any) -> bool:
    """Return ``True`` iff *poset_matrix* is the 1×1 identity element ``[[1]]``."""
    return poset_matrix.shape == (1, 1) and int(poset_matrix.item()) == 1


def is_chain_or_antichain(matrix: Any) -> bool:
    """Return ``True`` iff *matrix* encodes a total order (chain) or identity relation (antichain)."""
    matrix = xp.asarray(matrix)
    n = matrix.shape[0]
    if n == 0:
        return True

    row_sums = xp.sum(matrix, axis=1)

    if xp.all(row_sums == 1):          # antichain / identity
        return True
    if xp.array_equal(xp.sort(row_sums), xp.arange(1, n + 1)):  # chain
        return True
    return False


def is_non_trivial_poset(matrix: Any) -> bool:
    """Return ``True`` iff *matrix* is neither a chain nor an antichain."""
    matrix = xp.asarray(matrix)
    n = matrix.shape[0]
    if n == 0:
        return False
    return not is_chain_or_antichain(matrix)


def check_poset_connectivity(poset_matrix: Any) -> bool:
    """Return ``True`` iff the underlying undirected graph of *poset_matrix* is connected."""
    poset_matrix = xp.asarray(poset_matrix)
    n = poset_matrix.shape[0]
    if n == 0:
        return False
    if n == 1:
        return is_trivial_poset(poset_matrix)
        
    return get_connected_components_count(poset_matrix) == 1


def is_disconnected_poset(poset_matrix: Any) -> bool:
    """Return ``True`` iff the poset is a direct sum of two or more components."""
    return not check_poset_connectivity(poset_matrix)


def is_partial_semi_equidualizable(M: Any) -> bool:
    """Return ``True`` iff *M* has saturated boundary layers whose removal yields a disconnected core."""
    M = xp.asarray(M)
    n = M.shape[0]
    if n == 0:
        return False

    i = j = 0
    while i < n and bool(xp.all(M[i:, i] == 1)):
        i += 1
    while j < n and bool(xp.all(M[n - j - 1, : n - j] == 1)):
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

    is_core_connected = (get_connected_components_count(m_core) == 1)
    
    if not is_core_connected:
        logger.info(f"Discovered partial semi-equidualizable lattice structure at depths i={i}, j={j}")
        
    return not is_core_connected


def is_non_partial_semi_equidualizable(poset_matrix: Any) -> bool:
    """Return ``True`` iff *poset_matrix* is **not** partially semi-equidualizable."""
    return not is_partial_semi_equidualizable(poset_matrix)
