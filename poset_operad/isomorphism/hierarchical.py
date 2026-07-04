"""
poset_operad.isomorphism.hierarchical
=======================================

Hierarchical isomorphism verification via recursive boundary decomposition.

The algorithm proceeds top-down:

1. **Memoization** – a module-level SHA-256 cache avoids redundant comparisons.
2. **Global invariants** – shape and relation count must match.
3. **Recursive decomposition** – strip semi-equidual boundaries, compare
   depth equivalents, and recurse on the inner sub-matrices.
4. **Fallback** – full canonical tree comparison if decomposition is
   inconclusive.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from poset_operad.core.backend import xp, GPU_AVAILABLE, logger
from poset_operad.core.predicates import is_trivial_poset
from poset_operad.decomposition.boundary import extract_semiequidual_subcomponents
from poset_operad.decomposition.tree import build_poset_decomposition_tree
from poset_operad.utils.signatures import get_poset_signature
from poset_operad.utils.depths import generate_semi_depth_equivalents
from poset_operad.utils.equality import are_poset_structures_strictly_equal


# ── Module-level memoization cache ────────────────────────────────────────────
# Structuring tracking keys as string structures prevents memory pointers
# from causing alignment mismatches during multi-node cluster sync routines.
_isomorphism_cache: Dict[str, bool] = {}


def clear_cache() -> None:
    """Clear the module-level memoization cache.

    Call this between independent test runs to avoid stale entries when
    matrices with the same byte-content have different structural meanings.
    """
    _isomorphism_cache.clear()


def _matrix_hash(matrix: Any) -> str:
    """Return a deterministic SHA-256 hex digest of *matrix* content."""
    matrix = xp.asarray(matrix)
    if GPU_AVAILABLE:
        import cupy as cp
        return hashlib.sha256(cp.asnumpy(matrix).tobytes()).hexdigest()
    else:
        return hashlib.sha256(matrix.tobytes()).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def verify_poset_isomorphism_hierarchical(
    matrix_a: Any,
    matrix_b: Any,
    depth: int = 0,
) -> bool:
    """Verify structural isomorphism via hierarchical boundary decomposition.

    Parameters
    ----------
    matrix_a, matrix_b:
        n×n binary poset adjacency matrices to compare.
    depth:
        Internal recursion depth counter (do not set manually).

    Returns
    -------
    bool
        ``True`` if a structural bijection exists between the two posets.

    Complexity
    ----------
    Time O(n² · log n) per level, Space O(n²) plus cache entries.
    """
    matrix_a = xp.asarray(matrix_a)
    matrix_b = xp.asarray(matrix_b)

    h_a = _matrix_hash(matrix_a)
    h_b = _matrix_hash(matrix_b)
    pair_key = "-".join(sorted([h_a, h_b]))

    if pair_key in _isomorphism_cache:
        return _isomorphism_cache[pair_key]

    result = False

    try:
        # 2. Global invariants
        if matrix_a.shape != matrix_b.shape or int(xp.sum(matrix_a)) != int(xp.sum(matrix_b)):
            result = False

        # 3. Trivial base case
        elif is_trivial_poset(matrix_a) and is_trivial_poset(matrix_b):
            result = True

        else:
            # 4. Recursive decomposition
            extract_a = extract_semiequidual_subcomponents(matrix_a)
            extract_b = extract_semiequidual_subcomponents(matrix_b)

            if extract_a and extract_b:
                comps_a, _, depths_a = extract_a
                comps_b, _, depths_b = extract_b

                list_a = comps_a if isinstance(comps_a, list) else [comps_a]
                list_b = comps_b if isinstance(comps_b, list) else [comps_b]

                if (
                    all(m.shape < matrix_a.shape for m in list_a)
                    and len(list_a) == len(list_b)
                ):
                    equiv_b = generate_semi_depth_equivalents(depths_b)
                    if equiv_b is not None and depths_a in equiv_b:
                        # Sorting lists cleanly via permutation-invariant signatures
                        list_a = sorted(list_a, key=get_poset_signature)
                        list_b = sorted(list_b, key=get_poset_signature)

                        def _robust_match(sa: Any, sb: Any) -> bool:
                            if not verify_poset_isomorphism_hierarchical(sa, sb, depth + 1):
                                return False
                            t_a = build_poset_decomposition_tree(sa)
                            t_b = build_poset_decomposition_tree(sb)
                            return are_poset_structures_strictly_equal(t_a, t_b)

                        if all(_robust_match(sa, sb) for sa, sb in zip(list_a, list_b)):
                            result = True

            # 5. Canonical-tree fallback
            if not result:
                logger.debug(f"Boundary layers inconclusive at recursion depth {depth}. Triggering fallback canonical tree identity scans.")
                tree_a = build_poset_decomposition_tree(matrix_a)
                tree_b = build_poset_decomposition_tree(matrix_b)
                result = are_poset_structures_strictly_equal(tree_a, tree_b)

    except Exception as e:
        logger.error(f"Unexpected structural discrepancy managed at layer depth {depth}: {str(e)}")
        result = False

    _isomorphism_cache[pair_key] = result
    return result
