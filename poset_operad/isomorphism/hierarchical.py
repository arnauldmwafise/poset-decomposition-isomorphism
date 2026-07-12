"""
poset_operad.isomorphism.hierarchical
======================================

Hierarchical poset matrix tree isomorphism verification.
"""

from __future__ import annotations

import hashlib
from typing import Any

from poset_operad.core.backend import xp, GPU_AVAILABLE
from poset_operad.core.predicates import is_trivial_poset
from poset_operad.decomposition.boundary import extract_semiequidual_subcomponents
from poset_operad.decomposition.tree import build_poset_decomposition_tree
from poset_operad.utils.signatures import get_poset_signature
from poset_operad.utils.depths import generate_semi_depth_equivalents
from poset_operad.utils.equality import are_poset_structures_strictly_equal

_isomorphism_cache: dict[str, bool] = {}


def clear_cache() -> None:
    """Clear the memoization cache repository."""
    _isomorphism_cache.clear()


def _matrix_hash(matrix: Any) -> str:
    """Return a deterministic string hash of the matrix tensor array layout."""
    matrix = xp.asarray(matrix)
    if GPU_AVAILABLE:
        import cupy as cp
        return hashlib.sha256(cp.asnumpy(matrix).tobytes()).hexdigest()
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def verify_poset_isomorphism_hierarchical(matrix_a: Any, matrix_b: Any, depth: int = 0) -> bool:
    """Core Hierarchical Engine (Tier 1): Verifies isomorphism via boundary tree descent."""
    matrix_a = xp.asarray(matrix_a)
    matrix_b = xp.asarray(matrix_b)
    
    h_a = _matrix_hash(matrix_a)
    h_b = _matrix_hash(matrix_b)
    pair_key = "-".join(sorted([h_a, h_b]))
    
    if pair_key in _isomorphism_cache:
        return _isomorphism_cache[pair_key]
        
    res = False
    try:
        if matrix_a.shape != matrix_b.shape or int(xp.sum(matrix_a)) != int(xp.sum(matrix_b)):
            res = False
        elif is_trivial_poset(matrix_a) and is_trivial_poset(matrix_b):
            res = True
        else:
            extract_a = extract_semiequidual_subcomponents(matrix_a)
            extract_b = extract_semiequidual_subcomponents(matrix_b)
            
            if extract_a and extract_b:
                comps_a, _, depths_a = extract_a
                comps_b, _, depths_b = extract_b
                
                # CRITICAL RESEARCH CRITERIA: If no valid boundary layers can be extracted (0,0),
                # it is not a semi-equidual structure. Fail early and return False.
                if depths_a == (0, 0) or depths_b == (0, 0):
                    res = False
                else:
                    list_a = comps_a if isinstance(comps_a, list) else [comps_a]
                    list_b = comps_b if isinstance(comps_b, list) else [comps_b]
                    
                    if all(m.shape < matrix_a.shape for m in list_a) and len(list_a) == len(list_b):
                        equiv_b = generate_semi_depth_equivalents(depths_b)
                        if equiv_b is not None and depths_a in equiv_b:
                            list_a = sorted(list_a, key=get_poset_signature)
                            list_b = sorted(list_b, key=get_poset_signature)
                            
                            def _match(sa: Any, sb: Any) -> bool:
                                if not verify_poset_isomorphism_hierarchical(sa, sb, depth + 1):
                                    return False
                                return are_poset_structures_strictly_equal(
                                    build_poset_decomposition_tree(sa), 
                                    build_poset_decomposition_tree(sb)
                                )
                            if all(_match(sa, sb) for sa, sb in zip(list_a, list_b)):
                                res = True
                                
            # Only fall back to tree matching if a non-zero boundary structure has been processed
            if not res and depths_a != (0, 0) and depths_b != (0, 0):
                res = are_poset_structures_strictly_equal(
                    build_poset_decomposition_tree(matrix_a), 
                    build_poset_decomposition_tree(matrix_b)
                )
    except Exception:
        res = False
        
    _isomorphism_cache[pair_key] = res
    return res
