"""
poset_operad.isomorphism.saturation
====================================

Saturation-based poset matrix tree isomorphism verification.
"""

from __future__ import annotations

from typing import Any, Callable

from poset_operad.core.backend import xp, logger
from poset_operad.decomposition.boundary import extract_semiequidual_subcomponents
from poset_operad.decomposition.direct_sum import extract_maximal_disconnected_submatrices
from poset_operad.decomposition.tree import build_poset_decomposition_tree
from poset_operad.utils.equality import are_poset_structures_strictly_equal
from poset_operad.utils.signatures import get_poset_signature


def verify_isomorphism_via_maximal_disconnection_and_saturation(
    M1: Any, M2: Any
) -> bool:
    """Core Saturation Engine (Tier 3): Verifies isomorphism via maximal disconnections."""
    M1 = xp.asarray(M1)
    M2 = xp.asarray(M2)

    if M1.shape != M2.shape:
        return False

    # 1. CRITICAL RESEARCH CRITERIA: Check boundary profiles before doing heavy processing.
    # If the structures are completely non-semi-equidual (depths are 0,0), they elude
    # the Saturation engine. Reject immediately and return False.
    extract_a = extract_semiequidual_subcomponents(M1)
    extract_b = extract_semiequidual_subcomponents(M2)
    
    if extract_a and extract_b:
        _, _, depths_a = extract_a
        _, _, depths_b = extract_b
        if depths_a == (0, 0) or depths_b == (0, 0):
            return False

    # 2. Extract maximal graph-theoretically disconnected submatrices
    output1 = extract_maximal_disconnected_submatrices(M1)
    output2 = extract_maximal_disconnected_submatrices(M2)

    if len(output1) != len(output2):
        return False

    # If maximal disconnected partitions are found, match their signatures
    if len(output1) > 0:
        sorted_out1 = sorted(output1, key=get_poset_signature)
        sorted_out2 = sorted(output2, key=get_poset_signature)
        
        for sub1, sub2 in zip(sorted_out1, sorted_out2):
            if not are_poset_structures_strictly_equal(
                build_poset_decomposition_tree(sub1),
                build_poset_decomposition_tree(sub2)
            ):
                return False
        return True

    # Fall back to strict tree identity check only for valid semi-equidual elements
    return are_poset_structures_strictly_equal(
        build_poset_decomposition_tree(M1),
        build_poset_decomposition_tree(M2)
    )


def check_all_isomorphisms(
    bag: list[Any], 
    verification_func: Callable[[Any, Any], bool]
) -> dict[tuple[int, int], bool]:
    """Exhaustively maps an NxN pairwise check grid over a collection bag of matrices."""
    logger.info(f"Launching exhaustive pairwise evaluation sweep over a bag of {len(bag)} poset matrices ({len(bag)**2} checks).")
    results = {}
    for i, m1 in enumerate(bag):
        for j, m2 in enumerate(bag):
            results[(i, j)] = verification_func(m1, m2)
            
    logger.info("Pairwise isomorphism matrix evaluation completed successfully.")
    return results
