"""Stateless utility helpers: signatures, depth equivalents, metrics, equality."""

from __future__ import annotations

from poset_operad.utils.collections import count_satisfying_posets, get_satisfying_posets
from poset_operad.utils.depths import generate_semi_depth_equivalents
from poset_operad.utils.equality import are_poset_structures_strictly_equal
from poset_operad.utils.metrics import compute_triangular_saturation_metrics
from poset_operad.utils.signatures import get_poset_signature, get_signature

__all__ = [
    "get_signature",
    "get_poset_signature",
    "generate_semi_depth_equivalents",
    "compute_triangular_saturation_metrics",
    "are_poset_structures_strictly_equal",
    "get_satisfying_posets",
    "count_satisfying_posets",
]
