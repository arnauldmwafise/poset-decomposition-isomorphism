"""
poset_operad
============

Structural decomposition and isomorphism verification for Partially Ordered Set
(Poset) matrices via the Poset Operad.

Public API
----------
The library is organised into four sub-packages:

- ``poset_operad.core``           – atomic predicates and matrix primitives
- ``poset_operad.decomposition``  – boundary-depth extraction, direct-sum, tree
- ``poset_operad.isomorphism``    – hierarchical, direct-sum, saturation checks
- ``poset_operad.utils``          – signatures, depth equivalents, metrics

Convenience re-exports are provided here for the most commonly used symbols.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# ── Dynamic Switching Router Elements ─────────────────────────────────────────
from poset_operad.core.backend import GPU_AVAILABLE, logger, xp

try:
    __version__: str = version("poset-operad")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0.dev"

# ── Core predicates ───────────────────────────────────────────────────────────
from poset_operad.core.predicates import (
    check_poset_connectivity,
    is_chain_or_antichain,
    is_disconnected_poset,
    is_non_partial_semi_equidualizable,
    is_non_trivial_poset,
    is_partial_semi_equidualizable,
    is_trivial_poset,
)

# ── Decomposition ─────────────────────────────────────────────────────────────
from poset_operad.decomposition.boundary import (
    extract_disconnected_core_with_depths,
    extract_semiequidual_subcomponents,
)
from poset_operad.decomposition.direct_sum import (
    extract_direct_sum_components,
    extract_maximal_disconnected_submatrices,
    extract_poset_direct_sum_components,
)
from poset_operad.decomposition.tree import (
    build_poset_decomposition_tree,
    decompose_dual_core_into_components,
)

# ── Isomorphism ───────────────────────────────────────────────────────────────
from poset_operad.isomorphism.direct_sum import (
    verify_isomorphism_via_direct_sum_decomposition,
)
from poset_operad.isomorphism.hierarchical import (
    verify_poset_isomorphism_hierarchical,
)
from poset_operad.isomorphism.saturation import (
    verify_isomorphism_via_maximal_disconnection_and_saturation,
)

# ── Utils ─────────────────────────────────────────────────────────────────────
from poset_operad.utils.depths import generate_semi_depth_equivalents
from poset_operad.utils.metrics import compute_triangular_saturation_metrics
from poset_operad.utils.signatures import get_poset_signature, get_signature

__all__: list[str] = [
    # hardware telemetry
    "xp",
    "GPU_AVAILABLE",
    "logger",
    # meta
    "__version__",
    # core
    "is_trivial_poset",
    "is_non_trivial_poset",
    "is_chain_or_antichain",
    "is_partial_semi_equidualizable",
    "is_non_partial_semi_equidualizable",
    "is_disconnected_poset",
    "check_poset_connectivity",
    # decomposition
    "extract_semiequidual_subcomponents",
    "extract_disconnected_core_with_depths",
    "extract_direct_sum_components",
    "extract_poset_direct_sum_components",
    "extract_maximal_disconnected_submatrices",
    "build_poset_decomposition_tree",
    "decompose_dual_core_into_components",
    # isomorphism
    "verify_poset_isomorphism_hierarchical",
    "verify_isomorphism_via_direct_sum_decomposition",
    "verify_isomorphism_via_maximal_disconnection_and_saturation",
    # utils
    "get_signature",
    "get_poset_signature",
    "generate_semi_depth_equivalents",
    "compute_triangular_saturation_metrics",
]
