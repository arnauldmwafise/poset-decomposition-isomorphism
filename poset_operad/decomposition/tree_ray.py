"""
poset_operad.decomposition.tree_ray
====================================

Distributed Cluster-Orchestrated Poset Decomposition Tree using Ray Core.
"""

from __future__ import annotations

import os
from typing import Any, List, Tuple
import numpy as np
import ray

from poset_operad.core.backend import xp, logger
from poset_operad.decomposition.tree import decompose_dual_core_into_components


@ray.remote
def ray_remote_decompose_node(matrix: Any) -> tuple[list[tuple[Any, Any]], list[Any]]:
    """Ray Remote Task that computes a single node decomposition on a cluster worker."""
    matrix_np = np.asarray(matrix) if hasattr(matrix, 'get') else matrix
    paired_results, submatrices = decompose_dual_core_into_components(matrix_np)
    serializable_subs = [m.get() if hasattr(m, 'get') else m for m in submatrices]
    return paired_results, serializable_subs


def build_poset_decomposition_tree_distributed(root_matrix: Any) -> list[list[Any]]:
    """Recursively decomposes *root_matrix* concurrently across a distributed Ray cluster."""
    if not ray.is_initialized():
        logger.info("Initializing background distributed Ray computing context node...")
        ray.init(ignore_reinit_error=True)

    root_np = root_matrix.get() if hasattr(root_matrix, 'get') else np.asarray(root_matrix)
    hierarchy: list[list[Any]] = []
    
    # Pool tasks using Ray Object Futures
    current_level_futures = [ray_remote_decompose_node.remote(root_np)]
    level = 0

    while current_level_futures:
        # Block and resolve all level branches across nodes concurrently
        resolved_results = ray.get(current_level_futures)
        
        level_components: list[Any] = []
        next_level_pool: list[Any] = []
        
        for paired, submatrices in resolved_results:
            level_components.extend(submatrices)
            next_level_pool.extend(submatrices)
            
        if not level_components:
            break
            
        logger.info(f"Ray Cluster Engine: Level {level} completed. Discovered {len(level_components)} sub-components.")
        hierarchy.append(level_components)
        
        from poset_operad.core.predicates import is_non_trivial_poset
        current_level_futures = [
            ray_remote_decompose_node.remote(m) 
            for m in next_level_pool if is_non_trivial_poset(m)
        ]
        level += 1

    return hierarchy


def shutdown_cluster_engine() -> None:
    """Terminates the distributed Ray cluster runtime context securely."""
    if ray.is_initialized():
        logger.info("Tearing down distributed Ray core cluster session context safely.")
        ray.shutdown()
