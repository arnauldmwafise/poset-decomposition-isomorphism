"""
poset_operad.decomposition.tree_ray
====================================

Distributed Cluster-Orchestrated Poset Decomposition Tree using Ray Core.
Spins up asynchronous task actors to solve recursive tree branches concurrently
across an entire multi-node hardware network grid.
"""

from __future__ import annotations

import os
from typing import Any, List, Tuple
import numpy as np

# Import Ray directly to hook into your distributed server topology
import ray

from poset_operad.core.backend import xp, logger, runtime_hardware_router
from poset_operad.decomposition.tree import decompose_dual_core_into_components


# ── Distributed Ray Remote Worker Directives ──────────────────────────────────

@ray.remote
def ray_remote_decompose_node(matrix: Any) -> tuple[list[tuple[Any, Any]], list[Any]]:
    """Ray Remote Task that computes a single node decomposition on a cluster worker.
    
    Automatically captures the local array context, evaluates the size-aware 
    hardware routing requirements, and triggers the matrix partition task.
    """
    # Force safe type coercion inside the isolated worker process memory map
    matrix_np = np.asarray(matrix) if hasattr(matrix, 'get') else matrix
    
    # Run the canonical modular decomposition logic natively on the assigned cluster node
    paired_results, submatrices = decompose_dual_core_into_components(matrix_np)
    
    # Convert any CuPy device tensor footprints back into serializable NumPy arrays 
    # before shipping memory blocks back across the distributed cluster network bus
    serializable_subs = [m.get() if hasattr(m, 'get') else m for m in submatrices]
    return paired_results, serializable_subs


# ── Public Distributed API ───────────────────────────────────────────────────

def build_poset_decomposition_tree_distributed(root_matrix: Any) -> list[list[Any]]:
    """Recursively decomposes *root_matrix* concurrently across a distributed Ray cluster.

    Builds the decomposition tree level-by-level using asynchronous task graphs,
    drastically reducing execution latency for large, complex matrices.

    Parameters
    ----------
    root_matrix:
        The initial N x N poset matrix to decompose across the cluster network.

    Returns
    -------
    list[list[np.ndarray]]
        Nested hierarchy; result[k] contains the components discovered at depth k.
    """
    # 1. Guarantee Ray cluster infrastructure context is alive and operational
    if not ray.is_initialized():
        logger.info("Initializing background distributed Ray computing context node...")
        ray.init(ignore_reinit_error=True)

    # Coerce root matrix into standard numpy footprint for multi-node transport serialization
    root_np = root_matrix.get() if hasattr(root_matrix, 'get') else np.asarray(root_matrix)
    
    hierarchy: list[list[Any]] = []
    
    # Store initial tasks as a pool of Ray Object References (Futures)
    # The .remote() call returns a pointer token instantly without blocking your script!
    current_level_futures = [ray_remote_decompose_node.remote(root_np)]
    level = 0

    while current_level_futures:
        logger.info(f"Ray Cluster Engine: Resolving Level {level} concurrently across nodes...")
        
        # 2. Block and gather the results of all concurrent cluster worker nodes simultaneously
        # ray.get() fetches all asynchronously calculated matrix slices over the network bus at once
        resolved_results = ray.get(current_level_futures)
        
        level_components: list[Any] = []
        next_level_pool: list[Any] = []
        
        for paired, submatrices in resolved_results:
            level_components.extend(submatrices)
            next_level_pool.extend(submatrices)
            
        if not level_components:
            break
            
        logger.info(f"Ray Level {level} completed. Discovered and mapped {len(level_components)} sub-components.")
        hierarchy.append(level_components)
        
        # 3. Spawn the next level of asynchronous tasks graphs concurrently across the cluster grid
        from poset_operad.core.predicates import is_non_trivial_poset
        current_level_futures = [
            ray_remote_decompose_node.remote(m) 
            for m in next_level_pool if is_non_trivial_poset(m)
        ]
        level += 1

    return hierarchy


def shutdown_cluster_engine() -> None:
    """Termines the distributed Ray cluster runtime context securely."""
    if ray.is_initialized():
        logger.info("Tearing down distributed Ray core cluster session context safely.")
        ray.shutdown()
