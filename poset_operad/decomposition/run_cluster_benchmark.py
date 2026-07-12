"""
Distributed Local Cluster Client Orchestration Script.

Spins up a multi-node background Ray cluster topology on your machine,
parallelizes recursive poset decomposition tree branches across virtual server nodes,
and displays active cluster processing metrics.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

# Force register the local path directory context into the Python namespace
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import ray
except ImportError:
    print("❌ Ray is missing! Please install it by running: pip install ray")
    sys.exit(1)

from poset_operad.core.backend import logger
from poset_operad.decomposition.tree_ray import (
    build_poset_decomposition_tree_distributed,
    shutdown_cluster_engine,
)


def generate_massive_cluster_matrix(size: int) -> np.ndarray:
    """Synthesizes a dense, reflexive lower-triangular poset for cluster workloads."""
    print(f"📦 Synthesizing {size}x{size} master poset data structure...")
    mat = np.eye(size, dtype=np.int32)
    d = max(1, size // 10)
    lower_tri_mask = np.tril(np.ones((size, size), dtype=bool), k=-1)
    
    # Saturate nested boundary limits
    mat[d:, :d] = 1
    mat[-d:, :] = 1
    
    # Inject complex inner core blocks to force deep recursive branch splits
    core_size = size - (2 * d)
    if core_size > 2:
        rand_block = np.random.rand(core_size, core_size) > 0.45
        core_lower_mask = np.tril(np.ones((core_size, core_size), dtype=bool), k=-1)
        core_mat = np.where(core_lower_mask & rand_block, 1, np.eye(core_size, dtype=np.int32))
        mat[d:size-d, d:size-d] = core_mat
        
    mat = np.where(lower_tri_mask | np.eye(size, dtype=bool), mat, 0)
    
    # Fast boolean transitive closure matrix multiplication step
    for _ in range(size.bit_length()):
        mat_next = (mat @ mat) > 0
        mat_next = np.where(lower_tri_mask | np.eye(size, dtype=bool), mat_next.astype(np.int32), 0)
        if np.array_equal(mat, mat_next):
            break
        mat = mat_next
        
    return mat


def execute_cluster_orchestration():
    print("=" * 70)
    print("🛰️ INITIALIZING VIRTUAL MULTI-NODE COMPUTE CLUSTER ENGINE")
    print("=" * 70)
    
    # 1. Boot up a local virtual Ray cluster grid with 4 simulated independent server nodes
    if not ray.is_initialized():
        ray.init(
            num_cpus=4,  # Simulates a cluster environment with 4 individual processing workers
            include_dashboard=True,  # Activates the local browser telemetry panel
            ignore_reinit_error=True
        )
        
    # Query live environment details directly from the active compute mesh context
    cluster_resources = ray.cluster_resources()
    print(f"✅ Distributed compute mesh connected!")
    print(f"   Available Virtual Cluster Workers/CPUs: {cluster_resources.get('CPU')}")
    print(f"   Ray Local Dashboard Dashboard URL: http://127.0.0.1:8265")
    print("-" * 70)

    # 2. Synthesize a 1500x1500x target matrix size
    target_size = 1500
    poset_matrix = generate_massive_cluster_matrix(target_size)
    
    print("\n🚀 Firing asynchronous tree decomposition task graphs across cluster nodes...")
    start_time = time.perf_counter()
    
    # 3. Call your distributed Ray decomposition tree implementation
    decomposition_tree = build_poset_decomposition_tree_distributed(poset_matrix)
    
    elapsed_time = time.perf_counter() - start_time
    print("-" * 70)
    print("🏁 DISTRIBUTED CLUSTER WORKLOAD COMPLETED SUCCESSFULLY!")
    print(f"   Total Tree Depth Layers Mapped: {len(decomposition_tree)}")
    print(f"   Total Computation Time: {elapsed_time:.4f} seconds")
    print("=" * 70)
    
    # 4. Safely tear down cluster network connections
    shutdown_cluster_engine()


if __name__ == "__main__":
    execute_performance_profiling = execute_cluster_orchestration()
