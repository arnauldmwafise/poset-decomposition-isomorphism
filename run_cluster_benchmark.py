"""
Distributed Local Cluster Client Orchestration Script.

Tests the Ray multi-node engine over a robust sequence of canonical HPMT test items:
1. A 14x14 3-Level Hierarchical Poset Matrix Tree (HPMT)
2. A 10x10 2-Level Hierarchical Poset Matrix Tree (HPMT)
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import ray

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from poset_operad.decomposition.tree_ray import (
    build_poset_decomposition_tree_distributed,
    shutdown_cluster_engine,
)


def load_canonical_14x14_poset() -> np.ndarray:
    """Parses your exact 14x14 3-level LaTeX poset matrix from a plaintext string."""
    matrix_str = """
    1 0 0 0 0 0 0 0 0 0 0 0 0 0
    1 1 0 0 0 0 0 0 0 0 0 0 0 0
    1 1 1 0 0 0 0 0 0 0 0 0 0 0
    1 0 0 1 0 0 0 0 0 0 0 0 0 0
    1 0 0 1 1 0 0 0 0 0 0 0 0 0
    1 0 0 1 0 1 0 0 0 0 0 0 0 0
    1 0 0 0 0 0 1 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 1 0 0 0 0 0 0
    1 0 0 0 0 0 1 1 1 0 0 0 0 0
    1 0 0 0 0 0 0 0 0 1 0 0 0 0
    1 0 0 0 0 0 0 0 0 1 1 0 0 0
    1 0 0 0 0 0 0 0 0 1 0 1 0 0
    1 0 0 0 0 0 0 0 0 1 0 1 1 0
    1 0 0 0 0 0 0 0 0 1 0 1 0 1
    """
    lines = [line.strip() for line in matrix_str.strip().split("\n") if line.strip()]
    grid = [[int(val) for val in line.split()] for line in lines]
    return np.array(grid, dtype=np.int32)


def load_canonical_10x10_poset() -> np.ndarray:
    """Parses your new 10x10 2-level LaTeX poset matrix from a plaintext string."""
    matrix_str = """
    1 0 0 0 0 0 0 0 0 0
    1 1 0 0 0 0 0 0 0 0
    1 1 1 0 0 0 0 0 0 0
    1 1 0 1 0 0 0 0 0 0
    1 0 0 0 1 0 0 0 0 0
    1 0 0 0 1 1 0 0 0 0
    1 0 0 0 1 0 1 0 0 0
    1 0 0 0 0 0 0 1 0 0
    1 0 0 0 0 0 0 1 1 0
    1 0 0 0 0 0 0 1 0 1
    """
    lines = [line.strip() for line in matrix_str.strip().split("\n") if line.strip()]
    grid = [[int(val) for val in line.split()] for line in lines]
    return np.array(grid, dtype=np.int32)


def execute_cluster_orchestration():
    print("=" * 75)
    print("🛰️  INITIALIZING DISTRIBUTED MULTI-NODE WORKLOAD BATCH SPRINT")
    print("=" * 75)
    
    if not ray.is_initialized():
        ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
        
    print(f"✅ Distributed Ray compute mesh context established successfully!")
    print("-" * 75)

    # Define our robust sequential evaluation blueprint items
    test_pipeline = [
        {
            "name": "Canonical 14x14 3-Level HPMT Matrix",
            "matrix": load_canonical_14x14_poset(),
            "expected_depth": 3,
            "expected_layer_counts": {0: 4, 1: 6, 2: 2}
        },
        {
            "name": "Canonical 10x10 2-Level HPMT Matrix",
            "matrix": load_canonical_10x10_poset(),
            "expected_depth": 2,
            "expected_layer_counts": {0: 3, 1: 6}
        }
    ]

    for item_idx, item in enumerate(test_pipeline):
        print(f"\n▶️  [WORKLOAD ITEM {item_idx + 1}/{len(test_pipeline)}]: {item['name']}")
        print(f"   Input Tensor Dimensions: {item['matrix'].shape}")
        
        start_time = time.perf_counter()
        
        # Fire asynchronous tasks concurrently across the worker grid
        decomposition_tree = build_poset_decomposition_tree_distributed(item['matrix'])
        
        elapsed_time = time.perf_counter() - start_time
        actual_depth = len(decomposition_tree)
        
        print(f"   🏁 Processing finished in: {elapsed_time:.4f} seconds")
        print(f"   📈 Discovered Tree Depth Layers: {actual_depth}")
        
        # Print breakdown and inspect components dynamically
        for lvl_idx, level in enumerate(decomposition_tree):
            print(f"      👉 Layer {lvl_idx}: Discovered {len(level)} component sub-matrices")
            for c_idx, comp in enumerate(level):
                print(f"         - Component {c_idx} Dimension: {comp.shape[0]}x{comp.shape[1]}")
        
        # Robust Assertion Filters to secure math validity rules
        assert actual_depth == item['expected_depth'], \
            f"🚨 Mathematical mismatch: Expected tree depth {item['expected_depth']}, but got {actual_depth}."
            
        for lvl_idx, expected_count in item['expected_layer_counts'].items():
            actual_count = len(decomposition_tree[lvl_idx])
            assert actual_count == expected_count, \
                f"🚨 Structural mismatch on Layer {lvl_idx}: Expected {expected_count} components, but found {actual_count}."
                
        print(f"   ✅ [VALIDATION SUCCESS]: Structural components align perfectly with LaTeX rules.")
        print("-" * 75)

    print("\n🏆 ALL BATCH WORKLOAD RUNS PASSED PERFECTLY.")
    print("=" * 75)
    
    shutdown_cluster_engine()


if __name__ == "__main__":
    execute_cluster_orchestration()
