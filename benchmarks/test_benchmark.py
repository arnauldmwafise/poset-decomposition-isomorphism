import time
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from poset_operad.core.predicates import is_trivial_poset

def test_matrix_decomposition_speed():
    """Simple execution benchmark for processing a large poset matrix."""
    # Generate a large 500x500 mock poset matrix
    large_matrix = np.eye(500)
    
    start_time = time.perf_counter()
    
    # Run the core predicate logic
    result = is_trivial_poset(large_matrix)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print(f"\n🚀 Matrix Decomposition Execution Time: {execution_time:.6f} seconds")
    assert execution_time < 1.0  # Performance threshold: must run under 1 second
