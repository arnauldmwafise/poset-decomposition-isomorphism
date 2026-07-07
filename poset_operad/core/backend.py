"""
poset_operad.core.backend
=========================

Dynamic Hardware Switchboard Manager for the Poset Operad.
Automatically balances workloads between CPU and GPU based on input matrix dimensions.
"""

import logging
import sys

# Configure project-wide logging properties
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d): %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("poset_operad")

# 1. Probe the system for a baseline hardware accelerator card
try:
    import cupy as cp
    _ = cp.zeros(1)
    CUDA_HARDWARE_PRESENT = True
except Exception:
    CUDA_HARDWARE_PRESENT = False

import numpy as np

# Establish initial default system pointers pointing to the host CPU
GPU_AVAILABLE = False
xp = np

# Define the absolute mathematical threshold where GPU parallel processing outscales kernel launch overhead
DIMENSION_ACCELERATION_THRESHOLD = 1000


def runtime_hardware_router(matrix) -> tuple[object, bool]:
    """Dynamically yields the optimal backend context array engine based on matrix size.
    
    Routes smaller structures (N < 1000) to CPU registers to bypass kernel launch latency, 
    and larger workloads (N >= 1000) to CUDA cores if available.
    """
    if not CUDA_HARDWARE_PRESENT:
        return np, False
        
    # Extract the first integer index from the matrix shape tuple safely
    n = matrix.shape[0] if hasattr(matrix, 'shape') and len(matrix.shape) > 0 else 0
    
    if n >= DIMENSION_ACCELERATION_THRESHOLD:
        import cupy as cp
        return cp, True
    else:
        return np, False


def get_connected_components_count(M_core) -> int:
    """Calculates graph connectivity components using the size-optimized hardware backend."""
    # Query the live runtime context matrix dimensions
    active_xp, use_gpu = runtime_hardware_router(M_core)
    
    if use_gpu:
        import cupy as cp
        # Ensure the incoming array is cast cleanly into an active VRAM pointer address space
        M_core_device = cp.asarray(M_core)
        n = M_core_device.shape[0]
        if n == 0: return 0
        if n == 1: return 1
        
        M_undirected = ((M_core_device != 0) | (M_core_device.T != 0)).astype(cp.int32)
        cp.fill_diagonal(M_undirected, 1)
        
        labels = cp.arange(n, dtype=cp.int32)
        old_labels = cp.zeros(n, dtype=cp.int32)
        edges = cp.argwhere(M_undirected > 0)
        src, dst = edges[:, 0], edges[:, 1]
        
        while not cp.all(labels == old_labels):
            old_labels = labels.copy()
            cp.minimum.at(labels, src, old_labels[dst])
            cp.minimum.at(labels, dst, old_labels[src])
            labels = labels[labels]
            
        return int(cp.unique(labels).size)
    else:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        import numpy as np
        
        # Pull or read from a standard NumPy array fallback view safely
        M_core_host = np.asarray(M_core)
        if hasattr(M_core_host, 'get'):
            M_core_host = M_core_host.get()
            
        n_components, _ = connected_components(csr_matrix(M_core_host), directed=False)
        return int(n_components)
