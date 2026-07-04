import logging
import sys

# Configure a centralized project-wide logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d): %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("poset_operad")

# Dynamic hardware router: checks for CUDA, falls back to NumPy if missing
try:
    import cupy as cp
    # Micro-allocation check to confirm the GPU is actually responsive
    _ = cp.zeros(1)
    GPU_AVAILABLE = True
    xp = cp  # xp will act as CuPy
    logger.info("NVIDIA CUDA environment detected. Initialized high-speed CuPy VRAM backend.")
except Exception:
    import numpy as np
    GPU_AVAILABLE = False
    xp = np  # xp will act as NumPy
    logger.info("CUDA not found or unresponsive. Falling back to multi-threaded CPU NumPy backend.")


def get_connected_components_count(M_core) -> int:
    """Calculates graph connectivity components using the active hardware backend."""
    if GPU_AVAILABLE:
        import cupy as cp
        
        n = M_core.shape[0]
        if n == 0: return 0
        if n == 1: return 1
        
        # Build undirected relation graph directly inside VRAM
        M_undirected = ((M_core != 0) | (M_core.T != 0)).astype(cp.int32)
        cp.fill_diagonal(M_undirected, 1)
        
        # Initialize parallel pointer-jumping variables
        labels = cp.arange(n, dtype=cp.int32)
        old_labels = cp.zeros(n, dtype=cp.int32)
        edges = cp.argwhere(M_undirected > 0)
        src, dst = edges[:, 0], edges[:, 1]
        
        # Parallel tree relaxation loop
        while not cp.all(labels == old_labels):
            old_labels = labels.copy()
            cp.minimum.at(labels, src, old_labels[dst])
            cp.minimum.at(labels, dst, old_labels[src])
            labels = labels[labels]
            
        return int(cp.unique(labels).size)
    else:
        # High-velocity compiled C-graph fallback for standard CPUs
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        
        # FIX: Correctly unpack the tuple, ignoring the label array layout
        n_components, _ = connected_components(csr_matrix(M_core), directed=False)
        return int(n_components)
