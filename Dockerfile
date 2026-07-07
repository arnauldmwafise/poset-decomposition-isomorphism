# ==========================================
# Stage 1: Core CUDA Environment & System Deps
# ==========================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by SciPy and NumPy on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# Stage 2: Install Python Package Ecosystem
# ==========================================
WORKDIR /app

# Upgrade pip and install core high-performance ecosystem wheels
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install CuPy pre-compiled for CUDA 12.x alongside core pipeline modules
RUN python3 -m pip install --no-cache-dir \
    cupy-cuda12x \
    numpy \
    scipy \
    pytest \
    pytest-cov \
    networkx

# ==========================================
# Stage 3: Source Code Ingestion & Mounting
# ==========================================
# Copy the full project layout into the operational working directory
COPY . /app

# Expose the PYTHONPATH variable globally inside the container context
# This entirely replaces the unstable editable installation step
ENV PYTHONPATH="/app"

# Default terminal directive executes your multi-tier test runner verification
CMD ["pytest", "tests/", "-v"]
