"""
Shared pytest fixtures for the poset_operad test suite.

Matrices are defined once here and injected into test functions via
pytest's fixture mechanism, ensuring reproducibility across all test runs.
"""

from __future__ import annotations

import io
from typing import Any, List
import pytest

from poset_operad.core.backend import xp, GPU_AVAILABLE, logger


def pytest_configure(config: Any) -> None:
    """Logs system telemetry data before running any math validations."""
    logger.info("=" * 60)
    logger.info(f"RUNNING TEST SUITE ON HARDWARE ENGINE: {xp.__name__.upper()}")
    logger.info(f"GPU COMPUTATION HARDWARE DETECTED: {GPU_AVAILABLE}")
    logger.info("=" * 60)


def _load_matrix(matrix_str: str) -> Any:
    """Safely loads a matrix from a plaintext string bypassing markdown strip bugs."""
    lines = [line.strip() for line in matrix_str.strip().split("\n") if line.strip()]
    grid = [[int(val) for val in line.split()] for line in lines]
    return xp.asarray(grid, dtype=xp.int32)


# ── Canonical test matrices ────────────────────────────────────────────────────

@pytest.fixture
def trivial() -> Any:
    # Fixed to load as a 2D 1x1 matrix element [[1]]
    return _load_matrix("1")


@pytest.fixture
def antichain_3() -> Any:
    return _load_matrix("""
        1 0 0
        0 1 0
        0 0 1
    """)


@pytest.fixture
def chain_3() -> Any:
    return _load_matrix("""
        1 0 0
        1 1 0
        1 1 1
    """)


@pytest.fixture
def chain_4() -> Any:
    return _load_matrix("""
        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1
    """)


@pytest.fixture
def semi_right_4() -> Any:
    """4×4 semi-right dualizable: column-0 fully saturated, core disconnected."""
    return _load_matrix("""
        1 0 0 0
        1 1 0 0
        1 0 1 0
        1 0 0 1
    """)


@pytest.fixture
def semi_left_4() -> Any:
    """4×4 semi-left dualizable: last row fully saturated, core disconnected."""
    return _load_matrix("""
        1 0 0 0
        0 1 0 0
        0 0 1 0
        1 1 1 1
    """)


@pytest.fixture
def isobag1() -> List[Any]:
    """Two matrices known to be isomorphic to each other."""
    return [
        _load_matrix("""
            1 0 0 0
            0 1 0 0
            0 0 1 0
            1 1 1 1
        """),
        _load_matrix("""
            1 0 0 0
            1 1 0 0
            1 0 1 0
            1 0 0 1
        """)
    ]


@pytest.fixture
def isobag2() -> List[Any]:
    return [
        _load_matrix("1 0 0 0\n1 1 0 0\n1 0 1 0\n1 0 1 1"),
        _load_matrix("1 0 0 0\n0 1 0 0\n0 1 1 0\n1 1 1 1"),
        _load_matrix("1 0 0 0\n1 1 0 0\n1 1 1 0\n1 0 0 1"),
        _load_matrix("1 0 0 0\n1 1 0 0\n0 0 1 0\n1 1 1 1")
    ]


@pytest.fixture
def disconnected_2x2() -> Any:
    """Identity 2×2 — two isolated nodes."""
    return _load_matrix("""
        1 0
        0 1
    """)
