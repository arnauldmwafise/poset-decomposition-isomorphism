"""
Shared pytest fixtures for the poset_operad test suite.

Matrices are defined once here and injected into test functions via
pytest's fixture mechanism, ensuring reproducibility across all test runs.
"""

import numpy as np
import pytest


# ── Canonical test matrices ────────────────────────────────────────────────────

@pytest.fixture
def trivial():
    return np.array([[1]])


@pytest.fixture
def antichain_3():
    return np.eye(3, dtype=int)


@pytest.fixture
def chain_3():
    return np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])


@pytest.fixture
def chain_4():
    return np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ])


@pytest.fixture
def semi_right_4():
    """4×4 semi-right dualizable: column-0 fully saturated, core disconnected."""
    return np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
    ])


@pytest.fixture
def semi_left_4():
    """4×4 semi-left dualizable: last row fully saturated, core disconnected."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 1],
    ])


@pytest.fixture
def isobag1():
    """Two matrices known to be isomorphic to each other."""
    return [
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [1, 1, 1, 1]]),
        np.array([[1, 0, 0, 0],
                  [1, 1, 0, 0],
                  [1, 0, 1, 0],
                  [1, 0, 0, 1]]),
    ]


@pytest.fixture
def isobag2():
    return [
        np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]]),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]]),
        np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1]]),
        np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]]),
    ]


@pytest.fixture
def disconnected_2x2():
    """Identity 2×2 — two isolated nodes."""
    return np.eye(2, dtype=int)
