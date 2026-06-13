# poset-operad

**Structural decomposition and isomorphism verification for Partially Ordered Set (Poset) matrices via the Poset Operad.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/your-org/poset-operad/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/poset-operad/actions)
[![codecov](https://codecov.io/gh/your-org/poset-operad/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/poset-operad)

---

## Overview

`poset-operad` provides a composable Python library for analysing **poset adjacency matrices** and **determining poset isomorphism** through the lens of the **Poset Matrix Decomposition**. The library implements:

| Module | Responsibility |
|---|---|
| `core` | Primitive matrix predicates (trivial, chain, antichain, connectivity) |
| `decomposition` | Boundary-depth extraction, direct-sum splitting, tree decomposition |
| `isomorphism` | Hierarchical, saturation-based and direct-sum isomorphism tests |
| `utils` | Signatures, hashing, depth equivalents, triangular saturation |

---

## Installation

```bash
# From PyPI (once published)
pip install poset-operad

# From source (development)
git clone https://github.com/your-org/poset-operad.git
cd poset-operad
pip install -e ".[dev]"
```

---

## Quick-Start

```python
import numpy as np
from poset_operad.core.predicates import is_partial_semi_equidualizable, is_chain_or_antichain
from poset_operad.isomorphism.hierarchical import verify_poset_isomorphism_hierarchical

M = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1],
])

print(is_partial_semi_equidualizable(M))   # True
print(is_chain_or_antichain(M))            # False

M2 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 1, 1, 1],
])
print(verify_poset_isomorphism_hierarchical(M, M2))  # True
```

---

## Repository Layout

```
poset-operad/
├── poset_operad/               # Main package
│   ├── __init__.py
│   ├── core/                   # Atomic predicates & matrix primitives
│   │   ├── __init__.py
│   │   ├── predicates.py       # is_trivial, is_chain_or_antichain, connectivity
│   │   └── submatrix.py        # get_principal_submatrix, element finders
│   ├── decomposition/          # Structural decomposition algorithms
│   │   ├── __init__.py
│   │   ├── boundary.py         # Depth scanning, core extraction
│   │   ├── direct_sum.py       # Direct-sum component splitting
│   │   └── tree.py             # Recursive decomposition tree builder
│   ├── isomorphism/            # Isomorphism verification algorithms
│   │   ├── __init__.py
│   │   ├── hierarchical.py     # Hierarchical decomposition isomorphism
│   │   ├── direct_sum.py       # Direct-sum based isomorphism
│   │   └── saturation.py       # Saturation-metric based isomorphism
│   └── utils/                  # Stateless helpers & invariants
│       ├── __init__.py
│       ├── signatures.py       # get_signature / get_poset_signature
│       ├── depths.py           # generate_semi_depth_equivalents
│       └── metrics.py          # compute_triangular_saturation_metrics
├── tests/
│   ├── unit/                   # Per-function unit tests
│   └── integration/            # Cross-module scenario tests
├── benchmarks/                 # timeit / pytest-benchmark scripts
├── docs/                       # Sphinx source
├── scripts/                    # Reproducibility helpers (seed, generate, export)
├── .github/workflows/ci.yml    # GitHub Actions CI
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## Running the Test Suite

```bash
pytest                           # all tests + coverage
pytest tests/unit/               # unit tests only
pytest -k "isomorphism"          # filter by keyword
```

---

## Contributing

1. Fork the repo and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`.
3. Write tests **before** implementation (TDD encouraged).
4. Run `ruff check . && mypy poset_operad` before opening a PR.

---

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{mwafise2026poset,
  author       = {Mesinga Mwafise, Arnauld},
  title        = {poset-decomposition-isomorphism: Matrix Decomposition Algorithms for Accelerated Poset Isomorphism},
  year         = {2026},
  version      = {1.0.0},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  url          = {https://github.com}
}

```

---

## License

MIT — see [LICENSE](LICENSE).
