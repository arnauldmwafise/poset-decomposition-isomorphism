"""
poset_operad.utils.collections
================================

Functional helpers for filtering and counting matrices within nested
collections based on a predicate.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def get_satisfying_posets(
    collection: list | tuple | NDArray[np.int_],
    predicate_func: Callable[[NDArray[np.int_]], bool] | str,
) -> list[NDArray[np.int_]]:
    """Return all matrices in *collection* that satisfy *predicate_func*.

    Performs a depth-first recursive traversal.  Leaf nodes are cast to
    ``np.ndarray`` and evaluated; lists/tuples are traversed recursively.

    Parameters
    ----------
    collection:
        Arbitrarily nested structure of potential poset matrices.
    predicate_func:
        A callable ``(np.ndarray) → bool``, or a string name of such a
        function in the global namespace (deprecated — prefer a direct
        callable).

    Returns
    -------
    list[np.ndarray]
        Flattened list of all matrices that returned ``True``.

    Complexity
    ----------
    Time O(M · T), Space O(D + S · n²).
    """
    if isinstance(predicate_func, str):
        import importlib
        mod = importlib.import_module("poset_operad")
        predicate: Callable[[NDArray[np.int_]], bool] = getattr(mod, predicate_func)
    else:
        predicate = predicate_func

    results: list[NDArray[np.int_]] = []

    for item in collection:
        if isinstance(item, (list, tuple)) and not isinstance(item, np.ndarray):
            results.extend(get_satisfying_posets(item, predicate))
        else:
            matrix = np.array(item)
            if predicate(matrix):
                results.append(matrix)

    return results


def count_satisfying_posets(
    collection: list | tuple | NDArray[np.int_],
    predicate_func: Callable[[NDArray[np.int_]], bool] | str,
) -> int:
    """Count matrices in *collection* satisfying *predicate_func* without storing them.

    Memory-efficient alternative to :func:`get_satisfying_posets`; uses a
    scalar accumulator instead of building a result list.

    Parameters
    ----------
    collection:
        Nested structure of potential poset matrices.
    predicate_func:
        Callable or string identifier (see :func:`get_satisfying_posets`).

    Returns
    -------
    int

    Complexity
    ----------
    Time O(M · T), Space O(L + n²).
    """
    if isinstance(predicate_func, str):
        import importlib
        mod = importlib.import_module("poset_operad")
        predicate: Callable[[NDArray[np.int_]], bool] = getattr(mod, predicate_func)
    else:
        predicate = predicate_func

    total = 0
    for item in collection:
        if isinstance(item, (list, tuple)) and not isinstance(item, np.ndarray):
            total += count_satisfying_posets(item, predicate)
        else:
            if predicate(np.array(item)):
                total += 1
    return total
