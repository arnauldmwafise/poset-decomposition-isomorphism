"""
poset_operad.utils.collections
================================

Functional helpers for filtering and counting matrices within nested
collections based on a predicate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, List, Union

from poset_operad.core.backend import xp, logger


def get_satisfying_posets(
    collection: list | tuple | Any,
    predicate_func: Callable[[Any], bool] | str,
) -> list[Any]:
    """Return all matrices in *collection* that satisfy *predicate_func*.

    Performs a depth-first recursive traversal.  Leaf nodes are cast to
    the active device matrix type and evaluated; lists/tuples are traversed recursively.

    Parameters
    ----------
    collection:
        Arbitrarily nested structure of potential poset matrices.
    predicate_func:
        A callable ``(ndarray) → bool``, or a string name of such a
        function in the core predicate namespace.

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
        # Safely resolve dynamic lookups against the core predicates module
        mod = importlib.import_module("poset_operad.core.predicates")
        predicate: Callable[[Any], bool] = getattr(mod, predicate_func)
    else:
        predicate = predicate_func

    results: list[Any] = []

    for item in collection:
        if isinstance(item, (list, tuple)) and not isinstance(item, xp.ndarray):
            results.extend(get_satisfying_posets(item, predicate))
        else:
            # Map leaf nodes safely to the active memory hardware device layer
            matrix = xp.asarray(item)
            if predicate(matrix):
                results.append(matrix)

    return results


def count_satisfying_posets(
    collection: list | tuple | Any,
    predicate_func: Callable[[Any], bool] | str,
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
        mod = importlib.import_module("poset_operad.core.predicates")
        predicate: Callable[[Any], bool] = getattr(mod, predicate_func)
    else:
        predicate = predicate_func

    total = 0
    for item in collection:
        if isinstance(item, (list, tuple)) and not isinstance(item, xp.ndarray):
            total += count_satisfying_posets(item, predicate)
        else:
            if predicate(xp.asarray(item)):
                total += 1
                
    logger.debug(f"Sieved subset. Accumulated matches for criteria: {total}")
    return total
