"""
poset_operad.utils.equality
==============================

Structural equality comparison for nested collections of poset matrices.

``are_poset_structures_strictly_equal`` checks **isomorphism** in the sense of
identical nesting, array shapes, and value multisets — it does **not** check
graph isomorphism.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import NDArray


def are_poset_structures_strictly_equal(
    list_a: list | NDArray[np.int_] | None,
    list_b: list | NDArray[np.int_] | None,
) -> bool:
    """Verify structural and value-level identity between two nested poset collections.

    Parameters
    ----------
    list_a, list_b:
        Nested lists / ndarrays to compare.

    Returns
    -------
    bool
        ``True`` if both structures have the same nesting, shapes, and element
        multisets at every level.

    Algorithm
    ---------
    * **Base case (ndarray)** – compare shape and element multiset (via sorted
      flattened array).
    * **Recursive case (list)** – collect item signatures, compare as multisets.

    Complexity
    ----------
    Time O(K · N² log N²), Space O(K · N²).
    """
    if list_a is None or list_b is None:
        return list_a == list_b
    if type(list_a) is not type(list_b):
        return False

    if isinstance(list_a, np.ndarray):
        assert isinstance(list_b, np.ndarray)
        if list_a.shape != list_b.shape:
            return False
        return np.array_equal(
            np.sort(list_a, axis=None), np.sort(list_b, axis=None)
        )

    if not isinstance(list_a, list) or not isinstance(list_b, list):
        return list_a == list_b

    if len(list_a) != len(list_b):
        return False

    def _signature(item: object) -> object:
        if isinstance(item, np.ndarray):
            return (item.shape, np.sort(item, axis=None).tobytes())
        if isinstance(item, list):
            return tuple(
                sorted((_signature(sub) for sub in item), key=lambda x: str(x))
            )
        return hash(item)

    return Counter(_signature(x) for x in list_a) == Counter(
        _signature(x) for x in list_b
    )
