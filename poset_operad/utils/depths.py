"""
poset_operad.utils.depths
============================

Generates equivalent depth-pair configurations for boundary comparison.
"""

from __future__ import annotations


def generate_semi_depth_equivalents(depths: tuple[int, int]) -> list[tuple[int, int]]:
    """Return all depth pairs that yield the same structural state as *depths*.

    Applies geometric transformations to the depth pair to explore the
    symmetry group of the equidualizable property:

    * Always includes the transpose ``(d2, d1)``.
    * If one depth is zero: includes the half-split ``(val//2, val//2)``
      when the other value is even.
    * If depths are equal: includes the expansions ``(2·d1, 0)`` and
      ``(0, 2·d2)``.

    Parameters
    ----------
    depths:
        ``(d1, d2)`` boundary depth pair.

    Returns
    -------
    list[tuple[int, int]]
        Unique list of equivalent depth pairs.

    Complexity
    ----------
    Time O(1), Space O(1).

    Examples
    --------
    >>> generate_semi_depth_equivalents((2, 0))
    [(2, 0), (0, 2), (1, 1)]
    >>> generate_semi_depth_equivalents((3, 3))
    [(3, 3), (6, 0), (0, 6)]
    """
    d1, d2 = depths
    equivs: list[tuple[int, int]] = [(d1, d2), (d2, d1)]

    if d1 == 0 or d2 == 0:
        val = max(d1, d2)
        if val % 2 == 0:
            equivs.append((val // 2, val // 2))
    elif d1 == d2:
        equivs.extend([(d1 * 2, 0), (0, d2 * 2)])

    return list(set(equivs))
