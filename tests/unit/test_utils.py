"""Unit tests for poset_operad.utils.*"""

import numpy as np
import pytest
from poset_operad.utils.signatures import get_signature, get_poset_signature
from poset_operad.utils.depths import generate_semi_depth_equivalents
from poset_operad.utils.metrics import compute_triangular_saturation_metrics
from poset_operad.utils.equality import are_poset_structures_strictly_equal


class TestSignatures:
    def test_empty_matrix(self):
        sig = get_signature(np.empty((0, 0), dtype=int))
        assert isinstance(sig, int)

    def test_same_matrix_same_sig(self, chain_3):
        assert get_signature(chain_3) == get_signature(chain_3)

    def test_different_matrices_different_sig(self, chain_3, antichain_3):
        # Very likely to differ; both have same shape but different sums
        assert get_signature(chain_3) != get_signature(antichain_3)

    def test_alias(self, chain_3):
        assert get_signature(chain_3) == get_poset_signature(chain_3)


class TestDepthEquivalents:
    def test_symmetric_pair(self):
        equivs = generate_semi_depth_equivalents((2, 0))
        assert (0, 2) in equivs
        assert (1, 1) in equivs  # half-split

    def test_equal_depths_expansion(self):
        equivs = generate_semi_depth_equivalents((3, 3))
        assert (6, 0) in equivs
        assert (0, 6) in equivs

    def test_zero_zero_transpose(self):
        equivs = generate_semi_depth_equivalents((0, 0))
        assert (0, 0) in equivs

    def test_uniqueness(self):
        equivs = generate_semi_depth_equivalents((2, 2))
        assert len(equivs) == len(set(equivs))


class TestTriangularSaturationMetrics:
    def test_full_lower_tri(self, chain_3):
        rows, cols = compute_triangular_saturation_metrics(chain_3)
        assert rows == 3
        assert cols == 3

    def test_empty(self):
        assert compute_triangular_saturation_metrics(np.empty((0, 0), dtype=int)) == (0, 0)

    def test_identity(self, antichain_3):
        rows, cols = compute_triangular_saturation_metrics(antichain_3)
        # Only diagonal entries are 1; row 0 is trivially satisfied (nothing required below diagonal)
        assert isinstance(rows, int) and isinstance(cols, int)


class TestArePosetStructuresStrictlyEqual:
    def test_equal_matrices(self, chain_3):
        assert are_poset_structures_strictly_equal(chain_3, chain_3.copy()) is True

    def test_different_matrices(self, chain_3, antichain_3):
        assert are_poset_structures_strictly_equal(chain_3, antichain_3) is False

    def test_equal_lists(self, chain_3, antichain_3):
        a = [chain_3, antichain_3]
        b = [antichain_3, chain_3]   # different order — multiset comparison
        assert are_poset_structures_strictly_equal(a, b) is True

    def test_none_handling(self):
        assert are_poset_structures_strictly_equal(None, None) is True
        assert are_poset_structures_strictly_equal(None, []) is False

    def test_type_mismatch(self, chain_3):
        assert are_poset_structures_strictly_equal(chain_3, [chain_3]) is False
