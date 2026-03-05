# -*- coding: utf-8 -*-

import unittest

import numpy as np

from tdlm.utils import _trans_overlap
from tdlm.utils import unique_permutations


class TestUtils(unittest.TestCase):
    def test_uperms(self):
        """Test whether unique_permutations works as intended."""
        x = np.arange(5)
        np.random.seed(0)

        for _ in range(10):  # repeat 10 times for repeatability
            for i in range(5):
                perms = unique_permutations(x, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(x, perm) <= i
                assert len(perms) <= 120

            for i in range(5):
                perms = unique_permutations(x, k=119, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(x, perm) <= i
                assert len(perms) <= 120 - 4 + i

            for i in range(5):
                perms = unique_permutations(x, k=40, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(x, perm) <= i
                assert len(perms) == 40

            for i in range(5):
                perms = unique_permutations(x, k=54, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(x, perm) <= i
                assert len(perms) == 54

            perms = unique_permutations(x, max_true_trans=6)
            assert len(perms) == 120

            with self.assertRaises(ValueError):
                unique_permutations(x, k=121)
