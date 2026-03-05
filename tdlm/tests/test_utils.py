# -*- coding: utf-8 -*-

import unittest
import numpy as np
from tdlm.utils import  unique_permutations
from tdlm.utils import _trans_overlap


class TestUtils(unittest.TestCase):

    def test_uperms(self):
        """test whether new implementation of unique_permutations works as intended"""
        X = np.arange(5)
        np.random.seed(0)

        for _ in range(10):  # repeat 10 times for repeatability
            for i in range(5):
                perms = unique_permutations(X, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(X, perm)<=i
                assert len(perms)<=120

            for i in range(5):
                perms = unique_permutations(X, k=119, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(X, perm)<=i
                assert len(perms)<=120-4+i

            for i in range(5):
                perms = unique_permutations(X, k=40, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(X, perm)<=i
                assert len(perms)==40

            for i in range(5):
                perms = unique_permutations(X, k=54, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(X, perm)<=i
                assert len(perms)==54

            perms = unique_permutations(X, max_true_trans=6)
            assert len(perms)==120

            with self.assertRaises(ValueError):
                perms = unique_permutations(X, k=121)



if __name__=='__main__':
    unittest.main()
