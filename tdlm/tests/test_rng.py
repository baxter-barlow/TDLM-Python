# -*- coding: utf-8 -*-
"""
Test the rng parameter behavior for reproducibility

@author: Simon
"""

import unittest
import numpy as np
import tdlm
from tdlm.utils import unique_permutations


class TestRNGBehavior(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        self.tf = np.roll(np.eye(5), 1, axis=1)
        self.probas = np.random.RandomState(42).rand(100, 5)

    def test_unique_permutations_with_int_seed(self):
        """Test that same integer seed produces same permutations"""
        perm1 = unique_permutations(np.arange(5), k=10, rng=42)
        perm2 = unique_permutations(np.arange(5), k=10, rng=42)

        self.assertTrue(np.array_equal(perm1, perm2),
                       "Same integer seed should produce identical permutations")

    def test_unique_permutations_with_generator(self):
        """Test that Generator object works correctly"""
        rng = np.random.default_rng(123)
        perm1 = unique_permutations(np.arange(5), k=10, rng=rng)

        self.assertEqual(perm1.shape, (10, 5),
                        "Should produce correct shape with Generator")

    def test_unique_permutations_different_seeds(self):
        """Test that different seeds produce different permutations"""
        perm1 = unique_permutations(np.arange(5), k=10, rng=42)
        perm2 = unique_permutations(np.arange(5), k=10, rng=99)

        self.assertFalse(np.array_equal(perm1[1:], perm2[1:]),
                        "Different seeds should produce different permutations")

    def test_compute_1step_with_int_seed(self):
        """Test that compute_1step produces reproducible results with int seed"""
        sf1, sb1 = tdlm.compute_1step(self.probas, self.tf, n_shuf=10,
                                       max_lag=10, rng=42)
        sf2, sb2 = tdlm.compute_1step(self.probas, self.tf, n_shuf=10,
                                       max_lag=10, rng=42)

        self.assertTrue(np.allclose(sf1, sf2, equal_nan=True),
                       "Same integer seed should produce identical forward results")
        self.assertTrue(np.allclose(sb1, sb2, equal_nan=True),
                       "Same integer seed should produce identical backward results")

    def test_compute_1step_with_generator(self):
        """Test that compute_1step works with Generator object"""
        rng = np.random.default_rng(123)
        sf, sb = tdlm.compute_1step(self.probas, self.tf, n_shuf=10,
                                     max_lag=10, rng=rng)

        self.assertEqual(sf.shape, (10, 11),
                        "Should produce correct shape with Generator")

    def test_compute_1step_different_seeds(self):
        """Test that different seeds produce different results"""
        sf1, sb1 = tdlm.compute_1step(self.probas, self.tf, n_shuf=10,
                                       max_lag=10, rng=42)
        sf2, sb2 = tdlm.compute_1step(self.probas, self.tf, n_shuf=10,
                                       max_lag=10, rng=99)

        # Results should be different (except possibly NaN values)
        # Check shuffled versions (rows 1+), row 0 is unshuffled so should be identical
        self.assertTrue(np.allclose(sf1[0], sf2[0], equal_nan=True),
                       "Unshuffled result (row 0) should be identical")
        self.assertFalse(np.allclose(sf1[1:], sf2[1:], equal_nan=True),
                        "Different seeds should produce different shuffled results")

    def test_compute_1step_with_none(self):
        """Test that rng=None produces non-deterministic results"""
        sf1, sb1 = tdlm.compute_1step(self.probas, self.tf, n_shuf=10,
                                       max_lag=10, rng=None)
        sf2, sb2 = tdlm.compute_1step(self.probas, self.tf, n_shuf=10,
                                       max_lag=10, rng=None)

        # Results should likely be different (with high probability)
        # Check shuffled versions (rows 1+)
        self.assertTrue(np.allclose(sf1[0], sf2[0], equal_nan=True),
                       "Unshuffled result (row 0) should be identical even with rng=None")
        # This test might occasionally fail due to random chance, but very unlikely
        self.assertFalse(np.allclose(sf1[1:], sf2[1:], equal_nan=True),
                        "rng=None should produce different results each time")

    def test_compute_2step_with_int_seed(self):
        """Test that compute_2step produces reproducible results with int seed"""
        sf1, sb1 = tdlm.compute_2step(self.probas, self.tf, n_shuf=10,
                                       max_lag=10, rng=42)
        sf2, sb2 = tdlm.compute_2step(self.probas, self.tf, n_shuf=10,
                                       max_lag=10, rng=42)

        self.assertTrue(np.allclose(sf1, sf2, equal_nan=True),
                       "Same integer seed should produce identical forward results")
        self.assertTrue(np.allclose(sb1, sb2, equal_nan=True),
                       "Same integer seed should produce identical backward results")

    def test_compute_2step_with_generator(self):
        """Test that compute_2step works with Generator object"""
        rng = np.random.default_rng(456)
        sf, sb = tdlm.compute_2step(self.probas, self.tf, n_shuf=10,
                                     max_lag=10, rng=rng)

        self.assertEqual(sf.shape, (10, 11),
                        "Should produce correct shape with Generator")

    def test_sequenceness_crosscorr_with_int_seed(self):
        """Cross-correlation mode should be reproducible with an integer seed."""
        sf1, sb1 = tdlm.sequenceness_crosscorr(self.probas, self.tf, n_shuf=10, max_lag=10, rng=42)
        sf2, sb2 = tdlm.sequenceness_crosscorr(self.probas, self.tf, n_shuf=10, max_lag=10, rng=42)
        self.assertTrue(np.allclose(sf1, sf2, equal_nan=True))
        self.assertTrue(np.allclose(sb1, sb2, equal_nan=True))

    def test_cross_correlation_alias_rng_forwarding(self):
        """cross_correlation alias should honor the rng argument."""
        sf1, sb1 = tdlm.cross_correlation(self.probas, self.tf, n_shuf=10, max_lag=10, rng=123)
        sf2, sb2 = tdlm.cross_correlation(self.probas, self.tf, n_shuf=10, max_lag=10, rng=123)
        self.assertTrue(np.allclose(sf1, sf2, equal_nan=True))
        self.assertTrue(np.allclose(sb1, sb2, equal_nan=True))

    def test_sequenceness_crosscorr_different_seeds(self):
        """Different seeds should alter shuffled cross-correlation rows."""
        sf1, _ = tdlm.sequenceness_crosscorr(self.probas, self.tf, n_shuf=10, max_lag=10, rng=5)
        sf2, _ = tdlm.sequenceness_crosscorr(self.probas, self.tf, n_shuf=10, max_lag=10, rng=6)
        self.assertTrue(np.allclose(sf1[0], sf2[0], equal_nan=True))
        self.assertFalse(np.allclose(sf1[1:], sf2[1:], equal_nan=True))

    def test_sequenceness_crosscorr_zero_shuffles_keeps_unshuffled_row(self):
        """n_shuf=0 should still return the unshuffled baseline permutation."""
        sf, sb = tdlm.sequenceness_crosscorr(self.probas, self.tf, n_shuf=0, max_lag=10, rng=7)
        self.assertEqual(sf.shape, (1, 11))
        self.assertEqual(sb.shape, (1, 11))

    def test_signflip_test_with_int_seed(self):
        """Test that signflip_test produces reproducible results"""
        # Create test data with multiple subjects
        sx = np.random.RandomState(42).randn(20, 10) + 0.1  # 20 subjects, 10 lags

        result1 = tdlm.signflip_test(sx, n_perms=100, rng=42)
        result2 = tdlm.signflip_test(sx, n_perms=100, rng=42)

        self.assertEqual(result1.pvalue, result2.pvalue,
                        "Same seed should produce identical p-values")
        self.assertEqual(result1.t_obs, result2.t_obs,
                        "Same seed should produce identical t_obs")
        self.assertTrue(np.allclose(result1.t_perms, result2.t_perms),
                       "Same seed should produce identical permutation distribution")


if __name__ == '__main__':
    unittest.main()
