# -*- coding: utf-8 -*-
"""
Comprehensive tests for TDLM utils functions

@author: Simon
"""

import unittest
import numpy as np
import pandas as pd
from tdlm.utils import (
    hash_array,
    char2num,
    num2char,
    tf2seq,
    seq2tf,
    seq2TF_2step,
    _trans_overlap,
    unique_permutations
)


class TestHashArray(unittest.TestCase):
    """Tests for hash_array function"""

    def test_hash_array_deterministic(self):
        """Test that hash_array produces consistent results"""
        arr = np.array([1, 2, 3, 4, 5])
        hash1 = hash_array(arr)
        hash2 = hash_array(arr)
        self.assertEqual(hash1, hash2)

    def test_hash_array_different_for_different_arrays(self):
        """Test that different arrays produce different hashes"""
        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([1, 2, 3, 4, 6])
        hash1 = hash_array(arr1)
        hash2 = hash_array(arr2)
        self.assertNotEqual(hash1, hash2)

    def test_hash_array_truncate(self):
        """Test that truncate parameter works"""
        arr = np.array([1, 2, 3])
        hash_short = hash_array(arr, truncate=4)
        hash_long = hash_array(arr, truncate=16)
        self.assertEqual(len(hash_short), 4)
        self.assertEqual(len(hash_long), 16)
        # Long hash should start with short hash
        self.assertTrue(hash_long.startswith(hash_short))

    def test_hash_array_multidimensional(self):
        """Test that hash works with multidimensional arrays"""
        arr_1d = np.array([1, 2, 3, 4])
        arr_2d = np.array([[1, 2], [3, 4]])
        # Flattened arrays should produce same hash (C-order)
        hash_1d = hash_array(arr_1d)
        hash_2d = hash_array(arr_2d)
        self.assertEqual(hash_1d, hash_2d)

    def test_hash_array_dtype_matters(self):
        """Test that dtype affects hash computation"""
        arr = np.array([1, 2, 3])
        hash_int64 = hash_array(arr, dtype=np.int64)
        hash_int32 = hash_array(arr, dtype=np.int32)
        # Different dtypes should produce different hashes
        self.assertNotEqual(hash_int64, hash_int32)


class TestChar2Num(unittest.TestCase):
    """Tests for char2num function"""

    def test_char2num_basic(self):
        """Test basic character to number conversion"""
        result = char2num("ABC")
        self.assertEqual(result, [0, 1, 2])

    def test_char2num_full_alphabet(self):
        """Test conversion of full alphabet range"""
        result = char2num("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        expected = list(range(26))
        self.assertEqual(result, expected)

    def test_char2num_lowercase(self):
        """Test that lowercase is converted to uppercase"""
        result = char2num("abc")
        self.assertEqual(result, [0, 1, 2])

    def test_char2num_mixed_case(self):
        """Test mixed case input"""
        result = char2num("AbCdE")
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_char2num_list_input(self):
        """Test that list input works"""
        result = char2num(['A', 'B', 'C'])
        self.assertEqual(result, [0, 1, 2])

    def test_char2num_single_char(self):
        """Test single character"""
        result = char2num("A")
        self.assertEqual(result, [0])
        result = char2num("Z")
        self.assertEqual(result, [25])


class TestNum2Char(unittest.TestCase):
    """Tests for num2char function"""

    def test_num2char_basic(self):
        """Test basic number to character conversion"""
        result = num2char([0, 1, 2])
        expected = np.array(['A', 'B', 'C'])
        np.testing.assert_array_equal(result, expected)

    def test_num2char_single_int(self):
        """Test single integer input"""
        result = num2char(0)
        self.assertEqual(result, 'A')
        result = num2char(25)
        self.assertEqual(result, 'Z')

    def test_num2char_array_input(self):
        """Test numpy array input"""
        arr = np.array([0, 1, 2, 3])
        result = num2char(arr)
        expected = np.array(['A', 'B', 'C', 'D'])
        np.testing.assert_array_equal(result, expected)

    def test_num2char_2d_array(self):
        """Test 2D array preserves shape"""
        arr = np.array([[0, 1], [2, 3]])
        result = num2char(arr)
        expected = np.array([['A', 'B'], ['C', 'D']])
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (2, 2))

    def test_char2num_num2char_roundtrip(self):
        """Test that char2num and num2char are inverses"""
        original = "ABCDEFG"
        nums = char2num(original)
        recovered = ''.join(num2char(nums))
        self.assertEqual(original, recovered)


class TestSeq2TF(unittest.TestCase):
    """Tests for seq2tf function"""

    def test_seq2tf_basic(self):
        """Test basic sequence to transition matrix"""
        tf = seq2tf("ABC")
        expected = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ], dtype=float)
        np.testing.assert_array_equal(tf, expected)

    def test_seq2tf_circular(self):
        """Test circular sequence (wrapping)"""
        tf = seq2tf("ABCA")  # A->B->C->A
        expected = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ], dtype=float)
        np.testing.assert_array_equal(tf, expected)

    def test_seq2tf_with_n_states(self):
        """Test with explicit n_states larger than sequence"""
        tf = seq2tf("AB", n_states=4)
        self.assertEqual(tf.shape, (4, 4))
        # Only A->B transition should exist
        self.assertEqual(tf[0, 1], 1)
        self.assertEqual(tf.sum(), 1)

    def test_seq2tf_skip_states(self):
        """Test sequence that skips states"""
        tf = seq2tf("ACE", n_states=5)
        self.assertEqual(tf.shape, (5, 5))
        # A(0)->C(2) and C(2)->E(4)
        self.assertEqual(tf[0, 2], 1)
        self.assertEqual(tf[2, 4], 1)
        self.assertEqual(tf.sum(), 2)

    def test_seq2tf_single_transition(self):
        """Test two-state sequence"""
        tf = seq2tf("AB")
        expected = np.array([
            [0, 1],
            [0, 0]
        ], dtype=float)
        np.testing.assert_array_equal(tf, expected)


class TestTF2Seq(unittest.TestCase):
    """Tests for tf2seq function"""

    def test_tf2seq_basic(self):
        """Test basic transition matrix to sequence"""
        tf = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        seq = tf2seq(tf)
        self.assertEqual(seq, "ABC")

    def test_tf2seq_circular(self):
        """Test circular transition matrix"""
        tf = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        seq = tf2seq(tf)
        # Should start from A and go A->B->C
        self.assertEqual(seq, "ABC")

    def test_tf2seq_disjoint(self):
        """Test disjoint sequences"""
        tf = np.array([
            [0, 1, 0, 0],  # A->B
            [0, 0, 0, 0],  # B (end)
            [0, 0, 0, 1],  # C->D
            [0, 0, 0, 0]   # D (end)
        ])
        seq = tf2seq(tf)
        # Should return two disjoint sequences
        self.assertIn("AB", seq)
        self.assertIn("CD", seq)
        self.assertIn("_", seq)

    def test_seq2tf_tf2seq_roundtrip(self):
        """Test that seq2tf and tf2seq are inverses for simple sequences"""
        for original_seq in ["ABC", "ABCDE", "ABCDEFGH"]:
            tf = seq2tf(original_seq)
            recovered_seq = tf2seq(tf)
            self.assertEqual(original_seq, recovered_seq)


class TestSeq2TF2Step(unittest.TestCase):
    """Tests for seq2TF_2step function"""

    def test_seq2TF_2step_basic(self):
        """Test basic 2-step transition matrix creation"""
        tf2 = seq2TF_2step("ABCD")
        # Should create AB->C and BC->D transitions
        self.assertIsInstance(tf2, pd.DataFrame)
        # Check that expected transitions exist
        self.assertIn('AB', tf2.index)
        self.assertIn('BC', tf2.index)

    def test_seq2TF_2step_output_structure(self):
        """Test the structure of 2-step transition matrix"""
        tf2 = seq2TF_2step("ABCDE")
        # Triplets: ABC, BCD, CDE
        # So rows should be AB, BC, CD
        self.assertEqual(len(tf2), 3)

    def test_seq2TF_2step_with_n_states(self):
        """Test 2-step with explicit n_states"""
        tf2 = seq2TF_2step("ACE", n_states=6)
        # Should handle skipped states correctly
        self.assertIsInstance(tf2, pd.DataFrame)


class TestTransOverlap(unittest.TestCase):
    """Tests for _trans_overlap function"""

    def test_trans_overlap_identical(self):
        """Test overlap with identical sequences"""
        seq = [0, 1, 2, 3]
        overlap = _trans_overlap(seq1=seq, seq2=seq)
        # All 3 transitions should overlap
        self.assertEqual(overlap, 3)

    def test_trans_overlap_no_overlap(self):
        """Test completely different sequences"""
        seq1 = [0, 1, 2, 3]  # Transitions: 0->1, 1->2, 2->3
        seq2 = [3, 2, 1, 0]  # Transitions: 3->2, 2->1, 1->0
        overlap = _trans_overlap(seq1=seq1, seq2=seq2)
        self.assertEqual(overlap, 0)

    def test_trans_overlap_partial(self):
        """Test partial overlap"""
        seq1 = [0, 1, 2, 3]  # Transitions: 0->1, 1->2, 2->3
        seq2 = [0, 1, 3, 2]  # Transitions: 0->1, 1->3, 3->2
        overlap = _trans_overlap(seq1=seq1, seq2=seq2)
        self.assertEqual(overlap, 1)  # Only 0->1 overlaps

    def test_trans_overlap_with_precomputed(self):
        """Test using precomputed transitions"""
        seq = [0, 1, 2]
        trans = set([(0, 1), (1, 2)])
        overlap = _trans_overlap(trans1=trans, trans2=trans)
        self.assertEqual(overlap, 2)

    def test_trans_overlap_single_transition(self):
        """Test with minimal sequences"""
        seq1 = [0, 1]  # Single transition 0->1
        seq2 = [0, 1]
        overlap = _trans_overlap(seq1=seq1, seq2=seq2)
        self.assertEqual(overlap, 1)


class TestUniquePermutations(unittest.TestCase):
    """Additional tests for unique_permutations function"""

    def test_unique_permutations_all_unique(self):
        """Test that all returned permutations are unique"""
        X = np.arange(5)
        perms = unique_permutations(X, k=50)
        # Convert to tuples for uniqueness check
        perm_tuples = [tuple(p) for p in perms]
        self.assertEqual(len(perm_tuples), len(set(perm_tuples)))

    def test_unique_permutations_first_is_original(self):
        """Test that first permutation is always the original"""
        X = np.array([0, 1, 2, 3, 4])
        for _ in range(5):  # Test multiple times
            perms = unique_permutations(X, k=10)
            np.testing.assert_array_equal(perms[0], X)

    def test_unique_permutations_max_true_trans(self):
        """Test max_true_trans parameter"""
        X = np.arange(4)
        perms = unique_permutations(X, k=10, max_true_trans=0)
        # Verify no permutation (except first) has any overlap
        original_trans = set(zip(X[:-1], X[1:]))
        for perm in perms[1:]:
            overlap = _trans_overlap(seq1=perm, trans2=original_trans)
            self.assertEqual(overlap, 0)

    def test_unique_permutations_reproducible_with_rng(self):
        """Test reproducibility with rng parameter"""
        X = np.arange(5)
        perms1 = unique_permutations(X, k=10, rng=42)
        perms2 = unique_permutations(X, k=10, rng=42)
        np.testing.assert_array_equal(perms1, perms2)

    def test_unique_permutations_different_with_different_rng(self):
        """Test that different seeds give different results"""
        X = np.arange(5)
        perms1 = unique_permutations(X, k=50, rng=42)
        perms2 = unique_permutations(X, k=50, rng=43)
        # First row should be same (original), rest should differ
        np.testing.assert_array_equal(perms1[0], perms2[0])
        # At least some permutations should differ
        self.assertFalse(np.array_equal(perms1, perms2))


if __name__ == '__main__':
    unittest.main()
