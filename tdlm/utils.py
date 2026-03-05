# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:10:07 2024

util functions for Temporally Delayed Linear Modelling

@author: simon.kern
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from itertools import permutations
from typing import Any

import numpy as np
import pandas as pd
import warnings
from numpy.typing import ArrayLike, DTypeLike, NDArray


def hash_array(arr: ArrayLike, dtype: DTypeLike = np.int64, truncate: int = 8) -> str:
    """
    create a persistent hash for a numpy array based on the byte representation
    only the last `truncate` (default=8) characters are returned for simplicity

    Parameters
    ----------
    arr : np.ndarray
        DESCRIPTION.
    dtype : type, optional
        which data type to use. smaller type will be faster.
        The default is np.int64.

    Returns
    -------
    str
        unique hash for that array.

    """
    arr_np = np.asarray(arr).astype(dtype)
    sha1_hash = hashlib.sha1(arr_np.flatten("C").tobytes()).hexdigest()
    return sha1_hash[:truncate]


def _trans_overlap(
    seq1: Sequence[int] | None = None,
    seq2: Sequence[int] | None = None,
    trans1: set[tuple[int, int]] | None = None,
    trans2: set[tuple[int, int]] | None = None,
) -> int:
    """
    calculate how many overlapping 1 step transitions exist between
    seq1 and seq2. For optimization reasons instead of the sequence, already
    computed transitions can also be supplied

    """
    if trans1 is None:
        if seq1 is None:
            raise ValueError("seq1 must be provided when trans1 is None")
        trans1 = set(zip(seq1[:-1], seq1[1:]))
    if trans2 is None:
        if seq2 is None:
            raise ValueError("seq2 must be provided when trans2 is None")
        trans2 = set(zip(seq2[:-1], seq2[1:]))
    return len(trans1.intersection(trans2))


def unique_permutations(
    X: ArrayLike,
    k: int | None = None,
    max_true_trans: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """"""
    X = np.array(X).squeeze()
    if X.ndim != 1:
        raise ValueError(f"X must be 1D after squeeze, got shape={X.shape}")
    if len(X) <= 1:
        raise ValueError(f"X must contain at least 2 elements, got len(X)={len(X)}")

    rng = np.random.default_rng(rng)

    uniques = np.unique(X)

    max_perms = math.factorial(len(uniques))

    if k is None:
        k = max_perms  # default to computing all unique permutations

    if k > max_perms:
        raise ValueError(f'requested {k=} larger than all possible permutations {max_perms=}')


    # enumerate all transitions in case max_overlap is set
    trans = set(zip(X[:-1], X[1:]))
    seq = tuple(X.tolist())

    # overlap-constrained mode: build valid pool once, then sample from it
    if max_true_trans is not None:
        valid_perms = [seq]
        for perm in permutations(seq):
            if perm == seq:
                continue
            if _trans_overlap(seq1=perm, trans2=trans) <= max_true_trans:
                valid_perms.append(perm)

        if len(valid_perms) < k:
            warnings.warn(f'Fewer valid permutations {len(valid_perms)=} possible than {k=} requested')
            return np.array(valid_perms)
        if len(valid_perms) == k:
            return np.array(valid_perms)

        idx = rng.choice(np.arange(1, len(valid_perms)), size=k-1, replace=False)
        sampled = [seq] + [valid_perms[i] for i in idx]
        return np.array(sampled)

    # full enumeration is exact and avoids slow random coupon collection
    if k == max_perms:
        perms = [seq]
        for perm in permutations(seq):
            if perm == seq:
                continue
            perms.append(perm)
        return np.array(perms)

    # unconstrained partial sampling mode
    uperms = {seq}
    while len(uperms) < k:
        perm = tuple(rng.permutation(seq))
        if perm in uperms:
            continue
        uperms.add(perm)

    # ensure the original sequence is always first
    uperms.remove(seq)
    all_perms = [seq]
    all_perms += list(uperms)
    return np.array(all_perms)



def _unique_permutations_MATLAB(
    X: ArrayLike, k: int | None = None
) -> tuple[int, NDArray[np.uint32], NDArray[np.int_]]:
    """
    DEPRECATED

    original implementation of the unique permutation function from MATLAB

    #uperms: unique permutations of an input vector or rows of an input matrix
    # Usage:  nPerms              = uperms(X)
    #        [nPerms pInds]       = uperms(X, k)
    #        [nPerms pInds Perms] = uperms(X, k)
    #
    # Determines number of unique permutations (nPerms) for vector or matrix X.
    # Optionally, all permutations' indices (pInds) are returned. If requested,
    # permutations of the original input (Perms) are also returned.
    #
    # If k < nPerms, a random (but still unique) subset of k of permutations is
    # returned. The original/identity permutation will be the first of these.
    #
    # Row or column vector X results in Perms being a [k length(X)] array,
    # consistent with MATLAB's built-in perms. pInds is also [k length(X)].
    #
    # Matrix X results in output Perms being a [size(X, 1) size(X, 2) k]
    # three-dimensional array (this is inconsistent with the vector case above,
    # but is more helpful if one wants to easily access the permuted matrices).
    # pInds is a [k size(X, 1)] array for matrix X.
    #
    # Note that permutations are not guaranteed in any particular order, even
    # if all nPerms of them are requested, though the identity is always first.
    #
    # Other functions can be much faster in the special cases where they apply,
    # as shown in the second block of examples below, which uses perms_m.
    #
    # Examples:
    #  uperms(1:7),       factorial(7)        # verify counts in simple cases,
    #  uperms('aaabbbb'), nchoosek(7, 3)      # or equivalently nchoosek(7, 4).
    #  [n pInds Perms] = uperms('aaabbbb', 5) # 5 of the 35 unique permutations
    #  [n pInds Perms] = uperms(eye(3))       # all 6 3x3 permutation matrices
    #
    #  # A comparison of timings in a special case (i.e. all elements unique)
    #  tic; [nPerms P1] = uperms(1:20, 5000); T1 = toc
    #  tic; N = factorial(20); S = sample_no_repl(N, 5000);
    #  P2 = zeros(5000, 20);
    #  for n = 1:5000, P2(n, :) = perms_m(20, S(n)); end
    #  T2 = toc # quicker (note P1 and P2 are not the same random subsets!)
    #  # For me, on one run, T1 was 7.8 seconds, T2 was 1.3 seconds.
    #
    #  # A more complicated example, related to statistical permutation testing
    #  X = kron(eye(3), ones(4, 1));  # or similar statistical design matrix
    #  [nPerms pInds Xs] = uperms(X, 5000); # unique random permutations of X
    #  # Verify correctness (in this case)
    #  G = nan(12,5000); for n = 1:5000; G(:, n) = Xs(:,:,n)*(1:3)'; end
    #  size(unique(G', 'rows'), 1)    # 5000 as requested.
    #
    # See also: randperm, perms, perms_m, signs_m, nchoosek_m, sample_no_repl
    # and http://www.fmrib.ox.ac.uk/fsl/randomise/index.html#theory

    # Copyright 2010 Ged Ridgway
    # http://www.mathworks.com/matlabcentral/fileexchange/authors/27434
    """
    # Count number of repetitions of each unique row, and get representative x

    X = np.array(X).squeeze()
    if len(X) <= 1:
        raise ValueError(f"X must contain at least 2 elements, got len(X)={len(X)}")

    if X.ndim == 1:
        uniques, _uind, c = np.unique(X, return_index=True, return_counts=True)
    else:
        # [u uind x] = unique(X, 'rows'); % x codes unique rows with integers
        uniques, _uind, c = np.unique(X, axis=0, return_index=True, return_counts=True)

    max_perms = math.factorial(len(uniques))

    if k is None:
        k = max_perms  # default to computing all unique permutations

    if k > max_perms:
        raise ValueError('requested {k=} larger than all possible permutations')

    uniques = uniques.tolist()
    x = np.array([uniques.index(i) for i in X.tolist()])

    c_sorted = sorted(np.asarray(c, dtype=int).tolist())
    nPerms_float = np.prod(np.arange(c_sorted[-1] + 1, np.sum(c_sorted) + 1)) / np.prod(
        [math.factorial(v) for v in c_sorted[:-1]]
    )
    nPerms = int(nPerms_float)
    # % computation of permutation
    # Basics
    n = len(X)


    # % Identity permutation always included first:
    pInds = np.zeros([int(k), n], dtype=np.uint32)
    Perms: NDArray[np.int_] = pInds.copy().astype(int)
    pInds[0, :] = np.arange(0, n)
    Perms[0, :] = x

    # Add permutations that are unique
    u = 0  # to start with
    while u < k - 1:
        pInd = np.random.permutation(int(n))
        pInd = np.array(pInd).astype(int)  # just in case MATLAB permutation was monkey patched
        if x[pInd].tolist() not in Perms.tolist():
            u += 1
            pInds[u, :] = pInd
            Perms[u, :] = x[pInd]
    # %
    # Construct permutations of input
    if X.ndim == 1:
        Perms = np.repeat(np.atleast_2d(X), int(k), 0).astype(int)
        for i in range(1, int(k)):
            Perms[i, :] = X[pInds[i, :]]
    else:
        Perms_3d = np.repeat(np.atleast_3d(X), int(k), axis=2).astype(int)
        for i in range(1, int(k)):
            Perms_3d[:, :, i] = X[pInds[i, :], :]
        Perms = np.asarray(Perms_3d, dtype=int)
    return nPerms, pInds, Perms


def char2num(seq: str | Sequence[str]) -> list[int]:
    """convert list of chars to integers eg ABC=>012"""
    if isinstance(seq, str):
        seq = list(seq)
    nums = [ord(c.upper())-65 for c in seq]
    if not all(0 <= n <= 25 for n in nums):
        raise ValueError(f"seq must only contain letters A-Z, got seq={seq}")
    return nums


def num2char(arr: int | ArrayLike) -> str | NDArray[np.str_]:
    """convert list of ints to alphabetical chars eg 012=>ABC"""
    if isinstance(arr, int):
        return chr(arr+65)
    arr = np.array(arr, dtype=int)
    return np.array([chr(x+65) for x in arr.ravel()]).reshape(*arr.shape)


def tf2seq(transition_matrix: ArrayLike) -> str:
    """
    Convert a transition matrix into a sequence string.
    If there are disjoint sequences, separate them with "_".

    :param transition_matrix: A square numpy array representing the transition matrix.
                              Each row should have at most one outgoing transition (1).
    :return: A string representing the sequence(s), e.g., "ABC_DEF".
    """
    transition_matrix_arr = np.asarray(transition_matrix, dtype=float)
    if transition_matrix_arr.ndim != 2:
        raise ValueError(f"transition_matrix must be 2D, got shape={transition_matrix_arr.shape}")
    if transition_matrix_arr.shape[0] != transition_matrix_arr.shape[1]:
        raise ValueError(f"transition_matrix must be square, got shape={transition_matrix_arr.shape}")

    n_states = transition_matrix_arr.shape[0]
    visited = set()
    sequences = []

    def find_sequence(start_state: int) -> list[int]:
        """Helper function to find a sequence starting from a given state."""
        sequence = []
        current_state = start_state
        while current_state not in visited:
            sequence.append(current_state)
            visited.add(current_state)
            next_state = int(np.argmax(transition_matrix_arr[current_state]))  # Find the next state
            if transition_matrix_arr[current_state, next_state] == 0:  # No valid transition
                break
            current_state = next_state
        return sequence

    # Iterate through all states to find disjoint sequences
    for state in range(n_states):
        if state not in visited and np.sum(transition_matrix_arr[state]) > 0:  # Unvisited and has outgoing transitions
            sequence = find_sequence(state)
            sequences.append(sequence)

    # Convert numeric sequences to character sequences and join with "_"
    sequence_strings = []
    for sequence in sequences:
        sequence_str = ''.join(chr(65 + state) for state in sequence)
        sequence_strings.append(sequence_str)

    return '_'.join(sequence_strings)


def seq2tf(sequence: str | Sequence[str], n_states: int | None = None) -> NDArray[np.float64]:
    """
    create a transition matrix from a sequence string,
    e.g. ABCDEFG
    Please note that sequences will not be wrapping automatically,
    i.e. a wrapping sequence should be denoted by appending the first state.

    :param sequence: sequence in format "ABCD..."
    :param seqlen: if not all states are part of the sequence,
                   the number of states can be specified
                   e.g. if the sequence is ABE, but there are also states F,G
                   n_states would be 7

    """

    seq = char2num(sequence)
    if n_states is None:
        n_states = max(seq)+1
    # assert max(seq)+1==n_states, 'not all positions have a transition'
    TF = np.zeros([n_states, n_states], dtype=int)
    for i, p1 in enumerate(seq):
        if i+1>=len(seq): continue
        p2 = seq[(i+1) % len(seq)]
        TF[p1, p2] = 1
    return TF.astype(float)

def seq2TF_2step(seq: str, n_states: int | None = None) -> pd.DataFrame:
    """create a transition matrix with all 2 steps from a sequence string,
    e.g. ABCDEFGE.  AB->C BC->D ..."""
    triplets = []
    if n_states is None:
        n_states = max(char2num(seq))+1
    TF2 = np.zeros([n_states**2, n_states], dtype=int)
    for i, p1 in enumerate(seq):
        if i+2>=len(seq): continue
        if p1=='_': continue
        triplet = seq[i] + seq[(i+1) % len(seq)] + seq[(i+2)% len(seq)]
        i = char2num(triplet[0])[0] * n_states + char2num(triplet[1])[0]
        j = char2num(triplet[2])
        TF2[i, j] = 1
        triplets.append(triplet)

    seq_set = num2char(np.arange(n_states))
    # for visualiziation purposes
    df = pd.DataFrame({c:TF2.T[i] for i,c in enumerate(seq_set)})
    df['index'] = [f'{y}{x}' for y in seq_set for x in seq_set]
    df = df.set_index('index')
    TF2 = df.loc[~(df==0).all(axis=1)]
    return TF2
