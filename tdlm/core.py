# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:10:07 2024

core functions for Temporally Delayed Linear Modelling

@author: simon.kern
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast

import numpy as np
from numba import njit
from numpy.linalg import pinv
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import toeplitz

from tdlm.utils import unique_permutations

# try:
    # from jax.numpy.linalg import pinv
# except ModuleNotFoundError:
#     logging.warning('jaxlib not installed, can speed up computation')

# helper functions for compact array construction
def ones(*shape: int) -> NDArray[np.float64]:
    return np.ones(shape, dtype=float)


def zeros(*shape: int) -> NDArray[np.float64]:
    return np.zeros(shape, dtype=float)


def nan(*shape: int) -> NDArray[np.float64]:
    return np.full(shape=shape, fill_value=np.nan)


def squash(arr: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(np.ravel(arr, "F"), dtype=float)  # Preserve Fortran-order flattening.

class TDLMResult(NamedTuple):
    forward_sequenceness: NDArray[np.float64]
    backward_sequenceness: NDArray[np.float64]


class WindowedResult(NamedTuple):
    window_values: NDArray[np.float64]
    window_starts: NDArray[np.int_]
    forward_sequenceness: NDArray[np.float64]
    backward_sequenceness: NDArray[np.float64]


class SignflipResult(NamedTuple):
    pvalue: float
    t_obs: float
    t_perms: NDArray[np.float64]


tdlmresult = TDLMResult
windowedresult = WindowedResult


def _solve_lstsq(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve linear regression with a fast path and robust fallbacks."""
    try:
        if A.shape[0] >= A.shape[1]:
            ata = A.T @ A
            atb = A.T @ B
            try:
                return np.asarray(np.linalg.solve(ata, atb), dtype=float)
            except np.linalg.LinAlgError:
                pass
        return np.asarray(np.linalg.lstsq(A, B, rcond=None)[0], dtype=float)
    except np.linalg.LinAlgError:
        return np.asarray(pinv(A) @ B, dtype=float)


def _validate_probas_tf_tb(
    probas: ArrayLike,
    tf: ArrayLike,
    tb: ArrayLike | None = None,
    *,
    func_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate core TDLM matrix inputs and return normalized arrays."""
    probas = np.asarray(probas, dtype=float)
    tf = np.asarray(tf, dtype=float)

    if probas.ndim != 2:
        raise ValueError(f"{func_name}: probas must be 2D (n_timepoints, n_states), got shape={probas.shape}")
    if tf.ndim != 2:
        raise ValueError(f"{func_name}: tf must be 2D, got shape={tf.shape}")
    if tf.shape[0] != tf.shape[1]:
        raise ValueError(f"{func_name}: tf must be square, got shape={tf.shape}")
    if tf.shape[0] != probas.shape[1]:
        raise ValueError(
            f"{func_name}: tf size ({tf.shape[0]}) must match probas n_states ({probas.shape[1]}), "
            f"got probas shape={probas.shape}, tf shape={tf.shape}"
        )

    if tb is None:
        tb = tf.T
    else:
        tb = np.asarray(tb, dtype=float)
        if tb.ndim != 2:
            raise ValueError(f"{func_name}: tb must be 2D, got shape={tb.shape}")
        if tb.shape[0] != tb.shape[1]:
            raise ValueError(f"{func_name}: tb must be square, got shape={tb.shape}")
        if tb.shape != tf.shape:
            raise ValueError(
                f"{func_name}: tb shape must match tf shape, got tb shape={tb.shape}, tf shape={tf.shape}"
            )

    return np.asarray(probas, dtype=float), np.asarray(tf, dtype=float), np.asarray(tb, dtype=float)


def _find_betas(
    probas: NDArray[np.float64], n_states: int, max_lag: int, alpha_freq: int | None = None
) -> NDArray[np.float64]:
    """for prediction matrix X (states x time), get transitions up to max_lag.
    Similar to cross-correlation, i.e. shift rows of matrix iteratively

    paralellizeable version
    """
    n_bins = max_lag + 1;

    # design matrix is now a matrix of nsamples X (n_states*max_lag)
    # with each column a shifted version of the state vector (shape=nsamples)
    dm = np.hstack([toeplitz(probas[:, kk].ravel(),
                             np.ravel([zeros(n_bins, 1)]))[:, 1:]
                    for kk in range(n_states)])

    betas = nan(n_states * max_lag, n_states);

    ## GLM: state regression, with other lags
    #TODO: Check if this does what is expected
    bins = alpha_freq if alpha_freq else max_lag

    for ilag in list(range(bins)):
        # create individual GLMs for each time lagged version
        ilag_idx = np.arange(0, n_states * max_lag, bins) + ilag;
        # add a vector of ones for controlling the regression
        ilag_X = np.pad(dm[:, ilag_idx], [[0, 0], [0, 1]], constant_values=1)

        # add control for certain time lags to reduce alpha
        # Now find coefficients that solve the linear regression for this timelag
        # this a the second stage regression
        ilag_betas = _solve_lstsq(ilag_X, probas)
        betas[ilag_idx, :] = ilag_betas[0:-1, :];

    return betas


@njit  # type: ignore[untyped-decorator]
def _numba_roll(X: NDArray[np.float64], shift: int) -> NDArray[np.float64]:
    """
    numba optimized np.roll function
    taken from https://github.com/tobywise/online-aversive-learning
    """
    # Rolls along 1st axis
    new_X = np.zeros_like(X)
    for i in range(X.shape[1]):
        new_X[:, i] = np.roll(X[:, i], shift)
    return new_X


def _mean_column_correlation(lhs: NDArray[np.float64], rhs: NDArray[np.float64]) -> float:
    """Return mean Pearson correlation across matching columns."""
    if lhs.shape[0] < 2:
        return float(np.nan)
    lhs_centered = lhs - np.mean(lhs, axis=0, keepdims=True)
    rhs_centered = rhs - np.mean(rhs, axis=0, keepdims=True)
    numerator = np.sum(lhs_centered * rhs_centered, axis=0)
    denom = np.sqrt(np.sum(lhs_centered ** 2, axis=0) * np.sum(rhs_centered ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denom
    corr = np.where(denom == 0, np.nan, corr)
    return float(np.nanmean(corr))


@njit  # type: ignore[untyped-decorator]
def _mean_column_correlation_numba(lhs: NDArray[np.float64], rhs: NDArray[np.float64]) -> float:
    """Numba-accelerated mean columnwise Pearson correlation."""
    n_time = lhs.shape[0]
    n_cols = lhs.shape[1]
    if n_time < 2:
        return np.nan

    corr_sum = 0.0
    n_valid = 0
    for col in range(n_cols):
        mean_l = 0.0
        mean_r = 0.0
        for t in range(n_time):
            mean_l += lhs[t, col]
            mean_r += rhs[t, col]
        mean_l /= n_time
        mean_r /= n_time

        num = 0.0
        den_l = 0.0
        den_r = 0.0
        for t in range(n_time):
            dl = lhs[t, col] - mean_l
            dr = rhs[t, col] - mean_r
            num += dl * dr
            den_l += dl * dl
            den_r += dr * dr

        denom = np.sqrt(den_l * den_r)
        if denom == 0.0:
            continue
        corr = num / denom
        if np.isnan(corr):
            continue
        corr_sum += corr
        n_valid += 1

    if n_valid == 0:
        return np.nan
    return corr_sum / n_valid


@njit  # type: ignore[untyped-decorator]
def _cross_correlation_numba(
    probas: NDArray[np.float64],
    probas_f: NDArray[np.float64],
    probas_b: NDArray[np.float64],
    max_lag: int,
    min_lag: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ff = np.empty(max_lag - min_lag, dtype=np.float64)
    fb = np.empty(max_lag - min_lag, dtype=np.float64)
    n_time = probas.shape[0]

    for lag in range(min_lag, max_lag):
        if lag == 0:
            x_lag = probas
            probas_f_lag = probas_f
            probas_b_lag = probas_b
        else:
            x_lag = probas[lag:, :]
            end = n_time - lag
            probas_f_lag = probas_f[:end, :]
            probas_b_lag = probas_b[:end, :]

        ff[lag - min_lag] = _mean_column_correlation_numba(x_lag, probas_f_lag)
        fb[lag - min_lag] = _mean_column_correlation_numba(x_lag, probas_b_lag)

    return ff, fb


def _cross_correlation(
    probas: NDArray[np.float64],
    tf: NDArray[np.float64],
    tb: NDArray[np.float64],
    max_lag: int = 40,
    min_lag: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes sequenceness by cross-correlation

    taken from https://github.com/tobywise/online-aversive-learning
    """
    probas_f = probas @ tf
    probas_b = probas @ tb
    try:
        return cast(
            tuple[NDArray[np.float64], NDArray[np.float64]],
            _cross_correlation_numba(
                np.asarray(probas, dtype=np.float64),
                np.asarray(probas_f, dtype=np.float64),
                np.asarray(probas_b, dtype=np.float64),
                max_lag,
                min_lag,
            ),
        )
    except Exception:
        ff = np.empty(max_lag - min_lag, dtype=float)
        fb = np.empty(max_lag - min_lag, dtype=float)

        for lag in range(min_lag, max_lag):
            x_lag = probas[lag:, :]
            if lag == 0:
                probas_f_lag = probas_f
                probas_b_lag = probas_b
            else:
                # Equivalent to _numba_roll(..., lag)[lag:, :] without allocation.
                probas_f_lag = probas_f[:-lag, :]
                probas_b_lag = probas_b[:-lag, :]

            ff[lag - min_lag] = _mean_column_correlation(x_lag, probas_f_lag)
            fb[lag - min_lag] = _mean_column_correlation(x_lag, probas_b_lag)

        return ff, fb


def signflip_test(
    sx: ArrayLike,
    n_perms: int = 10000,
    rng: int | np.random.Generator | None = None,
) -> SignflipResult:
    """
    One-sided max-t sign-flip permutation test across columns.


    For each permutation, flip a random subset observations by -1 (i.e.
    participant's sequenceness score for each time lag). Then, run a ttest for
    all time lags separately and note the maximum t-value this permutation.
    As a result, we have a distribution of t-values, which we compare against
    the ground truth base t-value of the original data. This accounts for the
    multiple comparison problem but measures random effects instead of
    fixed effects, making the test more robust than the previously used state-
    shuffling. Use tdlm.plot_tval_distribution(..) to plot the results

    Parameters
    ----------
    sx : ndarray
        Data matrix of shape (n_obs, n_cols). This will usually be your
        sequenceness results in form of (n_subjects, n_time_lags).
        All nan columns will be ignored (e.g. for the zero-th time lag)
    n_perms : int
        Number of sign-flip permutations.
    rng : int or numpy.random.Generator, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    SignflipResult : namedtuple
        pvalue : float
            Finite-sample corrected familywise p-value.
        t_obs : float
            Observed max t-statistic across columns.
        t_perms : ndarray
            Max t-statistic per permutation of shape (n_perms,).
    """
    sx = np.asarray(sx, dtype=float)
    if sx.ndim != 2:
        raise ValueError(f'sx must be 2D (n_subj, n_lags) but is {sx.shape=}, e.g. without permutations')
    rng = np.random.default_rng(rng)

    # remove all-NaN lag columns (commonly the first lag)
    valid_cols = ~np.isnan(sx).all(axis=0)
    sx = sx[:, valid_cols]
    if sx.shape[1] == 0:
        raise ValueError("sx must contain at least one non-NaN column")

    # calculate mean sequenceness
    mean_seq = np.nanmean(sx, axis=0)
    n = sx.shape[0]  # number of observations
    if n <= 1:
        raise ValueError(f"for signflip test n>1 is required, but n={n}, sx.shape={sx.shape}")

    # Columnwise SE (ddof=1). NaNs if n<2 propagate as intended.
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.nanstd(sx, axis=0, ddof=1)
        se = s / np.sqrt(n)

    # True column t-stats and max (one-sided, positive direction)
    with np.errstate(invalid='ignore', divide='ignore'):
        t_cols = mean_seq / se

    t_obs = np.nanmax(t_cols)

    # Vectorized sign flips: (B, n_obs) in {-1, +1}
    n_obs = sx.shape[0]
    flips = rng.integers(0, 2, size=(n_perms, n_obs), dtype=np.int8) * 2 - 1

    # Permutation means per column
    perm_sums = flips @ np.nan_to_num(sx, copy=False, nan=0.0)  # (B, n_cols)
    perm_means = perm_sums / n  # broadcast

    # Permutation "t" using fixed SE from original data
    with np.errstate(invalid='ignore', divide='ignore'):
        t_perm = perm_means / se  # (B, n_cols)

    # Max across columns per permutation (one-sided)
    t_perms = np.nanmax(t_perm, axis=1)

    # p value = number of samples above threshold, finite sample corrected
    ge = np.count_nonzero(t_perms >= t_obs)
    pvalue = (ge + 1) / (n_perms + 1)

    return SignflipResult(float(pvalue), float(t_obs), t_perms)


def sequenceness_crosscorr(
    probas: ArrayLike,
    tf: ArrayLike,
    tb: ArrayLike | None = None,
    n_shuf: int = 1000,
    min_lag: int = 0,
    max_lag: int = 50,
    alpha_freq: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> TDLMResult:

    probas, tf, tb = _validate_probas_tf_tb(probas, tf, tb, func_name="sequenceness_crosscorr")
    if n_shuf is None:
        raise ValueError("sequenceness_crosscorr: n_shuf must be an integer >= 0, got None")
    if int(n_shuf) < 0:
        raise ValueError(f"sequenceness_crosscorr: n_shuf must be >= 0, got {n_shuf}")
    n_shuf = int(n_shuf)
    if int(max_lag) < 0:
        raise ValueError(f"sequenceness_crosscorr: max_lag must be >= 0, got {max_lag}")
    if int(min_lag) < 0:
        raise ValueError(f"sequenceness_crosscorr: min_lag must be >= 0, got {min_lag}")
    if int(min_lag) > int(max_lag):
        raise ValueError(f"sequenceness_crosscorr: min_lag ({min_lag}) must be <= max_lag ({max_lag})")
    rng = np.random.default_rng(rng)

    n_states = probas.shape[-1]
    # unique permutations
    unique_perms = unique_permutations(np.arange(n_states), n_shuf, rng=rng)
    n_perms = len(unique_perms)

    seq_fwd_corr = nan(n_perms, max_lag + 1)  # forward cross-correlation
    seq_bkw_corr = nan(n_perms, max_lag + 1)  # backward cross-correlation

    for i in range(n_perms):
        # select next unique permutation of transitions
        # index 0 is the non-shuffled original transition matrix
        rp = unique_perms[i, :]
        tf_perm = tf[rp, :][:, rp]
        tb_perm = tb[rp, :][:, rp]
        seq_fwd_corr[i, :-1], seq_bkw_corr[i, :-1] = _cross_correlation(probas,
                                                                        tf_perm,
                                                                        tb_perm,
                                                                        max_lag=max_lag,
                                                                        min_lag=min_lag)
    return tdlmresult(seq_fwd_corr, seq_bkw_corr)


def cross_correlation(
    probas: ArrayLike,
    tf: ArrayLike,
    tb: ArrayLike | None = None,
    n_shuf: int = 1000,
    min_lag: int = 0,
    max_lag: int = 50,
    alpha_freq: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> TDLMResult:
    """Backward-compatible alias for sequenceness_crosscorr."""
    return sequenceness_crosscorr(probas, tf, tb=tb, n_shuf=n_shuf,
                                  min_lag=min_lag, max_lag=max_lag,
                                  alpha_freq=alpha_freq, rng=rng)


def _stack_numeric(values: Sequence[ArrayLike], *, value_name: str) -> NDArray[np.float64]:
    """Stack values into a consistent numeric ndarray."""
    arrays = [np.asarray(v) for v in values]
    if not arrays:
        return np.empty((0,), dtype=float)
    try:
        stacked = np.stack(arrays)
    except ValueError as exc:
        raise ValueError(
            f"{value_name} must have a consistent shape across windows/trials; "
            f"received shapes={[a.shape for a in arrays]}"
        ) from exc
    if not np.issubdtype(stacked.dtype, np.number):
        raise TypeError(f"{value_name} must be numeric, got dtype={stacked.dtype}")
    return np.asarray(stacked, dtype=float)


def _default_window_aggregate(seq_fwd: NDArray[np.float64], seq_bkw: NDArray[np.float64]) -> float:
    """Default aggregation for windowed TDLM: mean forward-backward difference."""
    return float(np.nanmean(seq_fwd[0] - seq_bkw[0]))


def compute_1step_per_trial(
    probas_trials: ArrayLike,
    tfs: Sequence[ArrayLike] | NDArray[np.float64],
    tbs: Sequence[ArrayLike | None] | NDArray[np.float64] | None = None,
    n_shuf: int = 100,
    min_lag: int = 0,
    max_lag: int = 50,
    alpha_freq: int | None = None,
    max_true_trans: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> TDLMResult:
    """
    Compute 1-step TDLM for multiple trials with trial-specific transition matrices.

    Parameters
    ----------
    probas_trials : np.ndarray
        3D prediction array of shape (n_trials, n_timepoints, n_states).
    tfs : np.ndarray | list[np.ndarray]
        Either one transition matrix (shared across trials) or one per trial.
    tbs : np.ndarray | list[np.ndarray] | None
        Optional backward transition matrices (shared or per-trial).
    rng : int or numpy.random.Generator, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    tdlmresult
        forward_sequenceness and backward_sequenceness stacked per trial
        with shape (n_trials, n_shuf, max_lag + 1).
    """
    probas_trials = np.asarray(probas_trials, dtype=float)
    if probas_trials.ndim != 3:
        raise ValueError(f'probas_trials must be 3D (n_trials, n_time, n_states), got {probas_trials.shape=}')

    n_trials = probas_trials.shape[0]

    if isinstance(tfs, np.ndarray) and tfs.ndim == 2:
        tfs_raw: list[ArrayLike] = [tfs] * n_trials
    else:
        tfs_raw = list(tfs)
    if len(tfs_raw) != n_trials:
        raise ValueError(f'expected one tf per trial ({n_trials}), got {len(tfs_raw)}')

    tfs_list: list[NDArray[np.float64]] = []
    for idx, tf_i in enumerate(tfs_raw):
        tf_arr = np.asarray(tf_i, dtype=float)
        if tf_arr.ndim != 2:
            raise ValueError(f"tfs[{idx}] must be 2D, got shape={tf_arr.shape}")
        if tf_arr.shape[0] != tf_arr.shape[1]:
            raise ValueError(f"tfs[{idx}] must be square, got shape={tf_arr.shape}")
        if tf_arr.shape[0] != probas_trials.shape[2]:
            raise ValueError(
                f"tfs[{idx}] size ({tf_arr.shape[0]}) must match trial n_states ({probas_trials.shape[2]})"
            )
        tfs_list.append(tf_arr)

    if tbs is None:
        tbs_raw: list[ArrayLike | None] = [None] * n_trials
    elif isinstance(tbs, np.ndarray) and tbs.ndim == 2:
        tbs_raw = [tbs] * n_trials
    else:
        tbs_raw = list(tbs)
    if len(tbs_raw) != n_trials:
        raise ValueError(f'expected one tb per trial ({n_trials}), got {len(tbs_raw)}')

    tbs_list: list[NDArray[np.float64] | None] = []
    for idx, tb_i in enumerate(tbs_raw):
        if tb_i is None:
            tbs_list.append(None)
            continue
        tb_arr = np.asarray(tb_i, dtype=float)
        if tb_arr.ndim != 2:
            raise ValueError(f"tbs[{idx}] must be 2D, got shape={tb_arr.shape}")
        if tb_arr.shape != tfs_list[idx].shape:
            raise ValueError(f"tbs[{idx}] shape must match tfs[{idx}] shape {tfs_list[idx].shape}, got {tb_arr.shape}")
        tbs_list.append(tb_arr)

    rng = np.random.default_rng(rng)

    sf_trials = []
    sb_trials = []
    for trial_idx in range(n_trials):
        trial_seed = int(rng.integers(np.iinfo(np.int64).max))
        sf_i, sb_i = compute_1step(
            probas_trials[trial_idx],
            tfs_list[trial_idx],
            tb=tbs_list[trial_idx],
            n_shuf=n_shuf,
            min_lag=min_lag,
            max_lag=max_lag,
            alpha_freq=alpha_freq,
            max_true_trans=max_true_trans,
            rng=trial_seed,
        )
        sf_trials.append(sf_i)
        sb_trials.append(sb_i)

    return tdlmresult(
        _stack_numeric(sf_trials, value_name="forward_sequenceness"),
        _stack_numeric(sb_trials, value_name="backward_sequenceness"),
    )


def compute_windowed(
    probas: ArrayLike,
    tf: ArrayLike,
    tb: ArrayLike | None = None,
    win_size: int = 200,
    step_size: int | None = None,
    aggr_func: Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike] | None = None,
    seq_type: str = 'glm',
    n_shuf: int = 100,
    min_lag: int = 0,
    max_lag: int = 50,
    alpha_freq: int | None = None,
    max_true_trans: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> WindowedResult:
    """
    Compute windowed sequenceness by running TDLM on successive time windows.

    Parameters
    ----------
    probas : np.ndarray
        Prediction matrix of shape (n_timepoints, n_states).
    tf, tb : np.ndarray
        Forward / backward transition matrices.
    win_size : int
        Number of timepoints per window.
    step_size : int | None
        Step between window starts; defaults to win_size (non-overlapping).
    aggr_func : callable | None
        Function with signature aggr_func(seq_fwd, seq_bkw) applied per window.
        Defaults to mean differential sequenceness.
    seq_type : str
        One of {'glm', '1step', 'crosscorr', '2step'}.

    Returns
    -------
    WindowedResult
        window_values: aggregated value(s) per window
        window_starts: starting sample index per window
        forward_sequenceness / backward_sequenceness: per-window TDLM outputs
    """
    probas = np.asarray(probas)
    if probas.ndim != 2:
        raise ValueError(f'probas must be 2D (n_timepoints, n_states), got {probas.shape=}')

    if win_size is None:
        raise ValueError('win_size must be provided')
    win_size = int(win_size)
    if win_size <= 0:
        raise ValueError(f'win_size must be > 0, got {win_size}')
    if win_size > len(probas):
        raise ValueError(f'win_size ({win_size}) cannot exceed number of timepoints ({len(probas)})')

    if step_size is None:
        step_size = win_size
    step_size = int(step_size)
    if step_size <= 0:
        raise ValueError(f'step_size must be > 0, got {step_size}')

    if aggr_func is None:
        aggr_func = _default_window_aggregate
    if not callable(aggr_func):
        raise ValueError('aggr_func must be callable')

    seq_key = str(seq_type).lower().replace('-', '').replace('_', '')
    if seq_key in {'glm', '1step', 'onestep'}:
        seq_fn = 'glm'
    elif seq_key in {'crosscorr', 'crosscorrelation'}:
        seq_fn = 'crosscorr'
    elif seq_key in {'2step', 'twostep'}:
        seq_fn = '2step'
    else:
        raise ValueError(
            f"unknown seq_type={seq_type!r}, expected one of 'glm', 'crosscorr', '2step'"
        )

    window_starts = np.arange(0, len(probas) - win_size + 1, step_size, dtype=int)
    rng = np.random.default_rng(rng)

    window_values = []
    sf_windows = []
    sb_windows = []

    for start in window_starts:
        stop = start + win_size
        probas_win = probas[start:stop, :]
        trial_seed = int(rng.integers(np.iinfo(np.int64).max))

        if seq_fn == 'glm':
            sf_i, sb_i = compute_1step(
                probas_win,
                tf,
                tb=tb,
                n_shuf=n_shuf,
                min_lag=min_lag,
                max_lag=max_lag,
                alpha_freq=alpha_freq,
                max_true_trans=max_true_trans,
                rng=trial_seed,
            )
        elif seq_fn == '2step':
            sf_i, sb_i = compute_2step(
                probas_win,
                tf,
                tb=tb,
                n_shuf=n_shuf,
                min_lag=min_lag,
                max_lag=max_lag,
                alpha_freq=alpha_freq,
                rng=trial_seed,
            )
        else:
            sf_i, sb_i = sequenceness_crosscorr(
                probas_win,
                tf,
                tb=tb,
                n_shuf=n_shuf,
                min_lag=min_lag,
                max_lag=max_lag,
                alpha_freq=alpha_freq,
                rng=trial_seed,
            )

        sf_windows.append(sf_i)
        sb_windows.append(sb_i)
        window_values.append(aggr_func(sf_i, sb_i))

    return windowedresult(
        _stack_numeric(window_values, value_name="window_values"),
        window_starts,
        _stack_numeric(sf_windows, value_name="forward_sequenceness"),
        _stack_numeric(sb_windows, value_name="backward_sequenceness"),
    )


# @profile
def compute_1step(
    probas: ArrayLike,
    tf: ArrayLike,
    tb: ArrayLike | None = None,
    n_shuf: int = 100,
    min_lag: int = 0,
    max_lag: int = 50,
    alpha_freq: int | None = None,
    max_true_trans: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> TDLMResult:
    """
    Calculate 1-step-sequenceness for probability estimates and transitions.

    Parameters
    ----------
    probas : np.ndarray
        2d matrix with predictions, shape= (timesteps, n_states), where each
        timestep contains n_states prediction values for states at that time
    tf : np.ndarray
        transition matrix with expected transitions for the underlying states.
    tb : np.ndarray
        backward transition matrix expected transitions for the underlying
        states. In case transitions are non-directional, the backwards matrix
        is simply set to be the transpose of tf. Default tb = tf.T
    n_shuf : int
        number of random shuffles to be done for permutation testing.
    max_lag : int
        maximum time lag to calculate. Time dimension is measured in sample
        steps of the probas time dimension.
    alpha_freq : int, optional
        Alpha oscillation frequency to control for. Time shifted copies of the
        signal are added in this frequency to the GLM, acting as a confounds.
        Warning: Must be supplied in sample points, not in Hertz!
        The default is None.
    max_true_trans : int, optional
        Maximum number of transitions that should be be overlapping between the
        real sequence and shuffles. E.g. if your sequence is A->B->C, the
        permutation B->C->A would contain one overlapping transition B->C.
        Setting max_true_trans=0 would remove this permutation from the test.
        The default is None, i.e. no limit.
    rng : int or numpy.random.Generator, optional
        Seed or Generator for reproducibility.
    n_steps : int, optional
        number of transition steps to look for. Not implemented yet.
        The default is 1.


    Returns
    -------
    sf : np.ndarray
        forward sequences for all time lags and shuffles. Row 0 is the
        non-shuffled version. First lag is NAN as it is undefined for lag = 0
    sb : np.ndarray
        backward sequences for all time lags and shuffles. Row 0 is the
        non-shuffled version. First lag is NAN as it is undefined for lag = 0
    """
    probas, tf, tb = _validate_probas_tf_tb(probas, tf, tb, func_name="compute_1step")
    if n_shuf is None:
        raise ValueError("compute_1step: n_shuf must be an integer >= 0, got None")
    if int(n_shuf) < 0:
        raise ValueError(f"compute_1step: n_shuf must be >= 0, got {n_shuf}")
    n_shuf = int(n_shuf)
    if int(max_lag) < 0:
        raise ValueError(f"compute_1step: max_lag must be >= 0, got {max_lag}")
    if int(min_lag) < 0:
        raise ValueError(f"compute_1step: min_lag must be >= 0, got {min_lag}")
    if int(min_lag) > int(max_lag):
        raise ValueError(f"compute_1step: min_lag ({min_lag}) must be <= max_lag ({max_lag})")
    if alpha_freq is not None:
        if alpha_freq <= 0:
            raise ValueError(f'alpha_freq must be positive, got {alpha_freq=}')
        if alpha_freq > max_lag:
            raise ValueError(f'alpha_freq must be <= max_lag, got {alpha_freq=} and {max_lag=}')

    rng = np.random.default_rng(rng)

    n_states = probas.shape[-1]
    # unique permutations
    unique_perms = unique_permutations(np.arange(n_states), n_shuf,
                                       max_true_trans=max_true_trans, rng=rng)

    n_perms = len(unique_perms)  # this might be different to requested n_shuf!

    seq_fwd = nan(n_perms, max_lag + 1)  # forward sequenceness
    seq_bkw = nan(n_perms, max_lag + 1)  # backward sequencenes

    ## GLM: state regression, with other lags

    betas = _find_betas(probas, n_states, max_lag, alpha_freq=alpha_freq)
    # betas = find_betas_optimized(X, n_states, max_lag, alpha_freq=alpha_freq)
    # np.testing.assert_array_almost_equal(betas, betas2, decimal= 12)

    # reshape the coeffs for regression to be in the order of ilag x (n_states x n_states)
    betasn_ilag_stage = np.reshape(betas, [max_lag, n_states ** 2], order='F');

    for i in range(n_perms):
        rp = unique_perms[i, :]  # select next unique permutation of transitions
        tf_perm = tf[rp, :][:, rp]
        tb_perm = tb[rp, :][:, rp]
        t_auto = np.eye(n_states)  # control for auto correlations
        t_const = np.ones([n_states, n_states])  # keep betas in same range

        # create our design matrix for the second step analysis
        dm = np.vstack([squash(tf_perm), squash(tb_perm), squash(t_auto), squash(t_const)]).T
        # now calculate regression coefs for use with transition matrix
        bbb = _solve_lstsq(dm, betasn_ilag_stage.T)

        seq_fwd[i, 1:] = bbb[0, :]  # forward coeffs
        seq_bkw[i, 1:] = bbb[1, :]  # backward coeffs

    return tdlmresult(seq_fwd, seq_bkw)


def compute_2step(
    probas: ArrayLike,
    tf: ArrayLike,
    tb: ArrayLike | None = None,
    n_steps: int = 2,
    n_shuf: int | None = None,
    min_lag: int = 0,
    max_lag: int = 50,
    alpha_freq: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> TDLMResult:
    """
    2-step TDLM implementation.

    I do think there are conceptual problems with this implementation,
    therefore, I do not recommend using the method without further consideration
    e.g. if our data contains A->B, but never C, we will _still_ find backwards
    sequenceness evidence simply because (C*B) is regressed on A for the back-
    wards case and will induce spurious sequenceness of A->B->C when there is
    no triplet replay

    Parameters
    ----------
    rng : int or numpy.random.Generator, optional
        Seed or Generator for reproducibility.
    """
    probas, tf, tb = _validate_probas_tf_tb(probas, tf, tb, func_name="compute_2step")
    if n_steps != 2:
        raise ValueError(f"compute_2step currently supports only n_steps=2, got {n_steps}")
    if int(max_lag) < 1:
        raise ValueError(f"compute_2step: max_lag must be >= 1, got {max_lag}")
    if int(min_lag) < 0:
        raise ValueError(f"compute_2step: min_lag must be >= 0, got {min_lag}")
    if int(min_lag) > int(max_lag):
        raise ValueError(f"compute_2step: min_lag ({min_lag}) must be <= max_lag ({max_lag})")

    n_states = probas.shape[-1]
    est_lag_matrix_bytes = probas.shape[0] * (n_states ** 2) * np.dtype(float).itemsize
    max_allowed_bytes = 1_500_000_000  # ~1.5 GB lag-local intermediate matrix
    if est_lag_matrix_bytes > max_allowed_bytes:
        raise ValueError(
            "compute_2step input is too large for safe lag-local processing: "
            f"estimated lag matrix size is {est_lag_matrix_bytes / (1024 ** 3):.2f} GiB "
            f"for probas shape={probas.shape}. Reduce n_states or timepoints."
        )

    rng = np.random.default_rng(rng)

    # seq = tf2seq(tf)
    unique_perms = unique_permutations(np.arange(n_states), n_shuf, rng=rng)

    n_perms = len(unique_perms)  # this might be different to requested n_shuf!

    # create all two step transitions from our transition matrix
    tf_y = []
    tf_x2 = []
    tf_x1 = []

    seq_starts = np.where(tf)
    for x1, x2 in zip(*seq_starts):
        for y in np.where(tf[x2, :])[0]:
            tf_x1 += [x1]
            tf_x2 += [x2]
            tf_y  += [y]

    tr_y = tf_x1
    tr_x2 = tf_x2
    tr_x1 = tf_y

    tf2 = zeros(len(tf_y), n_states);
    tf_auto = zeros(len(tf_y), n_states);

    for i in range(len(tf_y)):
        tf2[i,tf_y[i]] = 1;
        tf_auto[i, np.unique([tf_x1[i], tf_x2[i]])]=1;

    tr2 = zeros(len(tr_y), n_states);
    tr_auto = zeros(len(tr_y), n_states);
    for i in range(len(tr_y)):
        tr2[i, tr_y[i]] = 1;
        tr_auto[i, np.unique([tr_x1[i], tr_x2[i]])]=1;


    # Initialize variables
    x = probas
    y = x

    beta_f = np.full((max_lag, len(tf_y), n_states), np.nan)
    beta_b = np.full((max_lag, len(tr_y), n_states), np.nan)

    # Second loop
    for lag_idx in range(1, max_lag + 1):
        pad = np.zeros((lag_idx, n_states))
        x1 = np.vstack([pad, pad, x[:-2 * lag_idx, :]])
        x2 = np.vstack([pad, x[:-lag_idx, :]])
        x_matrix = x1[:, :, None] * x2[:, None, :]
        lag = lag_idx - 1

        for state in range(len(tf_y)):
            x_fwd = x_matrix[:, :, tf_x2[state]]
            x_bkw = x_matrix[:, :, tr_x2[state]]

            temp_f = _solve_lstsq(np.hstack([x_fwd, np.ones((len(x_fwd), 1))]), y)
            beta_f[lag, state, :] = temp_f[tf_x1[state], :]

            temp_b = _solve_lstsq(np.hstack([x_bkw, np.ones((len(x_bkw), 1))]), y)
            beta_b[lag, state, :] = temp_b[tr_x1[state], :]

    beta_f = beta_f.reshape((max_lag, len(tf_y) * n_states), order='F')
    beta_b = beta_b.reshape((max_lag, len(tr_y) * n_states), order='F')

    seq_fwd = nan(n_perms, max_lag+1);
    seq_bkw = nan(n_perms, max_lag+1);

    # Third loop
    for shuffle_idx in range(n_perms):
        random_permutation = unique_perms[shuffle_idx, :]

        # 2nd level
        const_f = np.ones((len(tf_y), n_states))
        const_r = np.ones((len(tr_y), n_states))

        tf_shuffled = tf2[:, random_permutation]
        tr_shuffled = tr2[:, random_permutation]

        cc = _solve_lstsq(np.hstack([
            squash(tf_shuffled)[:, None],
            squash(tf_auto)[:, None],
            squash(const_f)[:, None]
        ]), beta_f.T)
        seq_fwd[shuffle_idx, 1:] = cc[0, :]

        cc = _solve_lstsq(np.hstack([
            squash(tr_shuffled)[:, None],
            squash(tr_auto)[:, None],
            squash(const_r)[:, None]
        ]), beta_b.T)
        seq_bkw[shuffle_idx, 1:] = cc[0, :]
    return tdlmresult(seq_fwd, seq_bkw)


