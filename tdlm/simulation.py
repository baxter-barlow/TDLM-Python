#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 21:35:04 2025

@author: simon
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.signal import lfilter

def simulate_meeg(
    length: float,
    sfreq: float,
    n_channels: int = 64,
    cov: ArrayLike | None = None,
    autocorr: float = 0.95,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Simulate M/EEG resting-state data.

    Parameters
    ----------
    length : float
        Total duration of the signal in seconds.
    sfreq : float
        Sampling frequency in Hz (samples per second).
    n_channels : int, optional
        Number of EEG channels (default is 64).
    cov : numpy.ndarray, optional
        Covariance matrix of shape (n_channels, n_channels).
        If None, a random covariance matrix is generated.
    autocorr : float, optional
        Temporal correlation of each sample with its neighbour samples.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    eeg_data : numpy.ndarray
        Simulated EEG data of shape (n_samples, n_channels).
    """
    if not (0 <= autocorr < 1):
        raise ValueError(f"autocorr must satisfy 0 <= autocorr < 1, got {autocorr}")
    n_samples = int(length * sfreq)
    rng = np.random.default_rng(rng)

    # 1. Setup Covariance (Cholesky Decomposition)
    if cov is None:
        A = rng.normal(size=(n_channels, n_channels))
        # Create symmetric positive-definite matrix
        cov_mat = (A + A.T) / 2
        _, U = np.linalg.eig(cov_mat)
        # Reconstruct with positive eigenvalues
        cov_mat = U @ np.diag(np.abs(rng.normal(size=n_channels))) @ U.T
    else:
        cov_mat = np.asarray(cov, dtype=float)
        if cov_mat.ndim != 2:
            raise ValueError(f"cov must be 2D, got shape={cov_mat.shape}")
        if cov_mat.shape[0] != cov_mat.shape[1]:
            raise ValueError(f"cov must be square, got shape={cov_mat.shape}")
        n_channels = int(cov_mat.shape[0])

    # Compute Mixing Matrix (L) from Covariance
    # We use Cholesky: Cov = L @ L.T
    # If Cov is not strictly positive definite
    L = np.linalg.cholesky(cov_mat)

    # 2. Generate White Noise (Standard Normal)
    Z = rng.standard_normal((n_samples, n_channels))

    # 3. Apply Temporal Filter to White Noise
    # Original logic: noise was scaled by autocorr before addition
    # Filter: y[n] = autocorr * y[n-1] + (autocorr * x[n])
    # To match original magnitude logic: Scale input noise by autocorr
    Z *= autocorr

    # Apply Filter along time axis (axis 0)
    # b=[1], a=[1, -autocorr]
    # We use zi to handle initial conditions smoothly if needed,
    # but strictly Z starts random, so standard filter is fine.
    Z = lfilter([1], [1, -autocorr], Z, axis=0)

    # 4. Apply Spatial Mixing (Matrix Multiplication)
    # X = Z_filtered @ L.T
    # This moves the heavy O(N*M^2) operation to a
    # single highly optimized BLAS call
    X = Z @ L.T

    return np.asarray(X, dtype=float)

def simulate_classifier_patterns(
    n_patterns: int = 10,
    n_channels: int = 306,
    noise: float = 4,
    scale: float = 1,
    n_train_per_stim: int = 18,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]]:
    """
    Generates synthetic training data and labels for TDLM experiments.

    Parameters
    ----------
    n_patterns : int
        Number of unique stimulus patterns.
    n_channels : int
        Number of sensor channels.
    noise : float
        Standard deviation of background noise.
    n_train_per_stim : int
        Repetitions per stimulus.
    rng : int or np.random.Generator, optional
        Random int seed or generator.

    Returns
    -------
    training_data : np.ndarray
        Simulated sensor data.
    training_labels : np.ndarray
        Labels (0 for null, 1-N for stimuli).
    patterns : np.ndarray
        Ground truth patterns.
    """
    rng = np.random.default_rng(rng)

    # Setup dimensions
    n_null = n_train_per_stim * n_patterns
    n_stim_total = n_patterns * n_train_per_stim
    n_total = n_null + n_stim_total

    # Generate Patterns

    # common "ERP"-style pattern that is common to all patterns
    common_pattern = rng.normal(size=(1, n_channels))
    # then repeat pattern for each class and add some gaussian noise
    patterns = np.tile(common_pattern, (n_patterns, 1)) + \
               rng.standard_normal((n_patterns, n_channels))

    # base noise that is added to the trials
    base_noise = noise * rng.standard_normal((n_total, n_channels))

    # construct individual trials, later add noise
    stim_signal = np.tile(patterns, (n_train_per_stim, 1))

    # create matrix with n_null empty spaces and the individual trials
    signal_component = np.vstack([
        np.zeros((n_null, n_channels)),
        stim_signal
    ])

    # add noise to the individual trials
    training_data = base_noise + signal_component

    # Generate Labels, the zero class will act as negative samples later on
    stim_labels = np.tile(np.arange(1, n_patterns + 1), n_train_per_stim)
    training_labels = np.concatenate([
        np.zeros(n_null, dtype=int),
        stim_labels
    ])

    # Inject Extra Noise to half the patterns
    n_noise_groups = n_patterns // 2

    if n_noise_groups > 0:
        # choose which classes to make more noisy, but don't add noise
        # to the negative zero class
        more_noise_inds = rng.choice(np.arange(1, n_patterns + 1),
                                     size=n_noise_groups,
                                     replace=False)

        for idx in more_noise_inds:
            start_rel = (idx - 1) * n_train_per_stim
            end_rel = idx * n_train_per_stim

            s_idx = n_null + start_rel
            e_idx = n_null + end_rel

            segment_len = e_idx - s_idx
            training_data[s_idx:e_idx, :] += rng.standard_normal((segment_len, n_channels))

    # training data includes the zero null class, which is basically just noise
    return training_data*scale, training_labels, patterns*scale


def simulate_eeg_resting_state(
    length: float,
    sfreq: float,
    n_channels: int = 64,
    cov: ArrayLike | None = None,
    autocorr: float = 0.95,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Convenience wrapper for EEG resting-state simulations.

    This is a named alias of ``simulate_meeg`` for discoverability in EEG-focused
    workflows.
    """
    return simulate_meeg(
        length=length,
        sfreq=sfreq,
        n_channels=n_channels,
        cov=cov,
        autocorr=autocorr,
        rng=rng,
    )


def simulate_eeg_localizer(
    n_patterns: int = 10,
    n_channels: int = 306,
    noise: float = 4,
    scale: float = 1,
    n_train_per_stim: int = 18,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.float64]]:
    """
    Convenience wrapper to simulate EEG localizer-style training data.

    This is a named alias of ``simulate_classifier_patterns`` that returns:
    (training_data, training_labels, patterns).
    """
    return simulate_classifier_patterns(
        n_patterns=n_patterns,
        n_channels=n_channels,
        noise=noise,
        scale=scale,
        n_train_per_stim=n_train_per_stim,
        rng=rng,
    )



def insert_events(
    data: ArrayLike,
    insert_data: ArrayLike,
    insert_labels: ArrayLike,
    n_events: int,
    lag: int = 8,
    jitter: int = 0,
    n_steps: int = 1,
    refractory: int | list[int] | None = 16,
    distribution: str | ArrayLike = 'constant',
    transitions: ArrayLike | None = None,
    sequence: Sequence[int] | None = None,
    return_onsets: bool = False,
    rng: int | np.random.Generator | None = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], pd.DataFrame]:
    """
    inject decodable events into M/EEG data according to a certain pattern.


    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    insert_data : np.ndarray
        data that should be inserted. Length must be the same as insert_labels.
        Must be 2D, with the second dimension being the sensor dimension.
        If insert_data is 3D, last dimension is taken as a time dimension
    insert_labels : np.ndarray
        list of class labels/ids for the insert_data.
    mean_class: bool
        insert the mean of the class if True, else insert a random single event
        from insert_data.
    lag : TYPE, optional
        Sample space distance individual reactivation events events.
        The default is 7 (e.g. 70 ms replay speed time lag).
    jitter : int, optional
        By how many sample points to jitter the events (randomly).
        The default is 0.
    refractory: int | list of two int
        how many samples of blocking there should be before and after each
        sequence start and sequence end. If integer, apply same to both sides.
        If list of two ints, interpret as left and right blocking period.
        Example: [5, 10] would block 5 steps before the an event start,
        and 10 steps after the last event point. If an event starts at sample
        100 with 2 steps and a lag of 8, the last event point would be at
        100+8+8, so the period of (100-5) to (100+8+8+10+1) would be blocked
        If set to None, disregard and allow overlapping sequences.
    transitions: list of list
        the sequence transitions that should be sampled from.
        if it is a 1d list, transitions will be extracted automatically
    n_steps : int, optional
        Number of events to insert. The default is 2
    distribution : str | np.ndarray, optional
        How replay events should be distributed throughout the time series.
        Can either be 'constant', 'increasing' or 'decreasing' or a p vector
        with probabilities for each sample point in data.
        The default is 'constant'.
    rng : np.random.Generator | int
        random generator or integer seed

    Returns
    -------
    data : np.ndarray (shape=data.shape)
        data with inserted events.
    (optional) return_onsets: pd.DataFrame
        table with onsets of events
    """
    data = np.asarray(data, dtype=float)
    insert_data = np.asarray(insert_data, dtype=float)
    insert_labels = np.asarray(insert_labels)
    import logging
    if len(insert_data) != len(insert_labels):
        raise ValueError(
            f"each insert_data row must have a corresponding insert_labels entry; "
            f"got len(insert_data)={len(insert_data)}, len(insert_labels)={len(insert_labels)}"
        )
    if insert_data.ndim not in (2, 3):
        raise ValueError(f"insert_data must be 2D or 3D, got shape={insert_data.shape}")
    if insert_labels.ndim != 1:
        raise ValueError(f"insert_labels must be 1D, got shape={insert_labels.shape}")
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (time, channels), got shape={data.shape}")
    if data.shape[1] != insert_data.shape[1]:
        raise ValueError(
            f"channel dimension mismatch: data has {data.shape[1]}, insert_data has {insert_data.shape[1]}"
        )
    if np.min(insert_labels) != 0:
        raise ValueError(
            f"insert_labels must start at 0 and be consecutive, got min(insert_labels)={np.min(insert_labels)}"
        )

    if isinstance(distribution, np.ndarray):
        if len(distribution) != len(data):
            raise ValueError(
                f"distribution length must match data length, got len(distribution)={len(distribution)}, len(data)={len(data)}"
            )
        if distribution.ndim != 1:
            raise ValueError(f"distribution must be 1D, got shape={distribution.shape}")
        if not np.isclose(distribution.sum(), 1):
            raise ValueError(f"distribution must sum to 1, got sum={distribution.sum()}")

    if (sequence is None) == (transitions is None):
        raise ValueError('provide exactly one of sequence or transitions')

    # no events requested? simply return
    if not n_events:
        return (data, pd.DataFrame()) if return_onsets else data

    # assume refractory period is valid for both sides
    if isinstance(refractory, int):
        refractory = [refractory, refractory]

    if sequence is not None:
        if len(sequence) < (n_steps + 1):
            raise ValueError(f'sequence must contain at least {n_steps + 1} states')
        transitions = [sequence[i:i + n_steps + 1]
                       for i in range(len(sequence) - n_steps)]
    else:
        transitions = np.array(transitions)
        if transitions.ndim == 1:
            if len(transitions) < (n_steps + 1):
                raise ValueError(f'1D transitions must contain at least {n_steps + 1} states')
            transitions = [transitions[i:i + n_steps + 1]
                           for i in range(len(transitions) - n_steps)]

    transitions = np.asarray(transitions)
    if transitions.ndim != 2 or transitions.shape[1] != (n_steps + 1):
        raise ValueError(f'each transition must have exactly {n_steps + 1} steps')

    del sequence # for safety, can be removed later

    # convert data to 3d
    if insert_data.ndim==2:
        insert_data = insert_data.reshape([*insert_data.shape, 1])

    # work on copy of array to prevent mutable changes
    data_sim: NDArray[np.float64] = data.copy()

    # get reproducible seed
    rng = np.random.default_rng(rng)

    # Calculate probability distribution based on the specified distribution type
    if isinstance(distribution, str):
        if distribution=='constant':
            p = np.ones(len(data))
            p = p/p.sum()
        elif distribution=='decreasing':
            p = np.linspace(1, 0, len(data))**2
            p = p/p.sum()
        elif distribution=='increasing':
            p = np.linspace(0, 1, len(data))**2
            p = p/p.sum()
        else:
            raise ValueError(f'unknown {distribution=}')
    elif isinstance(distribution, (list, np.ndarray)):
        distribution = np.array(distribution)
        if len(distribution) != len(data):
            raise ValueError(
                f"distribution length must match data length, got distribution shape={distribution.shape}, len(data)={len(data)}"
            )
        if not np.isclose(distribution.sum(), 1):
            raise ValueError(f"distribution must sum to 1, got sum={distribution.sum()}")
        p = distribution
    else:
        raise ValueError(f'distribution must be string or p-vector, {distribution=}')

    # block impossible starting points (i.e. out of bounds)
    tspan = insert_data.shape[-1] # timespan of one pattern
    event_length = n_steps*lag + tspan -1  # time span of one replay events
    p[-event_length:] = 0  # dont start events at end of resting state
    p[:tspan] = 0  # block beginning of resting state
    if p.sum() <= 0:
        raise ValueError('no valid start positions are available with current settings')
    p = p/p.sum()  # normalize probability vector again after removing indices

    replay_start_idxs = []
    all_idx = np.arange(len(data))

    # iteratively select starting index for replay event
    # such that replay events are not overlapping
    for i in range(n_events):

        # Choose a random idx from the available indices to start replay event
        start_idx = rng.choice(all_idx, p=p)
        replay_start_idxs.append(start_idx)



        # Update the p array to zero out the region around the chosen index to prevent overlap
        if refractory is not None:
            # next block the refractory period to prevent overlap
            block_start = start_idx - refractory[0]
            block_end   = start_idx + lag*n_steps + refractory[1] + 1

            block_start = max(block_start, 0)
            block_end = min(block_end, len(p))

            p[block_start: block_end] = 0

            # normalize to create valid probability distribution
            if p.sum() <= 0:
                raise ValueError(f'no more positions to insert events! {n_events=} too high?')
            p = p/p.sum()

        # check that we actually still have enough positions to insert
        # another event of length lag*n_steps. Probably the function fails
        # beforehand though.
        if (p>0).sum() < n_steps*lag:
            raise ValueError(f'no more positions to insert events! {n_events=} too high?')



    # save data about inserted events here and return if requested
    events: dict[str, list[int]] = {
        'event_idx': [],
        'pos': [],
        'step': [],
        'class_idx': [],
        'span': [],
        'jitter': [],
    }

    for idx,  start_idx in enumerate(replay_start_idxs):
        smp_jitter = 0  # starting with no jitter
        pos = start_idx  # pos indicates where in data we insert the next event

        # randomly sample a transition that we would take
        trans = rng.choice(transitions)

        for step, class_idx  in enumerate(trans):
            # choose which item should be inserted based on sequence order
            # or take a single event (more noisy)
            data_class = insert_data[insert_labels==class_idx]
            idx_cls_i = rng.choice(np.arange(len(data_class)))
            insert_data_i = data_class[idx_cls_i]
            if insert_data_i.ndim != 2:
                raise ValueError(f"insert_data_i must be 2D after class sampling, got shape={insert_data_i.shape}")

            # time spans of the segments we want to insert
            t_start = pos - tspan // 2
            t_end = t_start + tspan
            data_sim[t_start:t_end, :] += insert_data_i.T
            logging.debug(f'{start_idx=} {pos=} {class_idx=}')

            events['event_idx'] += [int(idx)]
            events['pos'] += [int(pos)]
            events['step'] += [int(step)]
            events['class_idx'] += [int(class_idx)]
            events['span'] += [int(insert_data_i.shape[-1])]
            events['jitter'] += [int(smp_jitter)]

            # increment pos to select position of next reactivation event
            smp_jitter = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
            pos += lag + smp_jitter  # add next sequence step

    if return_onsets:
        df_onsets = pd.DataFrame(events)
        df_onsets['n_events'] = n_events
        return data_sim, df_onsets

    return data_sim


def create_travelling_wave(
    hz: float,
    seconds: float,
    sfreq: int,
    chs_pos: ArrayLike,
    source_idx: int = 0,
    speed: float = 50,
) -> NDArray[np.float64]:
    """
    Create a sinus wave of shape (size, len(sensor_pos)), where each
    entry in the second dimension is phase shifted according to propagation
    speed and the euclidean distance between sensor positions.

    Parameters
    ----------
    hz : float
        The frequency of the sinus curve in Hz.
    sfreq : int
        The sampling rate of the signal in Hz.
    chs_pos : np.array or list
        A list of 2d sensor/channel positions [(x, y), ...], with coordinates
        given in cm. Phase shift of the wave will be calculated according to
        the euclidean distance between sensors/channels.
    source_idx : int, optional
        Index of the sensor/channel at which the oscillation should start
        with phase 0 and travel from there to all other positions.
    speed : float, optional
        Speed of wave in m/second. The default is 0.5m/second which is
        a good average for alpha waves.

    Returns
    -------
    wave : np.ndarray
        Array of shape (size, len(sensor_pos)) representing the travelling wave.
    """
    if speed == 0 :
        speed = np.inf

    speed = speed * 100  # convert to cm/s, as positions are in cm
    # Convert sensor_pos to a numpy array if it's not already
    chs_pos = np.array(chs_pos)

    # Number of sensors
    n_sensors = len(chs_pos)
    size = int(seconds * sfreq)

    # Time array
    t = np.arange(size) / sfreq

    # Initialize wave array
    wave = np.zeros((size, n_sensors))

    # Calculate distances from the source sensor to all other sensors
    distances = np.linalg.norm(chs_pos - chs_pos[source_idx], axis=1)

    # Calculate the time delays for each sensor
    time_delays = distances / (speed)

    # Generate the sinusoidal wave for each sensor with the corresponding phase shift
    for i in range(n_sensors):
        wave[:, i] = np.sin(2 * np.pi * hz * (t - time_delays[i]))

    return wave
