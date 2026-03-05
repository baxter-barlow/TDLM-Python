import numpy as np
import pytest

import tdlm


N_STATES = 5
MAX_LAG = 12
N_SUBJECTS = 24
N_PERMS = 2000
ALPHA = 0.05


def _transition_matrix():
    return np.roll(np.eye(N_STATES), 1, axis=1)


def _generate_subject_probas(rng, n_time=500, lag=3, signal_strength=0.0):
    probas = rng.standard_normal((n_time, N_STATES))
    if signal_strength > 0:
        latent = rng.standard_normal(n_time)
        for state_idx in range(N_STATES):
            probas[:, state_idx] += signal_strength * np.roll(latent, state_idx * lag)
    return probas


def _run_group_experiment(seed, signal_strength):
    rng = np.random.default_rng(seed)
    tf = _transition_matrix()
    sx_rows = []

    for _ in range(N_SUBJECTS):
        probas = _generate_subject_probas(rng, signal_strength=signal_strength)
        sf, sb = tdlm.compute_1step(probas, tf, n_shuf=1, max_lag=MAX_LAG, rng=rng)
        sx_rows.append(sf[0] - sb[0])

    sx_group = np.vstack(sx_rows)
    return tdlm.signflip_test(sx_group, n_perms=N_PERMS, rng=seed)


def test_stats_smoke_signal_detectable():
    result = _run_group_experiment(seed=7001, signal_strength=0.25)
    assert result.pvalue < ALPHA


def test_stats_smoke_null_not_trivially_failing():
    pvals = [_run_group_experiment(seed=6100 + i, signal_strength=0.0).pvalue for i in range(20)]
    false_positive_rate = np.mean(np.array(pvals) < ALPHA)
    assert false_positive_rate <= 0.25


@pytest.mark.statistical_slow
def test_type1_error_rate_strict():
    n_experiments = 120
    pvals = [_run_group_experiment(seed=6000 + i, signal_strength=0.0).pvalue for i in range(n_experiments)]
    false_positive_rate = np.mean(np.array(pvals) < ALPHA)
    assert 0.03 <= false_positive_rate <= 0.07


@pytest.mark.statistical_slow
def test_power_strict():
    n_experiments = 120
    pvals = [_run_group_experiment(seed=6000 + i, signal_strength=0.16).pvalue for i in range(n_experiments)]
    power = np.mean(np.array(pvals) < ALPHA)
    assert power >= 0.80
