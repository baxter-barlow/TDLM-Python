from pathlib import Path

import mat73
import matplotlib
import numpy as np
import pytest

import tdlm


matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parent
PREDS_FIXTURE = ROOT / "data" / "preds.mat"
REPLAY_FIXTURE = ROOT / "matlab_code" / "simulate_replay_results.mat"


def _load_preds_fixture() -> np.ndarray:
    payload = mat73.loadmat(PREDS_FIXTURE)
    return np.asarray(payload["preds"], dtype=float)


def _load_replay_fixture() -> tuple[np.ndarray, np.ndarray]:
    payload = mat73.loadmat(REPLAY_FIXTURE)
    probas = np.asarray(payload["preds"], dtype=float)
    tf = np.asarray(payload["TF"], dtype=float)
    return probas, tf


def test_realdata_smoke_compute_plot_reproducible() -> None:
    probas = _load_preds_fixture()
    probas = probas[:1200, :]
    tf = np.roll(np.eye(probas.shape[1]), 1, axis=1)

    sf1, sb1 = tdlm.compute_1step(probas, tf, n_shuf=20, max_lag=15, rng=101)
    sf2, sb2 = tdlm.compute_1step(probas, tf, n_shuf=20, max_lag=15, rng=101)

    np.testing.assert_allclose(sf1, sf2, equal_nan=True)
    np.testing.assert_allclose(sb1, sb2, equal_nan=True)

    ax = tdlm.plotting.plot_sequenceness(sf1, sb1, which=["fwd-bkw"], plotsignflip=False)
    assert ax is not None

    diff = sf1[0] - sb1[0]
    assert diff.shape == (16,)
    assert np.isfinite(np.nanmean(diff))


@pytest.mark.integration_slow
def test_realdata_full_pipeline_contracts() -> None:
    probas, tf = _load_replay_fixture()
    probas = probas[:3000, :]

    # 1-step and 2-step
    sf1, sb1 = tdlm.compute_1step(probas, tf, n_shuf=15, max_lag=20, rng=7)
    sf2, sb2 = tdlm.compute_2step(probas, tf, n_shuf=10, max_lag=15, rng=7)
    assert sf1.shape == (15, 21)
    assert sb1.shape == (15, 21)
    assert sf2.shape == (10, 16)
    assert sb2.shape == (10, 16)

    # cross-correlation reproducibility
    ccf1, ccb1 = tdlm.sequenceness_crosscorr(probas, tf, n_shuf=12, max_lag=20, rng=123)
    ccf2, ccb2 = tdlm.cross_correlation(probas, tf, n_shuf=12, max_lag=20, rng=123)
    np.testing.assert_allclose(ccf1, ccf2, equal_nan=True)
    np.testing.assert_allclose(ccb1, ccb2, equal_nan=True)

    # windowed contracts
    win = tdlm.compute_windowed(probas, tf, win_size=600, step_size=300, n_shuf=8, max_lag=12, rng=5)
    assert win.forward_sequenceness.ndim == 3
    assert win.backward_sequenceness.ndim == 3
    assert win.window_values.ndim == 1
    assert win.forward_sequenceness.dtype != object

    # per-trial contracts
    probas_trials = np.stack([probas[0:900], probas[200:1100], probas[400:1300]], axis=0)
    sf_trials, sb_trials = tdlm.compute_1step_per_trial(
        probas_trials,
        [tf, tf, tf],
        n_shuf=6,
        max_lag=10,
        rng=99,
    )
    assert sf_trials.shape == (3, 6, 11)
    assert sb_trials.shape == (3, 6, 11)

    # signflip test reproducibility
    sx = sf1[0] - sb1[0]
    sx_group = np.vstack([sx, sx * 0.95, sx * 1.05])
    result1 = tdlm.signflip_test(sx_group, n_perms=2000, rng=42)
    result2 = tdlm.signflip_test(sx_group, n_perms=2000, rng=42)
    assert result1.pvalue == result2.pvalue
    np.testing.assert_allclose(result1.t_perms, result2.t_perms)
