import numpy as np

from tdlm.core import _cross_correlation


def _crosscorr_reference(probas: np.ndarray, tf: np.ndarray, tb: np.ndarray, *, min_lag: int, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    probas_f = probas @ tf
    probas_b = probas @ tb
    ff = np.zeros(max_lag - min_lag)
    fb = np.zeros(max_lag - min_lag)

    for lag in range(min_lag, max_lag):
        x_lag = probas[lag:, :]
        f_lag = np.roll(probas_f, lag, axis=0)[lag:, :]
        b_lag = np.roll(probas_b, lag, axis=0)[lag:, :]

        r_f = np.corrcoef(x_lag.T, f_lag.T)
        ff[lag - min_lag] = np.nanmean(np.diag(r_f, k=tf.shape[0]))

        r_b = np.corrcoef(x_lag.T, b_lag.T)
        fb[lag - min_lag] = np.nanmean(np.diag(r_b, k=tb.shape[0]))

    return ff, fb


def test_crosscorr_optimized_matches_reference_tightly() -> None:
    rng = np.random.default_rng(20260305)
    probas = rng.standard_normal((512, 8))
    tf = np.roll(np.eye(8), 1, axis=1)
    tb = tf.T

    expected_ff, expected_fb = _crosscorr_reference(probas, tf, tb, min_lag=2, max_lag=40)
    ff, fb = _cross_correlation(probas, tf, tb, min_lag=2, max_lag=40)

    np.testing.assert_allclose(ff, expected_ff, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fb, expected_fb, rtol=1e-12, atol=1e-12)


def test_crosscorr_handles_constant_columns_like_reference() -> None:
    rng = np.random.default_rng(123)
    probas = rng.standard_normal((400, 6))
    probas[:, 0] = 1.0
    tf = np.roll(np.eye(6), 1, axis=1)
    tb = tf.T

    expected_ff, expected_fb = _crosscorr_reference(probas, tf, tb, min_lag=1, max_lag=30)
    ff, fb = _cross_correlation(probas, tf, tb, min_lag=1, max_lag=30)

    np.testing.assert_allclose(ff, expected_ff, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(fb, expected_fb, rtol=1e-12, atol=1e-12, equal_nan=True)
