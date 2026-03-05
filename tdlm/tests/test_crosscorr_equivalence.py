import numpy as np

from tdlm.core import _cross_correlation


def _cross_correlation_roll_reference(probas, tf, tb, max_lag=40, min_lag=0):
    probas_f = probas @ tf
    probas_b = probas @ tb

    ff = np.zeros(max_lag - min_lag)
    fb = np.zeros(max_lag - min_lag)

    for lag in range(min_lag, max_lag):
        r = np.corrcoef(probas[lag:, :].T, np.roll(probas_f, lag, axis=0)[lag:, :].T)
        ff[lag - min_lag] = np.nanmean(np.diag(r, k=tf.shape[0]))

        r = np.corrcoef(probas[lag:, :].T, np.roll(probas_b, lag, axis=0)[lag:, :].T)
        fb[lag - min_lag] = np.nanmean(np.diag(r, k=tb.shape[0]))

    return ff, fb


def test_cross_correlation_matches_roll_reference():
    rng = np.random.default_rng(20260305)
    probas = rng.standard_normal((256, 7))
    tf = np.roll(np.eye(7), 1, axis=1)
    tb = tf.T

    expected_ff, expected_fb = _cross_correlation_roll_reference(
        probas, tf, tb, min_lag=2, max_lag=35
    )
    ff, fb = _cross_correlation(probas, tf, tb, min_lag=2, max_lag=35)

    np.testing.assert_allclose(ff, expected_ff, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fb, expected_fb, rtol=1e-12, atol=1e-12)
