import pathlib
from unittest.mock import patch

import mat73
import numpy as np

import tdlm


BASE_DIR = pathlib.Path(__file__).resolve().parent / "matlab_code"


def _load_fixture(name: str):
    return mat73.loadmat(BASE_DIR / name)


def _matlab_perms_to_python(unique_perms):
    return np.asarray(unique_perms, dtype=int) - 1


def test_matlab_parity_fixture_1step_vanilla():
    data = _load_fixture("simulate_replay_results.mat")
    preds = np.asarray(data["preds"], dtype=float)
    tf = np.asarray(data["TF"], dtype=float)
    sf_matlab = np.asarray(data["sf"], dtype=float).squeeze()
    sb_matlab = np.asarray(data["sb"], dtype=float).squeeze()
    unique_perms = _matlab_perms_to_python(data["uniquePerms"])

    with patch("tdlm.core.unique_permutations", lambda *args, **kwargs: unique_perms):
        sf, sb = tdlm.compute_1step(preds, tf, max_lag=60, n_shuf=100)

    np.testing.assert_allclose(sf_matlab, sf, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(sb_matlab, sb, rtol=1e-7, atol=1e-8)


def test_matlab_parity_fixture_1step_alpha():
    data = _load_fixture("simulate_replay_withalpha_results.mat")
    preds = np.asarray(data["preds"], dtype=float)
    tf = np.asarray(data["TF"], dtype=float)
    sf_matlab = np.asarray(data["sf"], dtype=float).squeeze()
    sb_matlab = np.asarray(data["sb"], dtype=float).squeeze()
    unique_perms = _matlab_perms_to_python(data["uniquePerms"])

    with patch("tdlm.core.unique_permutations", lambda *args, **kwargs: unique_perms):
        sf, sb = tdlm.compute_1step(preds, tf, max_lag=60, n_shuf=100, alpha_freq=10)

    np.testing.assert_allclose(sf_matlab, sf, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(sb_matlab, sb, rtol=1e-7, atol=1e-8)


def test_matlab_parity_fixture_2step_longerlength():
    data = _load_fixture("simulate_replay_longerlength_results.mat")
    preds = np.asarray(data["preds"], dtype=float)
    tf = np.asarray(data["TF"], dtype=float)
    sf_matlab = np.asarray(data["sf1"], dtype=float).squeeze()
    sb_matlab = np.asarray(data["sb1"], dtype=float).squeeze()
    max_lag = int(data["maxLag"])
    n_shuf = int(data["nShuf"])
    unique_perms = _matlab_perms_to_python(data["uniquePerms"])

    with patch("tdlm.core.unique_permutations", lambda *args, **kwargs: unique_perms):
        sf, sb = tdlm.compute_2step(preds, tf, max_lag=max_lag, n_shuf=n_shuf)

    np.testing.assert_allclose(sf_matlab, sf, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(sb_matlab, sb, rtol=1e-6, atol=1e-7)
