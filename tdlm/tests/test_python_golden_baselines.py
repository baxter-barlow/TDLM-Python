from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import tdlm
import tdlm.utils as tdlm_utils


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "python_golden"


def _load_fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURE_DIR / name, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def test_golden_compute_1step_vanilla() -> None:
    fixture = _load_fixture("compute_1step_vanilla.npz")
    unique_perms = fixture["unique_perms"]

    with patch("tdlm.core.unique_permutations", lambda *args, **kwargs: unique_perms):
        sf, sb = tdlm.compute_1step(
            fixture["probas"],
            fixture["tf"],
            max_lag=int(fixture["max_lag"]),
            n_shuf=int(fixture["n_shuf"]),
        )

    np.testing.assert_allclose(sf, fixture["sf"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sb, fixture["sb"], rtol=1e-8, atol=1e-10)


def test_golden_compute_1step_alpha() -> None:
    fixture = _load_fixture("compute_1step_alpha.npz")
    unique_perms = fixture["unique_perms"]

    with patch("tdlm.core.unique_permutations", lambda *args, **kwargs: unique_perms):
        sf, sb = tdlm.compute_1step(
            fixture["probas"],
            fixture["tf"],
            max_lag=int(fixture["max_lag"]),
            n_shuf=int(fixture["n_shuf"]),
            alpha_freq=int(fixture["alpha_freq"]),
        )

    np.testing.assert_allclose(sf, fixture["sf"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(sb, fixture["sb"], rtol=1e-8, atol=1e-10)


def test_golden_compute_2step_longerlength() -> None:
    fixture = _load_fixture("compute_2step_longerlength.npz")
    unique_perms = fixture["unique_perms"]

    with patch("tdlm.core.unique_permutations", lambda *args, **kwargs: unique_perms):
        sf, sb = tdlm.compute_2step(
            fixture["probas"],
            fixture["tf"],
            max_lag=int(fixture["max_lag"]),
            n_shuf=int(fixture["n_shuf"]),
        )

    np.testing.assert_allclose(sf, fixture["sf"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(sb, fixture["sb"], rtol=1e-7, atol=1e-9)


def test_python_only_contracts() -> None:
    removed_alias = "sign" + "flit_test"
    removed_helper = "_unique_permutations_" + "MATLAB"
    assert not hasattr(tdlm, removed_alias)

    with pytest.raises(AttributeError):
        getattr(tdlm_utils, removed_helper)
