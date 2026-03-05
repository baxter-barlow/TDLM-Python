import os
import pathlib
import sys

import numpy as np
import pytest
from scipy import io

from tdlm.core import _cross_correlation


pytestmark = pytest.mark.matlab_engine_optional


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MATLAB_CODE_DIR = SCRIPT_DIR / "matlab_code"
sys.path.append(str(MATLAB_CODE_DIR))


def _enabled() -> bool:
    return os.getenv("TDLM_ENABLE_MATLAB_ENGINE_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


@pytest.mark.skipif(not _enabled(), reason="set TDLM_ENABLE_MATLAB_ENGINE_TESTS=1 to run optional MATLAB engine tests")
def test_matlab_engine_crosscorr_matches_reference():
    matlab = pytest.importorskip("matlab")
    pytest.importorskip("matlab.engine")

    from matlab_funcs import get_matlab_engine

    params = io.loadmat(MATLAB_CODE_DIR / "sequenceness_crosscorr_params.mat")
    rd = params["rd"]
    tf = params["T"]

    ml = get_matlab_engine()
    ml.cd(str(MATLAB_CODE_DIR))

    rd_ml = matlab.double(rd.tolist())
    tf_ml = matlab.double(tf.tolist())
    tb_ml = matlab.double(tf.T.tolist())

    sf_ml = np.array([ml.sequenceness_Crosscorr(rd_ml, tf_ml, [], lag) for lag in range(30)], dtype=float)
    sb_ml = np.array([ml.sequenceness_Crosscorr(rd_ml, tb_ml, [], lag) for lag in range(30)], dtype=float)

    sf_py, sb_py = _cross_correlation(rd, tf, tf.T, max_lag=30)
    np.testing.assert_allclose(sf_ml - sb_ml, sf_py - sb_py, rtol=1e-3, atol=1e-3)
