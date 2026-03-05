#!/usr/bin/env python3
"""Build deterministic Python golden fixtures for TDLM tests."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

import tdlm
from tdlm.utils import unique_permutations


FIXTURE_DIR = Path(__file__).resolve().parent / "python_golden"
N_STATES = 5


def _transition_matrix() -> np.ndarray:
    return np.roll(np.eye(N_STATES), 1, axis=1)


def _fixed_perms(n_shuf: int, seed: int) -> np.ndarray:
    return unique_permutations(np.arange(N_STATES), k=n_shuf, rng=seed)


def _save_1step_vanilla() -> None:
    rng = np.random.default_rng(10101)
    probas = rng.standard_normal((1200, N_STATES))
    tf = _transition_matrix()
    n_shuf = 40
    max_lag = 60
    unique_perms = _fixed_perms(n_shuf=n_shuf, seed=9101)

    with patch("tdlm.core.unique_permutations", lambda *args, **kwargs: unique_perms):
        sf, sb = tdlm.compute_1step(probas, tf, max_lag=max_lag, n_shuf=n_shuf)

    np.savez(
        FIXTURE_DIR / "compute_1step_vanilla.npz",
        probas=probas,
        tf=tf,
        unique_perms=unique_perms,
        max_lag=np.array(max_lag, dtype=int),
        n_shuf=np.array(n_shuf, dtype=int),
        sf=sf,
        sb=sb,
    )


def _save_1step_alpha() -> None:
    rng = np.random.default_rng(20202)
    probas = rng.standard_normal((1400, N_STATES))
    tf = _transition_matrix()
    n_shuf = 36
    max_lag = 60
    alpha_freq = 10
    unique_perms = _fixed_perms(n_shuf=n_shuf, seed=9202)

    with patch("tdlm.core.unique_permutations", lambda *args, **kwargs: unique_perms):
        sf, sb = tdlm.compute_1step(
            probas,
            tf,
            max_lag=max_lag,
            n_shuf=n_shuf,
            alpha_freq=alpha_freq,
        )

    np.savez(
        FIXTURE_DIR / "compute_1step_alpha.npz",
        probas=probas,
        tf=tf,
        unique_perms=unique_perms,
        max_lag=np.array(max_lag, dtype=int),
        n_shuf=np.array(n_shuf, dtype=int),
        alpha_freq=np.array(alpha_freq, dtype=int),
        sf=sf,
        sb=sb,
    )


def _save_2step_longerlength() -> None:
    rng = np.random.default_rng(30303)
    probas = rng.standard_normal((1700, N_STATES))
    tf = _transition_matrix()
    n_shuf = 20
    max_lag = 25
    unique_perms = _fixed_perms(n_shuf=n_shuf, seed=9303)

    with patch("tdlm.core.unique_permutations", lambda *args, **kwargs: unique_perms):
        sf, sb = tdlm.compute_2step(probas, tf, max_lag=max_lag, n_shuf=n_shuf)

    np.savez(
        FIXTURE_DIR / "compute_2step_longerlength.npz",
        probas=probas,
        tf=tf,
        unique_perms=unique_perms,
        max_lag=np.array(max_lag, dtype=int),
        n_shuf=np.array(n_shuf, dtype=int),
        sf=sf,
        sb=sb,
    )


def _save_integration_preds() -> None:
    rng = np.random.default_rng(40404)
    probas = rng.standard_normal((3500, N_STATES))
    tf = _transition_matrix()

    np.savez(
        FIXTURE_DIR / "integration_preds.npz",
        probas=probas,
        tf=tf,
    )


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _save_1step_vanilla()
    _save_1step_alpha()
    _save_2step_longerlength()
    _save_integration_preds()
    print(f"Wrote fixtures to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
