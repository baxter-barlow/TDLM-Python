# TDLM Performance Regression Harness

This directory contains a CLI benchmark used for CI performance monitoring of core TDLM compute paths.

## Commands

```bash
python tdlm/tests/benchmarks/run_perf_regression.py --mode calibrate --baseline tdlm/tests/benchmarks/baseline_ubuntu_py310.json
python tdlm/tests/benchmarks/run_perf_regression.py --mode report --baseline tdlm/tests/benchmarks/baseline_ubuntu_py310.json --threshold 0.20
python tdlm/tests/benchmarks/run_perf_regression.py --mode enforce --baseline tdlm/tests/benchmarks/baseline_ubuntu_py310.json --threshold 0.20
```

## Modes

- `calibrate`: writes a new baseline file.
- `report`: compares current medians to baseline and prints metrics; never exits non-zero for regression.
- `enforce`: compares current medians to baseline and exits non-zero if any metric exceeds the allowed slowdown threshold.

## Metrics

- `compute_1step(max_lag=40, n_shuf=50)`
- `compute_2step(max_lag=30, n_shuf=20)`
- `sequenceness_crosscorr(max_lag=40, n_shuf=100)`

## Threshold source

If `--threshold` is omitted, the script reads `TDLM_PERF_THRESHOLD`, defaulting to `0.20`.

## Baseline provenance policy

- Baseline target defaults to `Linux` + Python `3.10` (override via `TDLM_PERF_EXPECTED_OS` and `TDLM_PERF_EXPECTED_PYTHON`).
- Baseline metadata must include `meta.target` and `meta.calibration_runtime`.
- In `report` mode, baseline target mismatches are reported as warnings.
- In `enforce` mode, baseline target mismatches fail the run (or set `TDLM_PERF_STRICT_BASELINE_META=1` to enforce this in any mode).
- Recalibrate baseline intentionally on the CI target runner before accepting updates to `baseline_ubuntu_py310.json`.
