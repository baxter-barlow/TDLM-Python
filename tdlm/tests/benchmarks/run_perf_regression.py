#!/usr/bin/env python3
import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import tdlm


def _default_threshold() -> float:
    return float(os.getenv("TDLM_PERF_THRESHOLD", "0.20"))


def _expected_target() -> dict:
    return {
        "os": os.getenv("TDLM_PERF_EXPECTED_OS", "Linux"),
        "python": os.getenv("TDLM_PERF_EXPECTED_PYTHON", "3.10"),
    }


def _python_matches(actual: str, expected: str) -> bool:
    return actual == expected or actual.startswith(f"{expected}.")


def _current_runtime_meta() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
    }


def _build_inputs(seed: int = 20260305):
    rng = np.random.default_rng(seed)
    probas = rng.standard_normal((3000, 8))
    tf = np.roll(np.eye(8), 1, axis=1)
    return probas, tf


def _metric_specs(probas, tf):
    return {
        "compute_1step": lambda: tdlm.compute_1step(probas, tf, max_lag=40, n_shuf=50, rng=42),
        "compute_2step": lambda: tdlm.compute_2step(probas, tf, max_lag=30, n_shuf=20, rng=42),
        "sequenceness_crosscorr": lambda: tdlm.sequenceness_crosscorr(probas, tf, max_lag=40, n_shuf=100),
    }


def _time_callable(fn, warmup_runs: int = 2, measured_runs: int = 9):
    for _ in range(warmup_runs):
        fn()

    runtimes = []
    for _ in range(measured_runs):
        t0 = time.perf_counter()
        fn()
        runtimes.append(time.perf_counter() - t0)

    return {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "runs_seconds": runtimes,
        "median_seconds": statistics.median(runtimes),
    }


def _run_benchmarks(expected_target: dict):
    runtime_meta = _current_runtime_meta()
    probas, tf = _build_inputs()
    metrics = {}
    for metric_name, metric_fn in _metric_specs(probas, tf).items():
        metrics[metric_name] = _time_callable(metric_fn)

    return {
        "meta": {
            "calibration_runtime": runtime_meta,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": runtime_meta["python"],
            "platform": runtime_meta["platform"],
            "system": runtime_meta["system"],
            "target": expected_target,
        },
        "metrics": metrics,
    }


def _load_baseline(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"baseline file not found: {path}")
    return json.loads(path.read_text())


def _extract_baseline_target(baseline: dict, expected_target: dict) -> dict:
    meta = baseline.get("meta", {})
    target = meta.get("target")
    if isinstance(target, dict) and "os" in target and "python" in target:
        return {"os": str(target["os"]), "python": str(target["python"])}

    # Backward compatibility with older baseline files.
    platform_value = str(meta.get("platform", ""))
    baseline_python = str(meta.get("python", ""))
    guessed_os = "Linux" if "Linux" in platform_value else platform.system()
    guessed_python = baseline_python.split(".")
    guessed_python = ".".join(guessed_python[:2]) if len(guessed_python) >= 2 else expected_target["python"]
    return {"os": guessed_os, "python": guessed_python}


def _validate_baseline_metadata(
    *,
    mode: str,
    baseline: dict,
    expected_target: dict,
    current_runtime: dict,
) -> dict:
    baseline_target = _extract_baseline_target(baseline, expected_target)
    baseline_meta = baseline.get("meta", {})
    calibration_runtime = baseline_meta.get("calibration_runtime")
    messages = []
    ok = True

    if not isinstance(calibration_runtime, dict):
        ok = False
        messages.append("Baseline metadata missing required 'calibration_runtime' section.")
    else:
        runtime_os = str(calibration_runtime.get("system", ""))
        runtime_python = str(calibration_runtime.get("python", ""))
        if runtime_os != expected_target["os"]:
            ok = False
            messages.append(
                f"Baseline calibration runtime OS mismatch: calibration={runtime_os}, expected={expected_target['os']}"
            )
        if not _python_matches(runtime_python, expected_target["python"]):
            ok = False
            messages.append(
                f"Baseline calibration runtime Python mismatch: calibration={runtime_python}, expected={expected_target['python']}"
            )

    if baseline_target["os"] != expected_target["os"]:
        ok = False
        messages.append(
            f"Baseline target OS mismatch: baseline={baseline_target['os']}, expected={expected_target['os']}"
        )
    if not _python_matches(baseline_target["python"], expected_target["python"]):
        ok = False
        messages.append(
            f"Baseline target Python mismatch: baseline={baseline_target['python']}, expected={expected_target['python']}"
        )

    runtime_os = current_runtime["system"]
    runtime_python = current_runtime["python"]
    if runtime_os != expected_target["os"]:
        messages.append(f"Current runtime OS differs from expected target: runtime={runtime_os}, expected={expected_target['os']}")
    if not _python_matches(runtime_python, expected_target["python"]):
        messages.append(
            f"Current runtime Python differs from expected target: runtime={runtime_python}, expected={expected_target['python']}"
        )

    strict_meta = mode == "enforce" or os.getenv("TDLM_PERF_STRICT_BASELINE_META", "0") == "1"
    for msg in messages:
        print(f"Baseline metadata check: {msg}")
    if strict_meta and not ok:
        raise ValueError(
            "Baseline metadata does not match configured CI target. "
            "Recalibrate baseline on target runner before enforce mode."
        )

    return {
        "ok": ok,
        "strict_mode": strict_meta,
        "expected_target": expected_target,
        "baseline_target": baseline_target,
        "current_runtime": current_runtime,
        "messages": messages,
    }


def _compare_against_baseline(results, baseline, threshold):
    comparisons = {}
    regressions = []
    for metric_name, current in results["metrics"].items():
        if metric_name not in baseline.get("metrics", {}):
            raise KeyError(f"Baseline missing metric '{metric_name}'")
        baseline_metric = baseline["metrics"][metric_name]
        baseline_median = float(baseline_metric["median_seconds"])
        current_median = float(current["median_seconds"])
        ratio = current_median / baseline_median if baseline_median > 0 else float("inf")
        regression = ratio > (1.0 + threshold)

        comparisons[metric_name] = {
            "baseline_median_seconds": baseline_median,
            "current_median_seconds": current_median,
            "ratio": ratio,
            "threshold_ratio": 1.0 + threshold,
            "regression": regression,
        }
        if regression:
            regressions.append(metric_name)

    return comparisons, regressions


def _print_summary(comparisons):
    print("Metric, Baseline(s), Current(s), Ratio, Threshold, Regression")
    for metric_name, info in comparisons.items():
        print(
            f"{metric_name}, "
            f"{info['baseline_median_seconds']:.6f}, "
            f"{info['current_median_seconds']:.6f}, "
            f"{info['ratio']:.3f}x, "
            f"{info['threshold_ratio']:.3f}x, "
            f"{info['regression']}"
        )


def _write_output(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Performance regression harness for TDLM core paths")
    parser.add_argument("--mode", choices=["report", "enforce", "calibrate"], required=True)
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON")
    parser.add_argument("--threshold", type=float, default=None, help="Allowed slowdown ratio delta (default from TDLM_PERF_THRESHOLD or 0.20)")
    parser.add_argument("--output", default=None, help="Optional output JSON path for report payload")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    threshold = args.threshold if args.threshold is not None else _default_threshold()
    expected_target = _expected_target()

    results = _run_benchmarks(expected_target)

    if args.mode == "calibrate":
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        _write_output(baseline_path, results)
        print(f"Wrote baseline to {baseline_path}")
        return 0

    baseline = _load_baseline(baseline_path)
    baseline_validation = _validate_baseline_metadata(
        mode=args.mode,
        baseline=baseline,
        expected_target=expected_target,
        current_runtime=_current_runtime_meta(),
    )
    comparisons, regressions = _compare_against_baseline(results, baseline, threshold)
    payload = {
        "meta": {
            **results["meta"],
            "mode": args.mode,
            "threshold": threshold,
            "baseline_path": str(baseline_path),
            "baseline_validation": baseline_validation,
        },
        "comparisons": comparisons,
    }

    _print_summary(comparisons)
    if args.output:
        _write_output(Path(args.output), payload)

    if args.mode == "enforce" and regressions:
        print("Regression detected in metrics:", ", ".join(regressions))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
