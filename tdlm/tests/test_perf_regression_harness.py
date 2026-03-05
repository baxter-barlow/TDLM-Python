import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parent / "benchmarks" / "run_perf_regression.py"
spec = importlib.util.spec_from_file_location("run_perf_regression", MODULE_PATH)
perf = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(perf)


def test_validate_baseline_metadata_report_mode_warn_only(capsys):
    baseline = {
        "meta": {
            "target": {"os": "Darwin", "python": "3.11"},
            "calibration_runtime": {"system": "Darwin", "python": "3.11.0"},
        }
    }
    current = {"system": "Linux", "python": "3.10.14", "platform": "Linux-x86_64"}
    expected = {"os": "Linux", "python": "3.10"}

    result = perf._validate_baseline_metadata(
        mode="report",
        baseline=baseline,
        expected_target=expected,
        current_runtime=current,
    )

    assert result["ok"] is False
    assert result["strict_mode"] is False
    assert result["baseline_target"] == {"os": "Darwin", "python": "3.11"}
    assert "Baseline metadata check:" in capsys.readouterr().out


def test_validate_baseline_metadata_enforce_mode_fails():
    baseline = {
        "meta": {
            "target": {"os": "Darwin", "python": "3.11"},
            "calibration_runtime": {"system": "Darwin", "python": "3.11.0"},
        }
    }
    current = {"system": "Linux", "python": "3.10.14", "platform": "Linux-x86_64"}
    expected = {"os": "Linux", "python": "3.10"}

    with pytest.raises(ValueError, match="Baseline metadata"):
        perf._validate_baseline_metadata(
            mode="enforce",
            baseline=baseline,
            expected_target=expected,
            current_runtime=current,
        )


def test_compare_against_baseline_retains_report_vs_enforce_logic():
    results = {
        "metrics": {
            "compute_1step": {"median_seconds": 1.15},
            "compute_2step": {"median_seconds": 0.80},
        }
    }
    baseline = {
        "metrics": {
            "compute_1step": {"median_seconds": 1.00},
            "compute_2step": {"median_seconds": 1.00},
        }
    }

    comparisons, regressions = perf._compare_against_baseline(results, baseline, threshold=0.10)

    assert regressions == ["compute_1step"]
    assert comparisons["compute_1step"]["regression"] is True
    assert comparisons["compute_2step"]["regression"] is False


def test_validate_baseline_metadata_detects_calibration_runtime_mismatch(capsys):
    baseline = {
        "meta": {
            "target": {"os": "Linux", "python": "3.10"},
            "calibration_runtime": {"system": "Darwin", "python": "3.14.3"},
        }
    }
    current = {"system": "Linux", "python": "3.10.14", "platform": "Linux-x86_64"}
    expected = {"os": "Linux", "python": "3.10"}

    result = perf._validate_baseline_metadata(
        mode="report",
        baseline=baseline,
        expected_target=expected,
        current_runtime=current,
    )

    assert result["ok"] is False
    out = capsys.readouterr().out
    assert "calibration runtime OS mismatch" in out
    assert "calibration runtime Python mismatch" in out
