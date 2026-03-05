import os
import pytest


def _env_enabled(var_name: str) -> bool:
    return os.getenv(var_name, "").strip().lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(config, items):
    run_statistical_slow = _env_enabled("TDLM_RUN_STATISTICAL_SLOW")

    skip_stats = pytest.mark.skip(reason="set TDLM_RUN_STATISTICAL_SLOW=1 to run statistical_slow tests")

    for item in items:
        if "statistical_slow" in item.keywords and not run_statistical_slow:
            item.add_marker(skip_stats)
