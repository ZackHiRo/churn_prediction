from pathlib import Path

import pytest


def test_dummy_training_entrypoint_import():
    """
    Smoke test that the training script is importable.
    Real training is executed in CI with data present.
    """
    try:
        from src import train  # noqa: F401
        assert True
    except (ImportError, Exception) as e:
        # XGBoost may fail to import/load if OpenMP runtime is missing
        # This is a system dependency issue, not a code issue
        error_str = str(e).lower()
        if "xgboost" in error_str or "libxgboost" in error_str or "openmp" in error_str:
            pytest.skip(f"XGBoost import failed (likely missing OpenMP runtime): {e}")
        raise


