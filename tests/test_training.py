from pathlib import Path

import pytest


def test_dummy_training_entrypoint_import():
    """
    Smoke test that the training script is importable.
    Real training is executed in CI with data present.
    """
    from src import train  # noqa: F401

    assert True


