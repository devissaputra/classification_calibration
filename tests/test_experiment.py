from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.run_experiment import (
    build_models,
    build_preprocessor,
    dataframe_fingerprint,
    expected_calibration_error,
    normalize_target,
    paired_bootstrap_interval,
)


def test_repository_is_research_bundle():
    root = Path(__file__).resolve().parents[1]
    for path in [
        "README.md",
        "RESEARCH_BUNDLE.md",
        "DATA.md",
        "REPRODUCIBILITY.md",
        "ETHICS.md",
        "src/run_experiment.py",
        "paper/paper.md",
        ".github/workflows/ci.yml",
        ".github/workflows/empirical.yml",
    ]:
        assert (root / path).exists(), path


def test_target_normalization():
    assert normalize_target(["yes", "no", "YES"]).tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        normalize_target(["yes", "maybe"])


def test_ece_known_values():
    assert expected_calibration_error([0, 0, 1, 1], [0, 0, 1, 1]) == 0
    assert np.isclose(expected_calibration_error([0, 1], [.25, .75], 2), .25)


def test_ece_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        expected_calibration_error([], [])
    with pytest.raises(ValueError):
        expected_calibration_error([0], [np.nan])
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.2, 1.2])


def test_preprocessor_fits_mixed_schema():
    X = pd.DataFrame({"age": [20, 30, 40], "job": ["a", "b", "a"]})
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape[0] == 3


def test_model_conditions_are_explicit():
    X = pd.DataFrame({"age": [20, 30, 40, 50], "job": ["a", "b", "a", "b"]})
    assert set(build_models(X, calibration_cv=3)) == {
        "dummy_prior",
        "uncalibrated",
        "sigmoid",
        "isotonic",
    }


def test_dataframe_fingerprint_is_deterministic():
    X = pd.DataFrame({"x": [1, 2], "group": ["a", "b"]})
    y = pd.Series([0, 1])
    assert dataframe_fingerprint(X, y) == dataframe_fingerprint(X.copy(), y.copy())
    assert len(dataframe_fingerprint(X, y)) == 64


def test_paired_bootstrap_is_seeded_and_descriptive():
    a = paired_bootstrap_interval([0.1, 0.0, -0.1, 0.05, -0.05], seed=7, repeats=200)
    b = paired_bootstrap_interval([0.1, 0.0, -0.1, 0.05, -0.05], seed=7, repeats=200)
    assert a == b
    assert a["n_splits"] == 5
    assert a["ci95_low"] <= a["mean_delta"] <= a["ci95_high"]
