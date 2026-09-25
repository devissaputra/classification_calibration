from pathlib import Path
import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.run_experiment import (
    _extract_bank_full,
    build_models,
    build_preprocessor,
    expected_calibration_error,
    normalize_target,
    paired_seed_differences,
    validate_dataset_hash,
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
        "logistic_uncalibrated",
        "logistic_sigmoid",
        "logistic_isotonic",
    }


def test_nested_uci_archive_extraction():
    csv_bytes = b"age;job;y\n20;a;no\n30;b;yes\n"
    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w") as inner:
        inner.writestr("bank-full.csv", csv_bytes)
    outer_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_buffer, "w") as outer:
        outer.writestr("bank.zip", inner_buffer.getvalue())
    assert _extract_bank_full(outer_buffer.getvalue()) == csv_bytes


def test_paired_seed_differences_are_descriptive():
    records = []
    for i, seed in enumerate([13, 29, 42, 73, 101]):
        base = 0.10 + i * 0.001
        records.append({
            "seed": seed,
            "metrics": {
                "logistic_uncalibrated": {
                    "roc_auc": 0.80,
                    "average_precision": 0.50,
                    "brier": base,
                    "log_loss": 0.30,
                    "ece_10": 0.04,
                },
                "logistic_sigmoid": {
                    "roc_auc": 0.80,
                    "average_precision": 0.50,
                    "brier": base - 0.002,
                    "log_loss": 0.29,
                    "ece_10": 0.03,
                },
                "logistic_isotonic": {
                    "roc_auc": 0.79,
                    "average_precision": 0.49,
                    "brier": base - 0.001,
                    "log_loss": 0.295,
                    "ece_10": 0.035,
                },
            },
        })
    result = paired_seed_differences(records)
    assert set(result) == {"logistic_sigmoid", "logistic_isotonic"}
    assert result["logistic_sigmoid"]["brier"]["mean_delta"] < 0
    assert "bootstrap_95_ci_of_mean_delta" in result["logistic_sigmoid"]["brier"]


def test_frozen_dataset_hash_guard():
    payload = b"research-bundle-fixture"
    expected = hashlib.sha256(payload).hexdigest()
    assert validate_dataset_hash(payload, expected) == expected
    with pytest.raises(ValueError, match="Unexpected bank-full.csv SHA-256"):
        validate_dataset_hash(payload, "0" * 64)
