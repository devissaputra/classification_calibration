from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.run_experiment import (
    build_preprocessor,
    error_analysis,
    evaluate_probabilities,
    expected_calibration_error,
    load_real_data,
    normalize_target,
    summarize_repeated,
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
    assert np.isclose(expected_calibration_error([0, 1], [0.25, 0.75], 2), 0.25)


def test_ece_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        expected_calibration_error([], [])
    with pytest.raises(ValueError):
        expected_calibration_error([0], [np.nan])
    with pytest.raises(ValueError):
        expected_calibration_error([0], [1.2])


def test_preprocessor_fits_mixed_schema():
    X = pd.DataFrame({"age": [20, 30, 40], "job": ["a", "b", "a"]})
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape[0] == 3


def test_local_loader_and_hash(tmp_path):
    csv = tmp_path / "bank-full.csv"
    csv.write_text(
        "age;job;duration;y\n20;a;30;no\n30;b;60;yes\n40;a;90;no\n",
        encoding="utf-8",
    )
    X, y, meta = load_real_data(data_path=csv, cache_dir=tmp_path / "cache")
    assert X.shape == (3, 3)
    assert y.tolist() == [0, 1, 0]
    assert meta["n_samples"] == 3
    assert len(meta["sha256"]) == 64


def test_probability_metrics_and_error_analysis():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = evaluate_probabilities(y, p)
    assert {"accuracy", "roc_auc", "average_precision", "brier", "log_loss", "ece_10"} <= metrics.keys()
    errors = error_analysis(y, p)
    assert errors["fp"] == 1
    assert errors["fn"] == 1


def test_repeated_summary():
    records = [
        {"metrics": {"m": {"brier": 0.2, "roc_auc": 0.7}}},
        {"metrics": {"m": {"brier": 0.1, "roc_auc": 0.9}}},
    ]
    summary = summarize_repeated(records)
    assert np.isclose(summary["m"]["brier"]["mean"], 0.15)
    assert np.isclose(summary["m"]["roc_auc"]["mean"], 0.8)
