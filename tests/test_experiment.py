from pathlib import Path

import numpy as np

from src.run_experiment import expected_calibration_error, run_experiment


def test_repository_structure():
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "DATA.md",
        "ETHICS.md",
        "REPRODUCIBILITY.md",
        "CITATION.cff",
        "src/run_experiment.py",
        "paper/paper.md",
        "paper/paper.tex",
        "assets/01_cover.svg",
        "assets/02_data_pipeline.svg",
        "assets/03_data_or_model.svg",
        "assets/04_evaluation_or_results.svg",
        ".github/workflows/ci.yml",
    ]
    for relative_path in required:
        assert (root / relative_path).exists(), relative_path


def test_perfect_probabilities_have_zero_ece():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p) == 0.0


def test_experiment_returns_valid_probability_metrics(tmp_path):
    result = run_experiment(tmp_path, make_plots=False)

    assert result["n_samples"] == 569
    assert set(result["models"]) == {"uncalibrated", "sigmoid", "isotonic"}

    for metrics in result["models"].values():
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert metrics["brier"] >= 0.0
        assert metrics["log_loss"] >= 0.0
        assert 0.0 <= metrics["ece_10"] <= 1.0

    assert (tmp_path / "metrics.json").exists()


def test_ece_rejects_invalid_inputs():
    import pytest
    for y, p, bins in [([], [], 10), ([0], [np.nan], 10), ([0], [np.inf], 10),
                       ([2], [.5], 10), ([0], [.5], 0), ([0], [.5], 2.5),
                       ([[0]], [[.5]], 10), ([0, 1], [.5], 10)]:
        with pytest.raises(ValueError):
            expected_calibration_error(y, p, bins)


def test_ece_known_weighted_error_and_boundaries():
    assert np.isclose(expected_calibration_error([0, 1], [.25, .75], 2), .25)
    assert expected_calibration_error([0, 1], [0., 1.], 10) == 0
