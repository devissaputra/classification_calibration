from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.run_experiment import expected_calibration_error, normalize_target, build_preprocessor

def test_repository_is_research_bundle():
    root = Path(__file__).resolve().parents[1]
    for path in [
        "README.md", "RESEARCH_BUNDLE.md", "DATA.md", "REPRODUCIBILITY.md",
        "ETHICS.md", "src/run_experiment.py", "paper/paper.md",
        ".github/workflows/ci.yml"
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

def test_preprocessor_fits_mixed_schema():
    X = pd.DataFrame({"age":[20, 30, 40], "job":["a", "b", "a"]})
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape[0] == 3
