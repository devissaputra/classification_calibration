from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from ucimlrepo import fetch_ucirepo
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
UCI_DATASET_ID = 222

def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("Expected nonempty, equal-length one-dimensional arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain binary labels 0 or 1")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=True)
    return float(sum(
        mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
        for bin_id in range(n_bins)
        if np.any(mask := (bin_ids == bin_id))
    ))

def normalize_target(raw) -> pd.Series:
    series = pd.Series(raw).astype(str).str.strip().str.lower()
    values = series.map({"yes": 1, "no": 0})
    if values.isna().any():
        bad = sorted(series[values.isna()].unique().tolist())
        raise ValueError(f"Unexpected target labels: {bad}")
    return values.astype(int)

def load_real_data():
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    X = dataset.data.features.copy()
    target = dataset.data.targets
    y = normalize_target(target.iloc[:, 0] if hasattr(target, "iloc") else target)
    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])

def base_estimator(X: pd.DataFrame, seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("model", LogisticRegression(max_iter=4000, random_state=seed)),
    ])

def build_models(X: pd.DataFrame, seed: int = SEED):
    base = base_estimator(X, seed)
    return {
        "uncalibrated": clone(base),
        "sigmoid": CalibratedClassifierCV(estimator=clone(base), method="sigmoid", cv=5),
        "isotonic": CalibratedClassifierCV(estimator=clone(base), method="isotonic", cv=5),
    }

def evaluate_model(model, X_test, y_test):
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "brier": float(brier_score_loss(y_test, probability)),
        "log_loss": float(log_loss(y_test, probability)),
        "ece_10": expected_calibration_error(y_test, probability, 10),
    }, probability

def run_experiment(results_dir: str | Path = "results", seed: int = SEED, make_plots: bool = True):
    X, y = load_real_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    models = build_models(X_train, seed)
    results = {
        "research_bundle": True,
        "dataset": {"name": "UCI Bank Marketing", "uci_id": UCI_DATASET_ID,
                    "doi": "10.24432/C5K306", "n_samples": int(len(X)),
                    "positive_rate": float(y.mean())},
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": {},
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__},
    }
    probabilities = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, probability = evaluate_model(model, X_test, y_test)
        results["models"][name] = metrics
        probabilities[name] = probability

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    if make_plots:
        figures = out / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 5))
        for name, probability in probabilities.items():
            observed, predicted = calibration_curve(y_test, probability, n_bins=10)
            plt.plot(predicted, observed, marker="o", label=name)
        plt.plot([0, 1], [0, 1], "--", label="ideal")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed positive fraction")
        plt.title("UCI Bank Marketing: calibration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "calibration_curve.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 5))
        for name, probability in probabilities.items():
            fpr, tpr, _ = roc_curve(y_test, probability)
            plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("UCI Bank Marketing: ROC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "roc_curve.png", dpi=160)
        plt.close()
    return results

if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
