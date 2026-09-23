from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Return a simple equal-width expected calibration error."""
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and probabilities must have the same length")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=True)
    ece = 0.0

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        confidence = float(p[mask].mean())
        observed = float(y[mask].mean())
        ece += float(mask.mean()) * abs(observed - confidence)

    return float(ece)


def load_split(seed: int = SEED):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return (*train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    ), len(X))


def base_estimator(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=4000, random_state=seed),
            ),
        ]
    )


def build_models(seed: int = SEED):
    base = base_estimator(seed)
    return {
        "uncalibrated": clone(base),
        "sigmoid": CalibratedClassifierCV(
            estimator=clone(base),
            method="sigmoid",
            cv=5,
        ),
        "isotonic": CalibratedClassifierCV(
            estimator=clone(base),
            method="isotonic",
            cv=5,
        ),
    }


def evaluate_model(model, X_test, y_test):
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "brier": float(brier_score_loss(y_test, probability)),
        "log_loss": float(log_loss(y_test, probability)),
        "ece_10": expected_calibration_error(y_test, probability, n_bins=10),
    }, probability


def run_experiment(
    results_dir: str | Path = "results",
    seed: int = SEED,
    make_plots: bool = True,
):
    X_train, X_test, y_train, y_test, n_samples = load_split(seed)
    results = {"n_samples": int(n_samples), "seed": int(seed), "models": {}}
    probabilities = {}

    for name, model in build_models(seed).items():
        model.fit(X_train, y_train)
        metrics, probability = evaluate_model(model, X_test, y_test)
        results["models"][name] = metrics
        probabilities[name] = probability

    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    if make_plots:
        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 5))
        for name, probability in probabilities.items():
            observed, predicted = calibration_curve(
                y_test,
                probability,
                n_bins=8,
            )
            plt.plot(predicted, observed, marker="o", label=name)
        plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed fraction")
        plt.title("Probability calibration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "calibration_curve.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        for name, probability in probabilities.items():
            fpr, tpr, _ = roc_curve(y_test, probability)
            plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / "roc_curve.png", dpi=150)
        plt.close()

    return results


def main() -> None:
    print(json.dumps(run_experiment(), indent=2))


if __name__ == "__main__":
    main()
