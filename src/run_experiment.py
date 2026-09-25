from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PRIMARY_SEED = 42
REPEATED_SEEDS = (13, 29, 42, 73, 101)
CALIBRATION_FOLDS = 5
UCI_DATASET_ID = 222
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_DOI = "10.24432/C5K306"
DATA_LICENSE = "CC BY 4.0"


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
    total = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            total += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(total)


def normalize_target(raw) -> pd.Series:
    series = pd.Series(raw).astype(str).str.strip().str.lower()
    values = series.map({"yes": 1, "no": 0})
    if values.isna().any():
        bad = sorted(series[values.isna()].unique().tolist())
        raise ValueError(f"Unexpected target labels: {bad}")
    return values.astype(int)


def _extract_bank_full(payload: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        names = set(outer.namelist())
        direct = next((n for n in names if n.endswith("bank-full.csv")), None)
        if direct:
            return outer.read(direct)
        nested = next((n for n in names if n.endswith("bank.zip")), None)
        if not nested:
            raise FileNotFoundError("bank-full.csv or bank.zip not found in UCI archive")
        with zipfile.ZipFile(io.BytesIO(outer.read(nested))) as inner:
            inner_name = next((n for n in inner.namelist() if n.endswith("bank-full.csv")), None)
            if not inner_name:
                raise FileNotFoundError("bank-full.csv not found in nested bank.zip")
            return inner.read(inner_name)


def load_real_data(data_path: str | Path | None = None, cache_dir: str | Path = "data/cache"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_csv = cache_dir / "bank-full.csv"

    if data_path is not None:
        csv_path = Path(data_path)
        csv_bytes = csv_path.read_bytes()
        source = f"local:{csv_path}"
    elif cached_csv.exists():
        csv_bytes = cached_csv.read_bytes()
        source = f"cache:{cached_csv}"
    else:
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            payload = response.read()
        csv_bytes = _extract_bank_full(payload)
        cached_csv.write_bytes(csv_bytes)
        source = DATA_URL

    frame = pd.read_csv(io.BytesIO(csv_bytes), sep=";")
    if "y" not in frame.columns:
        raise ValueError("Expected UCI Bank Marketing target column 'y'")
    y = normalize_target(frame.pop("y"))
    X = frame
    metadata = {
        "name": "UCI Bank Marketing (bank-full.csv)",
        "uci_id": UCI_DATASET_ID,
        "doi": DATA_DOI,
        "license": DATA_LICENSE,
        "source": source,
        "canonical_source": DATA_URL,
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
    }
    return X, y, metadata


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


def base_estimator(X: pd.DataFrame, seed: int = PRIMARY_SEED) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(X)),
        ("model", LogisticRegression(max_iter=4000, random_state=seed, solver="lbfgs")),
    ])


def build_models(X: pd.DataFrame, seed: int = PRIMARY_SEED, calibration_cv: int = CALIBRATION_FOLDS):
    base = base_estimator(X, seed)
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_uncalibrated": clone(base),
        "logistic_sigmoid": CalibratedClassifierCV(estimator=clone(base), method="sigmoid", cv=calibration_cv),
        "logistic_isotonic": CalibratedClassifierCV(estimator=clone(base), method="isotonic", cv=calibration_cv),
    }


def evaluate_probabilities(y_true, probability) -> dict[str, float]:
    p = np.asarray(probability, dtype=float)
    prediction = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "average_precision": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "ece_10": expected_calibration_error(y_true, p, 10),
    }


def error_analysis(y_true, probability) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    wrong = pred != y
    confidence = np.where(pred == 1, p, 1 - p)
    high_conf_wrong = wrong & (confidence >= 0.80)
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "error_rate": float(wrong.mean()),
        "high_confidence_errors_ge_0_80": int(high_conf_wrong.sum()),
        "mean_probability_false_positive": float(p[(y == 0) & (pred == 1)].mean()) if fp else None,
        "mean_probability_false_negative": float(p[(y == 1) & (pred == 0)].mean()) if fn else None,
    }


def run_split(X: pd.DataFrame, y: pd.Series, seed: int, calibration_cv: int = CALIBRATION_FOLDS):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    metrics, probabilities = {}, {}
    for name, model in build_models(X_train, seed=seed, calibration_cv=calibration_cv).items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        metrics[name] = evaluate_probabilities(y_test, probability)
        probabilities[name] = probability
    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
        "probabilities": probabilities,
        "y_test": np.asarray(y_test, dtype=int),
    }


def summarize_repeated(records: list[dict]) -> dict:
    model_names = records[0]["metrics"].keys()
    metric_names = records[0]["metrics"][next(iter(model_names))].keys()
    summary = {}
    for model in model_names:
        summary[model] = {}
        for metric in metric_names:
            values = np.asarray([r["metrics"][model][metric] for r in records], dtype=float)
            summary[model][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def paired_seed_differences(records: list[dict], reference: str = "logistic_uncalibrated") -> dict:
    rng = np.random.default_rng(20260925)
    metrics = ("roc_auc", "average_precision", "brier", "log_loss", "ece_10")
    out = {}
    for model in ("logistic_sigmoid", "logistic_isotonic"):
        out[model] = {}
        for metric in metrics:
            deltas = np.asarray([
                r["metrics"][model][metric] - r["metrics"][reference][metric] for r in records
            ])
            boot = []
            for _ in range(4000):
                idx = rng.integers(0, len(deltas), size=len(deltas))
                boot.append(float(deltas[idx].mean()))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out[model][metric] = {
                "mean_delta": float(deltas.mean()),
                "std_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "bootstrap_95_ci_of_mean_delta": [float(lo), float(hi)],
                "note": "Descriptive paired-split uncertainty; repeated holdouts are not independent, so no p-value is reported.",
            }
    return out


def ece_bin_sensitivity(y_true, probabilities: dict[str, np.ndarray], bins: Iterable[int] = (5, 10, 20)) -> dict:
    return {
        model: {str(n): expected_calibration_error(y_true, p, n) for n in bins}
        for model, p in probabilities.items()
    }


def calibration_cv_sensitivity(X: pd.DataFrame, y: pd.Series, seed: int = PRIMARY_SEED) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    out, base = {}, base_estimator(X_train, seed)
    for folds in (3, 5, 10):
        out[str(folds)] = {}
        for method in ("sigmoid", "isotonic"):
            model = CalibratedClassifierCV(estimator=clone(base), method=method, cv=folds)
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test)[:, 1]
            out[str(folds)][method] = evaluate_probabilities(y_test, p)
    return out


def feature_ablation(X: pd.DataFrame, y: pd.Series, seed: int = PRIMARY_SEED) -> dict:
    if "duration" not in X.columns:
        return {"status": "skipped", "reason": "duration column not present"}
    return {
        "purpose": "Operational sensitivity: call duration is only known after a call is underway/completed.",
        "with_duration": run_split(X, y, seed)["metrics"],
        "without_duration": run_split(X.drop(columns=["duration"]), y, seed)["metrics"],
    }


def _serializable_split(split: dict) -> dict:
    return {
        "seed": split["seed"],
        "n_train": split["n_train"],
        "n_test": split["n_test"],
        "metrics": split["metrics"],
    }


def write_figures(primary: dict, results_dir: Path) -> None:
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    y = primary["y_test"]

    plt.figure(figsize=(7, 5))
    for name, probability in primary["probabilities"].items():
        observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="uniform")
        plt.plot(predicted, observed, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("UCI Bank Marketing: calibration on primary split")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / "calibration_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, probability in primary["probabilities"].items():
        fpr, tpr, _ = roc_curve(y, probability)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("UCI Bank Marketing: ROC on primary split")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / "roc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, probability in primary["probabilities"].items():
        precision, recall, _ = precision_recall_curve(y, probability)
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("UCI Bank Marketing: precision-recall on primary split")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figures / "precision_recall_curve.png", dpi=180)
    plt.close()


def write_repeated_csv(records: list[dict], path: Path) -> None:
    rows = []
    for record in records:
        for model, metrics in record["metrics"].items():
            rows.append({"seed": record["seed"], "model": model, **metrics})
    pd.DataFrame(rows).to_csv(path, index=False)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def build_summary_markdown(dataset: dict, summary: dict, deltas: dict) -> str:
    lines = [
        "# Empirical Results Summary", "",
        "This file is generated by \`src/run_experiment.py\`. Do not hand-edit numerical results.", "",
        "## Dataset", "",
        f"- UCI Bank Marketing, dataset {dataset['uci_id']}",
        f"- n = {dataset['n_samples']:,}; predictors = {dataset['n_features']}",
        f"- positive rate = {_fmt(dataset['positive_rate'])}",
        f"- SHA-256 of \`bank-full.csv\`: \`{dataset['sha256']}\`", "",
        "## Repeated stratified holdout results", "",
        "Five fixed 80/20 stratified splits are used (seeds 13, 29, 42, 73, 101). Values are mean ± sample SD across splits.", "",
        "| Model | ROC-AUC | Avg precision | Brier ↓ | Log loss ↓ | ECE-10 ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, m in summary.items():
        lines.append(
            f"| {model} | {_fmt(m['roc_auc']['mean'])} ± {_fmt(m['roc_auc']['std'])} | "
            f"{_fmt(m['average_precision']['mean'])} ± {_fmt(m['average_precision']['std'])} | "
            f"{_fmt(m['brier']['mean'])} ± {_fmt(m['brier']['std'])} | "
            f"{_fmt(m['log_loss']['mean'])} ± {_fmt(m['log_loss']['std'])} | "
            f"{_fmt(m['ece_10']['mean'])} ± {_fmt(m['ece_10']['std'])} |"
        )
    lines += [
        "", "## Paired calibration deltas versus uncalibrated logistic regression", "",
        "Positive deltas are better for ROC-AUC and average precision; negative deltas are better for Brier, log loss and ECE. Intervals are descriptive bootstrap intervals over five paired split-level deltas, not inferential p-values.", "",
    ]
    for model, metrics in deltas.items():
        lines += [f"### {model}", ""]
        for metric, d in metrics.items():
            lo, hi = d["bootstrap_95_ci_of_mean_delta"]
            lines.append(f"- {metric}: mean Δ {_fmt(d['mean_delta'])}, 95% bootstrap interval [{_fmt(lo)}, {_fmt(hi)}]")
        lines.append("")
    lines += [
        "## Operational feature ablation", "",
        "\`duration\` is useful for retrospective prediction but is not available before a marketing call is made. The experiment therefore reports a second evaluation with \`duration\` removed. See \`metrics.json\` for the full metric table.", "",
        "## Interpretation boundary", "",
        "Results describe this historical UCI dataset and this protocol only. They do not establish causal effects, present-day population validity, or suitability for consequential banking decisions.", "",
    ]
    return "\n".join(lines)


def run_experiment(results_dir: str | Path = "results", data_path: str | Path | None = None, quick: bool = False):
    X, y, dataset = load_real_data(data_path=data_path)
    seeds = (PRIMARY_SEED,) if quick else REPEATED_SEEDS
    records = [run_split(X, y, seed) for seed in seeds]
    primary = next(r for r in records if r["seed"] == PRIMARY_SEED)

    repeated_summary = summarize_repeated(records)
    deltas = paired_seed_differences(records)
    results = {
        "research_bundle": True,
        "status": "complete" if not quick else "quick_smoke_run",
        "dataset": dataset,
        "protocol": {
            "primary_seed": PRIMARY_SEED,
            "repeated_seeds": list(seeds),
            "test_fraction": 0.20,
            "stratified": True,
            "calibration_cv": CALIBRATION_FOLDS,
            "ece_bins_primary": 10,
            "models": list(primary["metrics"].keys()),
        },
        "primary_split": _serializable_split(primary),
        "repeated_splits": [_serializable_split(r) for r in records],
        "repeated_summary": repeated_summary,
        "paired_deltas_vs_uncalibrated": deltas,
        "ece_bin_sensitivity": ece_bin_sensitivity(primary["y_test"], primary["probabilities"]),
        "calibration_cv_sensitivity": {"status": "skipped_in_quick_mode"} if quick else calibration_cv_sensitivity(X, y),
        "duration_ablation": {"status": "skipped_in_quick_mode"} if quick else feature_ablation(X, y),
        "error_analysis_primary_split": {
            name: error_analysis(primary["y_test"], probability)
            for name, probability in primary["probabilities"].items()
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_repeated_csv(records, out / "repeated_runs.csv")
    write_figures(primary, out)
    summary_md = build_summary_markdown(dataset, repeated_summary, deltas)
    (out / "summary.md").write_text(summary_md, encoding="utf-8")

    paper_results = Path("paper/results.md")
    paper_results.parent.mkdir(parents=True, exist_ok=True)
    paper_results.write_text("# Results\n\n" + summary_md.replace("# Empirical Results Summary\n\n", "", 1), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UCI Bank Marketing calibration study")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-path", default=None, help="Optional local bank-full.csv path")
    parser.add_argument("--quick", action="store_true", help="Primary split only; skips expensive sensitivity analyses")
    args = parser.parse_args()
    result = run_experiment(results_dir=args.results_dir, data_path=args.data_path, quick=args.quick)
    print(json.dumps({"status": result["status"], "dataset": result["dataset"]}, indent=2))


if __name__ == "__main__":
    main()
