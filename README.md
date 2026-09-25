# Probability Calibration Research Bundle

[![CI](https://github.com/devissaputra/classification_calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/classification_calibration/actions/workflows/ci.yml)

**Research Bundle · AI Engineering · empirical probability calibration**

This repository is a reproducible empirical study of probability calibration under class imbalance using the real **UCI Bank Marketing** dataset. The current protocol goes beyond a single demonstration split: it includes a class-prior baseline, repeated stratified holdouts, calibration-method comparison, sensitivity analysis, an operational feature ablation, descriptive paired bootstrap intervals, error analysis, and generated figures.

## Research question

How do sigmoid and isotonic calibration change the probability quality and discrimination of logistic regression on UCI Bank Marketing, and how stable are those changes across repeated train/test splits and reasonable calibration choices?

## Real dataset

**UCI Bank Marketing, dataset 222**

- source: UCI Machine Learning Repository
- task: binary prediction of term-deposit subscription
- canonical file used by the runner: bank-full.csv
- DOI: 10.24432/C5K306
- license reported by UCI: CC BY 4.0
- official archive: downloaded directly by the runner
- provenance: the exact raw CSV SHA-256 is recorded in results/metrics.json

No copy of the source dataset is committed. See DATA.md.

## Frozen empirical design

1. Download and extract bank-full.csv from the official UCI archive, or use a user-supplied local copy.
2. Validate and normalize the binary target.
3. Use a fixed stratified 80/20 primary holdout with seed 42.
4. Fit preprocessing only inside the training pipeline.
5. Compare four conditions:
   - class-prior dummy baseline
   - uncalibrated logistic regression
   - sigmoid-calibrated logistic regression
   - isotonic-calibrated logistic regression
6. Repeat the holdout protocol with seeds 13, 29, 42, 73, and 101.
7. Report ROC-AUC, average precision, Brier score, log loss, ECE-10, and accuracy.
8. Report descriptive paired bootstrap intervals for calibrated-minus-uncalibrated split-level metric deltas.
9. Test calibration-CV sensitivity at 3, 5, and 10 folds.
10. Test ECE sensitivity at 5, 10, and 20 bins.
11. Repeat the primary experiment without duration because that feature is unavailable before a marketing call occurs.
12. Report primary-split confusion counts and high-confidence errors.
13. Generate machine-readable results, a professor-facing summary, and calibration, ROC, and precision-recall figures.

## Run the full study

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    PYTHONPATH=. pytest -q
    PYTHONPATH=. python src/run_experiment.py

For a fast smoke run using only the primary split:

    PYTHONPATH=. python src/run_experiment.py --quick

## Generated evidence

A successful full run produces:

- results/metrics.json
- results/repeated_runs.csv
- results/summary.md
- results/figures/calibration_curve.png
- results/figures/roc_curve.png
- results/figures/precision_recall_curve.png
- paper/results.md

The JSON file contains the full calibration-fold sensitivity, ECE-bin sensitivity, duration ablation, error analysis, repeated-split summary, and paired-delta uncertainty results.

## Why this qualifies as a Research Bundle

The repository contains a real external dataset with provenance, a frozen protocol, multiple baselines, repeated experiments, sensitivity checks, operational ablation, uncertainty reporting, error analysis, reproducible code, offline tests, CI, an empirical workflow, a manuscript scaffold, and explicit interpretation limits.

See RESEARCH_BUNDLE.md for the evidence contract.

## Interpretation boundary

This is a methodology study, not a banking decision system. The dataset is historical and institution-specific. The results do not establish causal effects, fairness, current-population validity, transportability, or suitability for consequential financial decisions.

## Repository map

    README.md                     study overview
    RESEARCH_BUNDLE.md            evidence contract
    DATA.md                       dataset provenance
    REPRODUCIBILITY.md            frozen rerun protocol
    ETHICS.md                     responsible-use boundary
    src/run_experiment.py         full empirical runner
    tests/test_experiment.py      offline behavioral tests
    results/metrics.json          full machine-readable evidence
    results/repeated_runs.csv     split-level metrics
    results/summary.md            generated professor-facing results
    results/figures/              generated empirical figures
    paper/paper.md                manuscript scaffold
    paper/results.md              generated paper results
    CITATION.cff                  software citation

## Citation

Dataset: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Code and study design: see CITATION.cff.
