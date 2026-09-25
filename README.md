# Probability Calibration Research Bundle

[![CI](https://github.com/devissaputra/classification_calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/classification_calibration/actions/workflows/ci.yml)

**Research Bundle · AI Engineering · empirical probability calibration**

This repository is a professor-facing, reproducible empirical study of probability calibration under class imbalance. The default experiment uses the **UCI Bank Marketing** dataset rather than a packaged toy benchmark.

## Research question

> When a logistic classifier predicts subscription to a bank term deposit, how do uncalibrated, sigmoid-calibrated, and isotonic-calibrated probabilities differ in discrimination and probability quality on an untouched holdout set?

The study intentionally separates ranking quality from probability quality. A model may have strong ROC-AUC and still produce probabilities that are poorly calibrated.

## Real dataset

**UCI Bank Marketing (dataset 222)**, collected from direct-marketing campaigns of a Portuguese banking institution.

- source: UCI Machine Learning Repository
- instances: 45,211 in the full bank dataset
- mixed numerical and categorical predictors
- binary target: term-deposit subscription
- dataset license: CC BY 4.0
- DOI: 10.24432/C5K306

The dataset is fetched from UCI by the research runner through `ucimlrepo`. No copy of the source dataset is committed here. See [DATA.md](DATA.md).

## Frozen study design

1. Fetch the UCI dataset and record source metadata.
2. Normalize the binary target to `yes=1`, `no=0`.
3. Create a fixed, stratified 80/20 train/test split with seed 42.
4. Fit preprocessing **only on training data**.
5. Compare the same logistic-regression base learner in three conditions:
   - uncalibrated;
   - sigmoid calibration with five-fold cross-validation inside training data;
   - isotonic calibration with five-fold cross-validation inside training data.
6. Evaluate once on the untouched test set.
7. Report ROC-AUC, Brier score, log loss, ECE-10, and accuracy.
8. Preserve environment and dataset metadata in the result JSON.

No empirical winner is asserted in this README before the real-data experiment is run.

## Run the empirical study

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py
```

The runner writes `results/metrics.json` and, when plotting is enabled, calibration and ROC figures under `results/figures/`.

## Why this is a Research Bundle

The repository includes:

- an explicit empirical question;
- real external data with provenance and license;
- a frozen split/evaluation protocol;
- leakage-aware preprocessing;
- meaningful probabilistic baselines;
- machine-readable results;
- tests and CI;
- reproducibility documentation;
- ethics and deployment boundaries;
- a paper scaffold and software citation.

See [RESEARCH_BUNDLE.md](RESEARCH_BUNDLE.md) for the evidence contract.

## Interpretation boundary

This is a methodology study, **not a banking decision system**. The target is a historical marketing response, not customer value or eligibility. Calibration quality can change across time, campaigns, populations and acquisition channels. The study does not justify targeting, exclusion, credit decisions, or other consequential treatment of individuals.

## Repository map

```text
README.md                 research question and study overview
RESEARCH_BUNDLE.md        bundle evidence contract
DATA.md                   dataset card and provenance
REPRODUCIBILITY.md        rerun protocol
ETHICS.md                 responsible-use boundary
src/run_experiment.py     real-data empirical runner
tests/                    offline behavioural tests
results/                  generated empirical outputs
paper/                    paper-ready study scaffold
CITATION.cff              software citation
```

## Citation

Dataset: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Code and study design: see [CITATION.cff](CITATION.cff).
