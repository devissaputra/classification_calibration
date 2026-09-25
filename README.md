# Probability Calibration Under Class Imbalance

[![CI](https://github.com/devissaputra/classification_calibration/actions/workflows/ci.yml/badge.svg)](https://github.com/devissaputra/classification_calibration/actions/workflows/ci.yml)
[![Empirical Study](https://github.com/devissaputra/classification_calibration/actions/workflows/empirical.yml/badge.svg)](https://github.com/devissaputra/classification_calibration/actions/workflows/empirical.yml)

**Research Bundle · AI Engineering · empirical probability calibration**

This repository is a reproducible empirical study of probability calibration under class imbalance using the **UCI Bank Marketing** dataset. The study separates ranking quality from probability quality and tests whether apparent calibration gains remain stable across repeated train/test splits, calibration choices, ECE binning choices, and an operational feature ablation.

## Research question

> How do uncalibrated and calibrated logistic probabilities differ in discrimination and probability quality on UCI Bank Marketing, and how stable are those differences under repeated splits and realistic sensitivity checks?

A classifier can rank observations well while still producing probabilities that are systematically too high or too low. That distinction matters whenever predicted probabilities are interpreted as response likelihoods rather than only as ranking scores.

## Real dataset

**UCI Bank Marketing (dataset 222)** contains records from direct-marketing campaigns of a Portuguese banking institution. The full \`bank-full.csv\` version has 45,211 observations, 16 predictors and one binary target indicating whether a client subscribed to a term deposit. UCI reports a CC BY 4.0 license and DOI \`10.24432/C5K306\`.

The runner downloads the canonical UCI archive when no local copy is supplied, extracts \`bank-full.csv\`, caches it outside version control, and records a SHA-256 hash of the exact CSV used. See [DATA.md](DATA.md).

## Study design

### Primary conditions

1. \`dummy_prior\` — class-prior probability baseline.
2. \`logistic_uncalibrated\` — leakage-aware mixed-type preprocessing plus logistic regression.
3. \`logistic_sigmoid\` — the same logistic learner with five-fold sigmoid calibration inside training data.
4. \`logistic_isotonic\` — the same logistic learner with five-fold isotonic calibration inside training data.

### Evaluation

- fixed primary split: seed 42;
- repeated stratified 80/20 holdouts: seeds 13, 29, 42, 73 and 101;
- ROC-AUC and average precision for ranking;
- Brier score, log loss and ECE-10 for probability quality;
- accuracy at a 0.5 threshold as a secondary descriptive metric;
- paired split-level deltas against uncalibrated logistic regression;
- descriptive bootstrap intervals over paired split-level deltas, with **no p-value claim** because repeated holdouts are not independent.

### Sensitivity and ablation

- calibration CV sensitivity: 3, 5 and 10 folds;
- ECE bin sensitivity: 5, 10 and 20 equal-width bins;
- operational ablation: rerun without \`duration\`, because call duration is not known before a marketing call is completed;
- primary-split error analysis including FP, FN and high-confidence errors.

## Run the study

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/run_experiment.py
\`\`\`

For a faster networked empirical run:

\`\`\`bash
PYTHONPATH=. python src/run_experiment.py --quick
\`\`\`

The full runner writes:

\`\`\`text
results/metrics.json
results/repeated_runs.csv
results/summary.md
results/figures/calibration_curve.png
results/figures/roc_curve.png
results/figures/precision_recall_curve.png
paper/results.md
\`\`\`

The GitHub \`Empirical Study\` workflow reruns the full study after material changes to the runner and commits regenerated result artifacts. Numerical results should therefore come from the executable protocol rather than hand-edited prose.

## Research-bundle evidence

This repository includes a falsifiable empirical question, canonical real-data provenance and checksum capture, leakage-aware preprocessing, explicit baselines, repeated holdout evaluation, calibration-method sensitivity, ECE-bin sensitivity, an operational feature ablation, error analysis, machine-readable outputs, offline CI, a separate empirical workflow, and reproducibility/ethics documentation.

See [RESEARCH_BUNDLE.md](RESEARCH_BUNDLE.md) for the evidence contract.

## Interpretation boundary

This is a methodology study, **not a banking decision system**. The target is historical marketing response. Results do not justify credit decisions, customer exclusion, eligibility decisions or automated targeting. Calibration can shift across time, institutions, populations and acquisition channels.

## Professor review path

1. [README.md](README.md)
2. [DATA.md](DATA.md)
3. [src/run_experiment.py](src/run_experiment.py)
4. [results/summary.md](results/summary.md)
5. [results/metrics.json](results/metrics.json)
6. [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
7. [ETHICS.md](ETHICS.md)
8. [paper/paper.md](paper/paper.md)
9. [paper/results.md](paper/results.md)

## Citation

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Repository citation metadata is in [CITATION.cff](CITATION.cff).
