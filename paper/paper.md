# Probability Calibration Under Class Imbalance: A Reproducible Study on UCI Bank Marketing

## Abstract

Probability calibration can look convincing on a single split while remaining sensitive to the split, calibration procedure, and calibration metric. This study compares a class-prior baseline and logistic regression under uncalibrated, sigmoid-calibrated, and isotonic-calibrated conditions on UCI Bank Marketing. The protocol combines an untouched primary holdout, five repeated stratified holdouts, multiple probability-quality metrics, calibration-fold sensitivity, ECE-bin sensitivity, an operational feature ablation, calibration-bin error analysis, and descriptive paired bootstrap intervals. Numerical results are generated directly by the executable pipeline.

## Research question

How do sigmoid and isotonic calibration alter logistic-regression probability quality and discrimination on UCI Bank Marketing, and how stable are those changes across repeated splits and sensitivity checks?

## Data

The study uses UCI Bank Marketing, dataset 222, DOI 10.24432/C5K306, licensed CC BY 4.0. The runner records the returned sample count, feature count, positive-class prevalence, and a deterministic SHA-256 fingerprint of the loaded dataframe representation.

## Methods

### Preprocessing

Numeric predictors are median-imputed and standardized. Categorical predictors are mode-imputed and one-hot encoded with unknown-category tolerance. All preprocessing is fitted inside the training pipeline.

### Conditions

The primary comparison contains a class-prior dummy baseline, uncalibrated logistic regression, five-fold sigmoid-calibrated logistic regression, and five-fold isotonic-calibrated logistic regression.

### Evaluation

The primary split uses seed 42 with a stratified 80/20 partition. Robustness is assessed with seeds 13, 29, 42, 73, and 101. Ranking is assessed with ROC-AUC and average precision. Probability quality is assessed with Brier score, log loss, and expected calibration error. Accuracy at threshold 0.5 is retained as a secondary descriptive measure.

### Sensitivity analyses

The study varies calibration cross-validation folds across 3, 5, and 10 and recomputes ECE with 5, 10, and 20 equal-width bins. The primary design is also rerun without duration when that feature is present because call duration is unavailable before the contact has occurred.

### Uncertainty reporting

Repeated holdouts reuse observations and are not independent experiments. The study therefore reports descriptive means, standard deviations, and paired split-level bootstrap intervals without p-values or claims of formal statistical significance.

### Error analysis

For each primary model, the runner aggregates the untouched test predictions into calibration bins and reports bin size, mean predicted probability, observed event rate, and absolute calibration gap.

## Results

Generated numerical results are written to paper/results.md and results/summary.md. The manuscript intentionally avoids hand-entered performance values.

## Validity and limitations

The study is a methodological benchmark on historical marketing data. It does not establish causal effects, present-day validity, fairness, cross-institution transportability, or suitability for consequential banking decisions. Expected calibration error also depends on binning, which is why multiple bin counts are reported.

## Reproducibility

The executable protocol is src/run_experiment.py. Full evidence is stored in results/metrics.json, results/repeated_runs.csv, results/sensitivity_cv.csv, results/sensitivity_ece.csv, results/ablation.csv, results/error_analysis.csv, results/summary.md, results/figures, and paper/results.md.
