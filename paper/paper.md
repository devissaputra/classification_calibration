# Probability Calibration Under Class Imbalance: A Reproducible Study on UCI Bank Marketing

## Abstract

Probability calibration is often evaluated with a single split and a single calibration diagnostic, which can make small improvements look more stable than they are. This study compares uncalibrated logistic regression with sigmoid and isotonic calibration on UCI Bank Marketing, a class-imbalanced binary-response dataset. The protocol combines an untouched primary holdout, five repeated stratified holdouts, multiple probability-quality metrics, calibration-fold sensitivity, ECE-bin sensitivity, an operational feature ablation and error analysis. Numerical results are generated directly by the executable pipeline and stored as machine-readable artifacts.

## Research question

How do sigmoid and isotonic calibration alter logistic-regression probability quality and discrimination on UCI Bank Marketing, and are those changes stable under repeated splits and sensitivity checks?

## Data

The study uses UCI Bank Marketing dataset 222, \`bank-full.csv\`, DOI \`10.24432/C5K306\`, licensed CC BY 4.0. The runner records the exact CSV SHA-256 hash, sample count, feature count and positive-class prevalence for every full run.

## Methods

### Preprocessing

Numeric predictors are median-imputed and standardized. Categorical predictors are mode-imputed and one-hot encoded with unknown-category tolerance. All preprocessing is fitted inside the model pipeline on training data only.

### Conditions

The primary comparison contains a class-prior dummy baseline, uncalibrated logistic regression, five-fold sigmoid-calibrated logistic regression and five-fold isotonic-calibrated logistic regression.

### Evaluation

The primary split uses seed 42 with a stratified 80/20 partition. Robustness is assessed with seeds 13, 29, 42, 73 and 101. Ranking is assessed with ROC-AUC and average precision; probability quality with Brier score, log loss and ECE-10. Accuracy at 0.5 is retained as a secondary descriptive measure.

### Sensitivity analyses

The study varies calibration cross-validation folds across 3, 5 and 10, recomputes ECE with 5, 10 and 20 bins, and reruns the primary design without \`duration\`. The latter is an operational ablation because call duration is not available before contact occurs.

### Uncertainty reporting

Repeated holdouts reuse observations and are not independent experiments. The study therefore reports descriptive mean/SD and bootstrap intervals over paired split-level deltas without p-values or claims of formal statistical significance.

## Results

Generated numerical results are written to [results.md](results.md). This manuscript intentionally does not contain hand-entered performance values.

## Validity and limitations

The study is a methodological benchmark on historical marketing data. It does not establish causal effects, current-population validity, cross-institution transportability or suitability for consequential banking decisions. ECE also depends on binning, which is why multiple bin counts are reported.

## Reproducibility

The executable protocol is \`src/run_experiment.py\`. The full generated evidence is stored in \`results/metrics.json\`, \`results/repeated_runs.csv\`, \`results/summary.md\`, \`results/figures/\` and \`paper/results.md\`.
