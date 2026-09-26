# Calculation guide

## Question and evidence

Can predicted probabilities be trusted?

UCI Bank Marketing: 45,211 records and 16 predictors; subscription is the positive class.

**Status:** RECORDED EXTERNAL-DATA STUDY | full experiment not rerun in this review.

## Design

Five stratified holdouts; training-only preprocessing and cross-validated probability calibration.

## Calculation and interpretation

`Brier = mean((p - y)^2); ECE = sum(bin share × |mean p - mean y|).`

Lower Brier and log loss indicate better probability predictions. ECE depends on the chosen bins; a constant prevalence forecast can have low ECE while having no discrimination. Split bootstrap intervals are descriptive because holdouts overlap.

## Evidence table

Selected recorded values (units and context shown). Full precision below is for traceability, not a claim of measurement precision.

| Quantity | Value | Unit / meaning | JSON path |
|---|---:|---|---|
| dummy_prior | 0.10330837430613315 | mean Brier ↓ | `repeated_summary.dummy_prior.brier.mean` |
| uncalibrated | 0.07198757353865393 | mean Brier ↓ | `repeated_summary.logistic_uncalibrated.brier.mean` |
| sigmoid | 0.07197467863115595 | mean Brier ↓ | `repeated_summary.logistic_sigmoid.brier.mean` |
| isotonic | 0.06927224756658759 | mean Brier ↓ | `repeated_summary.logistic_isotonic.brier.mean` |

Source: [results/metrics.json](results/metrics.json). Values resolve directly from this file when figures are regenerated.

Isotonic calibration reduces mean Brier score from 0.0720 to 0.0693 across five holdouts, while ROC-AUC stays near 0.906. This supports a probability-quality improvement under the frozen protocol, rather than a substantial change in ranking. The call-duration ablation matters operationally because duration is unavailable before a call; results using it describe a retrospective task.

## Verification performed in this review

The existing suite requires unavailable dependencies; no full-suite pass is claimed. The complete data/model experiment was not rerun in this review. Stored empirical results were inspected, not independently reproduced from raw data.

The figure-generation check verifies agreement between the selected source values and SVGs. It does not validate the raw dataset, fitted model, identification assumptions, or external generalization.

```bash
python scripts/build_review_figures.py
python scripts/build_review_figures.py --check
```

## Implementation map

Follow these functions to inspect each transformation. Validation helpers and private functions remain visible in the linked modules.

| Function | Purpose / documented behavior |
|---|---|
| [`expected_calibration_error`](src/run_experiment.py#L54) | Inspect the explicit implementation and its callers. |
| [`validate_dataset_hash`](src/run_experiment.py#L75) | Inspect the explicit implementation and its callers. |
| [`normalize_target`](src/run_experiment.py#L85) | Inspect the explicit implementation and its callers. |
| [`load_real_data`](src/run_experiment.py#L110) | Inspect the explicit implementation and its callers. |
| [`build_preprocessor`](src/run_experiment.py#L155) | Inspect the explicit implementation and its callers. |
| [`base_estimator`](src/run_experiment.py#L172) | Inspect the explicit implementation and its callers. |
| [`build_models`](src/run_experiment.py#L179) | Inspect the explicit implementation and its callers. |
| [`evaluate_probabilities`](src/run_experiment.py#L189) | Inspect the explicit implementation and its callers. |
| [`error_analysis`](src/run_experiment.py#L202) | Inspect the explicit implementation and its callers. |
| [`run_split`](src/run_experiment.py#L219) | Inspect the explicit implementation and its callers. |
| [`summarize_repeated`](src/run_experiment.py#L239) | Inspect the explicit implementation and its callers. |
| [`paired_seed_differences`](src/run_experiment.py#L256) | Inspect the explicit implementation and its callers. |
| [`ece_bin_sensitivity`](src/run_experiment.py#L280) | Inspect the explicit implementation and its callers. |
| [`calibration_cv_sensitivity`](src/run_experiment.py#L287) | Inspect the explicit implementation and its callers. |
| [`feature_ablation`](src/run_experiment.py#L302) | Inspect the explicit implementation and its callers. |
| [`write_figures`](src/run_experiment.py#L321) | Inspect the explicit implementation and its callers. |
| [`write_repeated_csv`](src/run_experiment.py#L365) | Inspect the explicit implementation and its callers. |
| [`build_summary_markdown`](src/run_experiment.py#L377) | Inspect the explicit implementation and its callers. |
| [`build_results_latex`](src/run_experiment.py#L418) | Inspect the explicit implementation and its callers. |
| [`run_experiment`](src/run_experiment.py#L484) | Inspect the explicit implementation and its callers. |
| [`main`](src/run_experiment.py#L543) | Inspect the explicit implementation and its callers. |

## What remains before a stronger research claim

Lower Brier and log loss indicate better probability predictions. ECE depends on the chosen bins; a constant prevalence forecast can have low ECE while having no discrimination. Split bootstrap intervals are descriptive because holdouts overlap. A successful software test is not validation of a scientific construct. New experiments should state their split unit, comparator, outcome, uncertainty procedure and failure criteria before examining final test results.
