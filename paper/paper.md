# Classification + Calibration: Scientific-Style Technical Report

**Status:** reproducible portfolio report, not peer reviewed.  
**Difficulty:** ★★★  
**Dataset:** Wisconsin Diagnostic Breast Cancer dataset

## Abstract
This project studies a concrete AI Engineering problem using a real public dataset and a fully inspectable pipeline. The project focuses on classification, probability calibration, ROC-AUC, Brier score. Its central engineering goal is to make data preparation, model fitting, evaluation, and limitations reproducible rather than treating the model as a black box.

## 1. Research objective
Build a probabilistic classifier and evaluate discrimination as well as probability calibration.

## 2. Data
The dataset is **Wisconsin Diagnostic Breast Cancer dataset**. Provenance and the original reference are documented in [`DATA.md`](../DATA.md).

## 3. Method
The implemented pipeline is:
1. Load real data
2. Scale features
3. Logistic regression
4. Probability output
5. Calibration audit

## 4. Evaluation
**Primary metric(s):** ROC-AUC / Brier.  
**Validation design:** stratified hold-out.  
The experiment saves machine-readable metrics and visual diagnostics so claims can be traced to an executable run.

## 5. Results
Generated metrics:
```json
{
  "accuracy": 0.986013986013986,
  "roc_auc": 0.9976939203354298,
  "brier": 0.01806692891806697,
  "n": 569
}
```

## 6. Limitations and validity
Key concern: threshold and calibration sensitivity. Benchmark performance on one dataset does not imply universal performance. The project is intended to demonstrate research engineering discipline and to provide a base for stronger comparative studies.

## 7. Reproducibility
Run `python src/run_experiment.py` from the repository root after installing `requirements.txt`.

## 8. Next research extension
Add repeated cross-validation or temporal/external validation, stronger baselines, hyperparameter sensitivity, confidence intervals, and a domain-specific error analysis.

## References
- Dataset/reference page: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
