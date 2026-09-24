# Classification and Probability Calibration

## Abstract

This experiment examines whether post-hoc probability calibration improves logistic-regression predictions on the Wisconsin Diagnostic Breast Cancer benchmark. A stratified 75/25 split is used, with the final test set held out from all fitting. The base model is standardized logistic regression. Two calibrated variants are fitted with five-fold cross-validation inside the training set: sigmoid scaling and isotonic regression. Evaluation combines ROC-AUC with Brier score, log loss, accuracy, and a 10-bin expected calibration error (ECE).

## Method

The dataset contains 569 observations and 30 numerical features. Random seed 42 fixes the split. Calibration is learned only from training data through cross-validation.

## Results

| Model | Accuracy | ROC-AUC | Brier | Log loss | ECE-10 |
|---|---:|---:|---:|---:|---:|
| Uncalibrated | 0.9860 | 0.9977 | 0.0181 | 0.0679 | 0.0310 |
| Sigmoid | 0.9790 | 0.9979 | 0.0273 | 0.1152 | 0.0769 |
| Isotonic | 0.9720 | 0.9981 | 0.0179 | 0.0657 | 0.0345 |

All three variants rank cases very well. Isotonic calibration gives the lowest Brier score and log loss in this run, but the differences are small. The uncalibrated model has the lowest 10-bin ECE. Because ECE depends on binning and the test set contains only 143 cases, these differences should be treated as descriptive rather than definitive.

## Interpretation

The experiment demonstrates why discrimination and calibration should be evaluated separately. A high ROC-AUC does not by itself establish that predicted probabilities are trustworthy, and a calibration method should not be assumed to improve every metric on every dataset.

## Limitations

This is a small benchmark study with one internal hold-out split and no external clinical validation. It is not evidence for diagnostic use.

The positive class is benign (1). Calibrated predictions average five classifier/calibrator pairs, so the comparison includes ensemble effects. This is a technical demonstration report, not a peer-reviewed publication.
