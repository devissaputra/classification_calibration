# Classification and Probability Calibration

**Focus:** probability quality, calibration, and evaluation.

I compare an uncalibrated logistic-regression classifier with cross-validated sigmoid and isotonic calibration on the Wisconsin Diagnostic Breast Cancer benchmark. The experiment keeps the final test set untouched and evaluates discrimination and probability quality separately using ROC-AUC, Brier score, log loss, accuracy, and a 10-bin calibration-error estimate.

The most useful result is not a dramatic model win: the base logistic model is already strong. Isotonic calibration slightly improves Brier score and log loss, while the small held-out sample makes fine-grained calibration conclusions uncertain. The repository includes import-safe experiment code, behavioural tests, CI, provenance, reproducibility notes, and responsible-use documentation.
