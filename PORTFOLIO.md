# Probability Calibration

This study asks a more demanding question than whether a classifier is simply accurate: when a model assigns a probability, does that number match what actually happens? Using the UCI Bank Marketing dataset with 45,211 observations, 16 predictors, and an 11.7% positive class, the protocol compares a class-prior baseline with uncalibrated, sigmoid-calibrated, and isotonic-calibrated logistic regression across five fixed stratified holdouts. Discrimination is evaluated separately from probability quality using ROC-AUC and average precision alongside Brier score, log loss, and expected calibration error.

Isotonic calibration reduces mean Brier score from 0.0720 to 0.0693 across five holdouts, while ROC-AUC stays near 0.906. This supports a probability-quality improvement under the frozen protocol, rather than a substantial change in ranking. The call-duration ablation matters operationally because duration is unavailable before a call; results using it describe a retrospective task.

See [CALCULATIONS.md](CALCULATIONS.md) for evidence and verification scope.
