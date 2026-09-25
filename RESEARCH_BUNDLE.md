# Research Bundle Evidence Contract

## Bundle identity

Area: AI Engineering  
Study: Probability calibration under class imbalance  
Primary dataset: UCI Bank Marketing, dataset 222  
Primary evidence: untouched holdout plus repeated-split and sensitivity analyses

## Evidence required for a valid empirical run

A full run must record and generate:

1. UCI dataset identity, DOI, sample count, feature count, target prevalence, and a deterministic fingerprint of the loaded dataframe.
2. A fixed primary 80/20 stratified holdout using seed 42.
3. A class-prior dummy baseline plus uncalibrated, sigmoid-calibrated, and isotonic-calibrated logistic regression.
4. ROC-AUC, average precision, Brier score, log loss, ECE, and accuracy.
5. Five repeated stratified holdouts using the frozen seed set.
6. Paired split-level bootstrap intervals for calibrated-minus-uncalibrated metric deltas, reported descriptively rather than as significance tests.
7. Calibration-fold sensitivity for 3, 5, and 10 folds.
8. ECE-bin sensitivity for 5, 10, and 20 bins.
9. An operational ablation that removes duration when that feature is present.
10. Calibration-bin error analysis.
11. Generated figures, machine-readable tables, environment versions, offline tests, and CI.

## Non-claims

The bundle does not claim universal superiority of any calibration method, causal effects, current-population validity, cross-institution transportability, fairness, or suitability for consequential banking decisions.

## Professor review path

Read README.md, DATA.md, src/run_experiment.py, tests/test_experiment.py, results/summary.md, paper/paper.md, REPRODUCIBILITY.md, and ETHICS.md in that order.
