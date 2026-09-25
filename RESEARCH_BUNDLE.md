# Research Bundle Evidence Contract

## Bundle identity

Area: AI Engineering  
Study: Probability calibration under class imbalance  
Primary dataset: UCI Bank Marketing, dataset 222  
Primary evidence: untouched primary holdout plus repeated-split and sensitivity analyses

## Evidence required for a valid empirical run

A full run must record and generate:

1. Dataset identity, DOI, license, sample count, feature count, target prevalence, source location, and SHA-256 of the exact loaded bank-full.csv bytes; the hash must equal the frozen expected value before analysis proceeds.
2. A fixed primary 80/20 stratified holdout using seed 42.
3. A class-prior dummy baseline plus uncalibrated, sigmoid-calibrated, and isotonic-calibrated logistic regression.
4. ROC-AUC, average precision, Brier score, log loss, ECE-10, and accuracy.
5. Five repeated stratified holdouts using seeds 13, 29, 42, 73, and 101.
6. Descriptive paired split-level bootstrap intervals for calibrated-minus-uncalibrated metric deltas using 4,000 bootstrap resamples.
7. Calibration-fold sensitivity at 3, 5, and 10 folds.
8. ECE-bin sensitivity at 5, 10, and 20 bins.
9. An operational ablation that removes duration.
10. Primary-split confusion counts, error rate, and high-confidence error counts.
11. Machine-readable results, split-level CSV output, generated figures, environment versions, offline tests, CI, and an empirical regeneration workflow.

## Non-claims

The bundle does not claim universal superiority of any calibration method, causal effects, present-day population validity, cross-institution transportability, fairness, or suitability for consequential banking decisions.

## Professor review path

Read README.md, DATA.md, src/run_experiment.py, tests/test_experiment.py, results/summary.md, paper/paper.md, REPRODUCIBILITY.md, and ETHICS.md in that order.


## Frozen data identity

The accepted `bank-full.csv` SHA-256 is `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`. A source change is treated as a protocol change, not as an invisible refresh.
