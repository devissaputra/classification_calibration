# Research Bundle Evidence Contract

## Bundle identity
**Area:** AI Engineering  
**Study:** Probability calibration under class imbalance  
**Primary dataset:** UCI Bank Marketing (222)  
**Primary evidence:** untouched-test probabilistic evaluation

## Required evidence
A run qualifies as an empirical bundle result only when it records:

1. dataset source and UCI dataset ID;
2. sample count and target prevalence;
3. fixed random seed and split sizes;
4. preprocessing fitted on training data only;
5. model/calibration specification;
6. ROC-AUC, Brier score, log loss, ECE-10 and accuracy;
7. environment versions;
8. generated result file from the real UCI data.

## Non-claims
The bundle does not claim that one calibration method is universally superior, that the data are representative of current banking populations, or that the model is suitable for consequential decisions.

## Professor review path
Read in this order:

1. `README.md`
2. `DATA.md`
3. `src/run_experiment.py`
4. `tests/test_experiment.py`
5. `REPRODUCIBILITY.md`
6. `ETHICS.md`
7. `paper/paper.md`
8. generated `results/metrics.json`
