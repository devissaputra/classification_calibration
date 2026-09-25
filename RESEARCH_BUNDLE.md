# Research Bundle Evidence Contract

## Bundle identity

**Area:** AI Engineering  
**Study:** Probability calibration under class imbalance  
**Primary dataset:** UCI Bank Marketing, dataset 222  
**Primary evidence:** repeated untouched-test probabilistic evaluation plus sensitivity analysis

## A run counts as empirical evidence only when it records

1. canonical UCI source, DOI, license and SHA-256 of the exact \`bank-full.csv\` used;
2. sample count, feature count and target prevalence;
3. split seeds, train/test sizes and stratification;
4. preprocessing fitted only inside training pipelines;
5. dummy, uncalibrated, sigmoid and isotonic conditions;
6. ROC-AUC, average precision, Brier score, log loss, ECE-10 and accuracy;
7. repeated-split summary statistics;
8. paired calibration deltas relative to uncalibrated logistic regression;
9. ECE bin sensitivity and calibration-CV sensitivity;
10. \`duration\` feature ablation;
11. primary-split error analysis;
12. environment versions and generated artifacts.

## Statistical boundary

Repeated holdouts share observations and are not independent replications. The repository therefore reports descriptive mean/SD and bootstrap intervals over paired split-level deltas but does not label those intervals as formal hypothesis-test confidence intervals and does not report p-values.

## Non-claims

The bundle does not claim that one calibration method is universally superior, that this historical dataset represents current banking populations, that the observed associations are causal, or that the model is suitable for consequential decisions.

## Professor review path

Read in this order:

1. \`README.md\`
2. \`DATA.md\`
3. \`src/run_experiment.py\`
4. \`results/summary.md\`
5. \`results/metrics.json\`
6. \`tests/test_experiment.py\`
7. \`REPRODUCIBILITY.md\`
8. \`ETHICS.md\`
9. \`paper/paper.md\`
10. \`paper/results.md\`
