# Classification and Probability Calibration

## Question

Can a simple probabilistic classifier both separate the two classes well and produce useful probabilities?

## Data

I use the Wisconsin Diagnostic Breast Cancer dataset from scikit-learn. It contains 569 observations and 30 numerical features.

I split the data into 75% training and 25% test data with stratification and `random_state=42`.

## Method

The model is logistic regression inside a scikit-learn pipeline:

1. standardize the features;
2. fit logistic regression;
3. predict probabilities on the held-out set;
4. convert probabilities to classes at a 0.5 threshold.

I evaluate accuracy and ROC-AUC for classification performance, then use the Brier score and a calibration curve to look at probability quality.

## Results

The recorded run produced:

| Metric | Result |
|---|---:|
| Accuracy | 0.9860 |
| ROC-AUC | 0.9977 |
| Brier score | 0.0181 |

## Interpretation

The classifier separates the classes very well on this split. The Brier score is also low, but a single hold-out split is not enough to say that the probabilities will remain well calibrated on new data.

## Limitations

The dataset is small and the experiment uses one split. The calibration curve also depends on how probability bins are chosen.

A stronger version would repeat the evaluation across several splits, report confidence intervals, and compare the uncalibrated model with Platt scaling and isotonic regression.

## Reproduce

From the repository root:

```bash
pip install -r requirements.txt
python src/run_experiment.py
```

The script writes the numerical results to `results/metrics.json`.
