# Portfolio Summary

## Classification and Probability Calibration

I built this project to separate two questions that are often mixed together: can a classifier rank cases correctly, and can its probabilities be trusted?

The model is logistic regression on the Wisconsin Diagnostic Breast Cancer dataset. I evaluate accuracy, ROC-AUC, Brier score, and calibration behaviour on a held-out test set.

### Images

![Project overview](assets/01_cover.svg)

![Processing pipeline](assets/02_data_pipeline.svg)

![Probability view](assets/03_data_or_model.svg)

![Evaluation summary](assets/04_evaluation_or_results.svg)

**Key result:** 0.9977 ROC-AUC with a Brier score of 0.0181 on the recorded split.
