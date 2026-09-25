# Ethics and Responsible-Use Boundary

## Scope

This repository is a methodological study of probability calibration using the UCI Bank Marketing dataset. It is not a production banking system and it is not intended to decide who should receive financial products, credit, pricing, or other consequential treatment.

## Data limitations

The data describe historical direct-marketing campaigns from one Portuguese banking institution. They do not establish present-day population representativeness, cross-country validity, institutional transportability, or future performance.

Several predictors describe contact and campaign history. In particular, call duration is only known after a call occurs. The study therefore includes an explicit duration-removal ablation to show how conclusions change when that operationally unavailable feature is excluded.

## Modeling limitations

Calibration on one historical dataset does not establish fairness, causal validity, deployment safety, or subgroup reliability. Repeated train/test splits reuse observations and are treated as robustness checks rather than independent experiments.

## Appropriate use

Use the repository to study calibration methodology, reproducibility, metric sensitivity, and experimental design. Any real banking deployment would require current data, legal and governance review, subgroup evaluation, drift monitoring, external validation, and a decision-specific risk assessment.
