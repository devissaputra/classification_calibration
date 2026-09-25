# Dataset Card: UCI Bank Marketing

## Source

UCI Machine Learning Repository, dataset 222: Bank Marketing  
DOI: https://doi.org/10.24432/C5K306  
Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing  
License reported by UCI: CC BY 4.0.

UCI describes the dataset as direct-marketing campaign data from a Portuguese banking institution, with the binary goal of predicting whether a client subscribed to a term deposit. The repository records the sample and feature counts returned by the loader at run time rather than hard-coding them as experimental evidence.

## Study role

This is the real external dataset used by the empirical pipeline. No source data are committed to this repository.

## Target

The target is subscription to a term deposit. The runner maps yes to 1 and no to 0 and rejects unexpected target labels.

## Predictors and preprocessing

Numerical variables are median-imputed and standardized. Categorical variables are mode-imputed and one-hot encoded with unknown-category tolerance. Preprocessing lives inside the estimator pipeline and is fitted on training data only.

## Split discipline

The primary study uses a fixed stratified 80/20 holdout with seed 42. Calibrators are learned only inside the training partition through cross-validation. The untouched primary test partition is used only for final evaluation.

## Operational feature caveat

The duration variable, when present, is known only after a call occurs. The full protocol therefore repeats the primary experiment after removing duration. This ablation is required for an honest operational interpretation.

## Provenance fingerprint

The full runner stores a SHA-256 fingerprint derived from the loaded dataframe content and row order. This is a reproducibility fingerprint of the loaded data representation, not a claim about the raw archive byte hash.

## Limitations

This is historical data from one institutional setting. Results should not be interpreted as representative of current banking populations, channels, regulations, or deployment environments.
