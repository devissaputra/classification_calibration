# Dataset Card: UCI Bank Marketing

## Source

UCI Machine Learning Repository, dataset 222: Bank Marketing  
DOI: https://doi.org/10.24432/C5K306  
Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing  
License reported by UCI: CC BY 4.0.

The runner retrieves the official UCI archive and extracts bank-full.csv. No raw dataset is committed to this repository. The frozen protocol validates the exact CSV bytes before analysis.

## Frozen source identity

Expected SHA-256 of `bank-full.csv`: `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`

The runner fails if the loaded file does not match this value, even when row and column counts still look valid. A new upstream byte sequence must be reviewed and accepted as an explicit protocol revision rather than silently replacing the study data.

## Study role

This is the real external dataset used by the empirical calibration study. The runner records the exact SHA-256 of the loaded CSV bytes, source path, observed row count, observed predictor count, and positive-class prevalence.

## Target

The target y indicates whether the client subscribed to a term deposit. The runner maps yes to 1 and no to 0 and rejects unexpected labels.

## Predictors and preprocessing

Numerical variables are median-imputed and standardized. Categorical variables are mode-imputed and one-hot encoded with unknown-category tolerance. Preprocessing is fitted inside each training pipeline, not globally before splitting.

## Split discipline

The primary study uses a fixed stratified 80/20 holdout with seed 42. Calibrators are learned only inside the training partition through cross-validation. The primary test partition is not used for model fitting or calibrator fitting.

## Operational feature caveat

The duration feature records call duration and is not available before a call occurs. It can therefore make retrospective prediction look stronger than a pre-contact decision process could be. The full protocol explicitly reruns the primary experiment after removing duration.

## Local reproducibility

A local copy can be supplied with:

    PYTHONPATH=. python src/run_experiment.py --data-path /path/to/bank-full.csv

The same raw-file SHA-256 mechanism is applied to a local file.

## Limitations

The dataset is historical and comes from one institutional setting. It should not be treated as representative of current banking populations, channels, regulations, or deployment environments.
