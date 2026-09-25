# Dataset Card — UCI Bank Marketing

## Source
UCI Machine Learning Repository, dataset 222: **Bank Marketing**  
DOI: https://doi.org/10.24432/C5K306  
Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing  
License reported by UCI: CC BY 4.0.

## Study role
The dataset is the default empirical data source for this repository. It replaces the earlier scikit-learn breast-cancer demonstration.

## Target
The UCI target is whether the client subscribed to a term deposit. The runner maps `yes` to 1 and `no` to 0.

## Predictors
The dataset contains numerical and categorical campaign/client attributes. Categorical variables are one-hot encoded; numerical variables are standardized. Preprocessing is inside the estimator pipeline and is fitted only on the training partition.

## Missing and special values
UCI data may contain categorical values such as `unknown`. The study preserves them as observed categories rather than silently imputing a substantive meaning.

## Split
The default empirical protocol uses a fixed stratified 80/20 holdout split with seed 42. Calibration is learned within the training partition through cross-validation. The test set is not used for model fitting or calibrator fitting.

## Governance
No raw UCI data are committed to this repository. The runner downloads them from the original repository through `ucimlrepo`. Users are responsible for respecting the dataset license and attribution requirements.

## Known limitations
This is historical campaign data from one institutional context. It should not be treated as representative of present-day banking populations, channels or policy environments.
