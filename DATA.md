# Dataset Card — UCI Bank Marketing

## Canonical source

UCI Machine Learning Repository, dataset 222: **Bank Marketing**  
DOI: https://doi.org/10.24432/C5K306  
Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing  
Canonical archive: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip  
License reported by UCI: **CC BY 4.0**.

## File used

The experiment uses \`bank-full.csv\`, the full 45,211-row version of the older 16-predictor Bank Marketing schema. The CSV contains 16 predictors plus the binary target \`y\`.

The runner records a SHA-256 hash of the exact CSV used in \`results/metrics.json\`, allowing a reviewer to verify that two runs used identical source bytes.

## Target

\`y\` indicates whether the client subscribed to a term deposit. The runner maps \`yes\` to 1 and \`no\` to 0.

## Predictors

The schema mixes numeric and categorical campaign/client attributes. Categorical variables are one-hot encoded and numeric variables are median-imputed and standardized. All preprocessing lives inside the estimator pipeline and is fitted on training data only.

## Special values

The dataset contains categorical values such as \`unknown\`. The study preserves them as observed categories rather than silently assigning a substantive interpretation.

## Operational caveat: duration

\`duration\` records the last contact duration. It can be predictive retrospectively, but it is not available before the call has occurred. The full-feature model is therefore paired with an explicit no-\`duration\` ablation so predictive performance is not confused with pre-contact deployability.

## Split protocol

The primary empirical protocol uses a fixed stratified 80/20 holdout split with seed 42. Robustness is examined with four additional fixed seeds: 13, 29, 73 and 101. Calibrators are learned entirely inside each training partition through cross-validation. Test partitions are not used for preprocessing, model fitting or calibrator fitting.

## Governance

Raw UCI data are not committed to this repository. The runner downloads the canonical archive and caches the extracted CSV under \`data/cache/\`, which is gitignored. Users must preserve UCI attribution and comply with CC BY 4.0.

## Known limitations

The data are historical and come from one institutional and campaign context. They should not be treated as representative of present-day banking populations, channels, policy environments or customer behavior.
