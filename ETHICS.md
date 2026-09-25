# Ethics, Deployment Boundaries and Responsible Use

This repository analyzes historical **bank marketing response**, not creditworthiness, customer value, fraud risk, eligibility or financial need.

## Prohibited interpretation

The experiment must not be presented as evidence that a bank should approve, deny, price or otherwise allocate financial products to a person. The target only records whether a client subscribed to a term deposit after a marketing campaign.

## Population and temporal validity

The UCI data describe campaigns from one institutional context and historical period. Calibration is especially sensitive to prevalence and distribution shift, so a probability that is well calibrated in this dataset may be miscalibrated in a later campaign, another institution or another population.

## Sensitive and proxy information

The dataset includes demographic and socioeconomic attributes such as age, job, marital status and education. Even when used only for methodology research, these variables can encode or proxy unequal social conditions. A deployment study would require a justified feature policy, subgroup analysis, fairness review, legal review and monitoring for drift and disparate outcomes.

## Duration limitation

Call duration is unavailable before a call is made. Treating it as a pre-contact targeting feature would leak future information into a decision that occurs earlier in time. This repository therefore reports a no-\`duration\` ablation and explicitly separates retrospective prediction from operational use.

## What would be required before deployment

A real deployment would need fresh local data, temporal validation, prospective calibration monitoring, subgroup performance analysis, decision-specific utility analysis, human oversight, privacy controls, documentation of lawful purpose and governance appropriate to the jurisdiction.

This research bundle does not perform or authorize such deployment.
