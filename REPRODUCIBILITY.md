# Reproducibility Protocol

## Environment

Use Python 3.11+.

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
PYTHONPATH=. python src/run_experiment.py
\`\`\`

## Frozen defaults

- dataset: UCI Bank Marketing, dataset 222, \`bank-full.csv\`;
- primary seed: 42;
- repeated seeds: 13, 29, 42, 73, 101;
- test fraction: 0.20;
- stratified split: yes;
- logistic regression: \`lbfgs\`, \`max_iter=4000\`;
- calibration CV: 5 folds in the primary comparison;
- calibration-CV sensitivity: 3, 5 and 10 folds;
- ECE primary: 10 equal-width bins;
- ECE sensitivity: 5, 10 and 20 bins;
- operational ablation: remove \`duration\`.

## Data identity

Every empirical run stores the SHA-256 hash of the exact \`bank-full.csv\` bytes in \`results/metrics.json\`. If two reviewers obtain different hashes, their numerical results should not be treated as directly reproducible until the data discrepancy is resolved.

## Outputs

The full runner creates:

- \`results/metrics.json\` — complete machine-readable protocol and metrics;
- \`results/repeated_runs.csv\` — one row per seed/model condition;
- \`results/summary.md\` — generated human-readable summary;
- \`results/figures/\` — calibration, ROC and precision-recall figures;
- \`paper/results.md\` — generated paper-facing results section.

## CI boundary

\`CI\` runs offline unit tests and never depends on UCI availability. \`Empirical Study\` is a separate networked workflow that downloads the canonical UCI data, runs the full analysis, and commits regenerated result artifacts after material changes to the empirical runner.

This separation prevents a transient dataset outage from breaking basic code QA while keeping numerical claims tied to a real-data execution path.

## Statistical interpretation

Five repeated holdout splits are used as a robustness diagnostic, not as five independent experiments. The analysis reports mean/SD and descriptive bootstrap intervals of paired split-level deltas. It deliberately avoids p-values from these dependent holdouts.

## Result policy

Do not manually copy or invent metrics. Numerical claims in the README, manuscript or portfolio should be taken from the latest generated \`results/summary.md\`, \`results/metrics.json\` and committed figures.
