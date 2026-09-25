# Data directory

Raw UCI data are intentionally not versioned.

\`src/run_experiment.py\` downloads the canonical UCI Bank Marketing archive and caches the extracted \`bank-full.csv\` under \`data/cache/\`. The cache directory is gitignored. You can also supply a local copy explicitly:

\`\`\`bash
python src/run_experiment.py --data-path /path/to/bank-full.csv
\`\`\`

Every empirical run records the SHA-256 hash of the CSV used.
