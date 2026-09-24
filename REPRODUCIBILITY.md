# Reproducing the experiment

## Environment

Recommended: Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
```

## Run

```bash
python src/run_experiment.py
```

The script uses a stratified 75/25 train/test split with seed 42. The held-out test set is never used to fit the logistic model or either calibration map. Sigmoid and isotonic calibration are fitted with five-fold cross-validation inside the training data.

Outputs:

- `results/metrics.json`
- `results/figures/calibration_curve.png`
- `results/figures/roc_curve.png`

## Test

```bash
pip install pytest
python -m pytest
```

GitHub Actions runs the same tests on pushes and pull requests.

## Numerical reproducibility

The seed fixes the data split and model randomness where applicable. Exact floating-point values can still move slightly as NumPy or scikit-learn change. The README therefore treats the committed metrics as a recorded run, not a universal constant.

## Probability and calibration conventions

Class 0 is malignant and class 1 is benign. Reported probabilities and ROC curves use benign as the positive class. ECE uses 10 equal-width bins; the reliability plot uses 8. The calibrated models average five fitted classifier/calibrator pairs, while the uncalibrated model uses the full training split. This comparison therefore includes ensembling effects as well as calibration.
