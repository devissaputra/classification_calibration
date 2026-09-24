setup:
	python -m pip install -r requirements.txt

run:
	python src/run_experiment.py

test:
	python -m pytest -q
