.PHONY: install test notebook reproduce

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/fd03_unsupervised_anomaly.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/fd03_unsupervised_anomaly.ipynb

reproduce: test notebook
