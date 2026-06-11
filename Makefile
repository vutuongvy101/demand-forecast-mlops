.PHONY: install install-dev data backtest train predict dashboard dag-test test lint clean

PYTHON ?= python
PIP ?= pip

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,dashboard,airflow]"

data:
	$(PYTHON) -m forecast.ingest

backtest:
	$(PYTHON) -m forecast.backtest

train:
	$(PYTHON) -m forecast.train

predict:
	$(PYTHON) -m forecast.predict

drift:
	$(PYTHON) -m forecast.drift

dashboard:
	streamlit run dashboard/app.py

dag-test:
	airflow dags test forecast_pipeline $$(date +%Y-%m-%d)

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ dashboard/ dags/

clean:
	rm -rf data/ mlruns/ artifacts/ outputs/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
