"""
Airflow DAG: ingest → features (implicit in train) → train → evaluate → branch → forecast/drift.

Run: airflow standalone, then `make dag-test`
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run_module(module: str, *args: str) -> None:
    cmd = [PYTHON, "-m", module, *args]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{module} failed:\n{result.stderr}")
    print(result.stdout)


def check_data(**_context) -> None:
    processed = ROOT / "data" / "processed" / "sales_long.parquet"
    if not processed.exists():
        _run_module("forecast.ingest")


def train_model(**_context) -> None:
    _run_module("forecast.train")


def evaluate_model(**_context) -> str:
    """Model quality gate: compare current vs. last accepted run."""
    reports = ROOT / "data" / "reports"
    results_path = reports / "backtest_results.json"
    if not results_path.exists():
        _run_module("forecast.backtest")

    with results_path.open() as f:
        results = json.load(f)

    lgbm = next((r for r in results if r["model"] == "lgbm"), None)
    if lgbm is None:
        return "quality_gate_fail"

    ref_path = ROOT / "data" / "models" / "accepted_wrmsse.json"
    current_wrmsse = lgbm.get("wrmsse", float("inf"))

    if ref_path.exists():
        with ref_path.open() as f:
            accepted = json.load(f)
        threshold = accepted.get("wrmsse", current_wrmsse) * 1.10
        if current_wrmsse > threshold:
            print(f"Quality gate FAIL: WRMSSE {current_wrmsse:.4f} > {threshold:.4f}")
            return "quality_gate_fail"

    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps({"wrmsse": current_wrmsse}, indent=2))
    print(f"Quality gate PASS: WRMSSE={current_wrmsse:.4f}")
    return "generate_forecast"


def generate_forecast(**_context) -> None:
    _run_module("forecast.predict")


def drift_check(**_context) -> None:
    _run_module("forecast.drift")


def quality_gate_fail(**_context) -> None:
    print("Model quality gate failed — skipping forecast generation.")


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 0,
    "execution_timeout": timedelta(hours=2),
}

with DAG(
    dag_id="forecast_pipeline",
    default_args=default_args,
    description="Demand forecast MLOps pipeline with quality gate",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["forecast", "mlops"],
) as dag:
    t_check = PythonOperator(task_id="check_data", python_callable=check_data)
    t_train = PythonOperator(task_id="train_model", python_callable=train_model)
    t_branch = BranchPythonOperator(task_id="evaluate", python_callable=evaluate_model)
    t_forecast = PythonOperator(task_id="generate_forecast", python_callable=generate_forecast)
    t_drift = PythonOperator(task_id="drift_check", python_callable=drift_check)
    t_fail = PythonOperator(task_id="quality_gate_fail", python_callable=quality_gate_fail)

    t_check >> t_train >> t_branch
    t_branch >> t_forecast >> t_drift
    t_branch >> t_fail
