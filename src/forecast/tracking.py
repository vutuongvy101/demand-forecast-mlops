"""MLflow experiment tracking helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import mlflow


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def setup_mlflow(cfg: dict[str, Any]) -> str:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    experiment = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_name"])
    if experiment is None:
        experiment_id = mlflow.create_experiment(cfg["mlflow"]["experiment_name"])
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_run_context(cfg: dict[str, Any], data_path: Path | None = None) -> None:
    mlflow.set_tag("git_sha", get_git_sha())
    mlflow.log_dict(cfg, "config.yaml")

    if data_path and data_path.exists():
        mlflow.set_tag("dataset_hash", hash_file(data_path))


def log_metrics_dict(metrics: dict[str, float], prefix: str = "") -> None:
    for key, value in metrics.items():
        if value is not None and not (isinstance(value, float) and value != value):
            name = f"{prefix}{key}" if prefix else key
            mlflow.log_metric(name, float(value))


def save_json_artifact(data: dict[str, Any], filename: str) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    mlflow.log_artifact(str(path))
    return path
