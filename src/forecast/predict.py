"""Generate the next 28-day forecast from the saved model (direct strategy)."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forecast.config import DEFAULT_CONFIG_PATH, load_config
from forecast.features import build_features
from forecast.models import encode_categories, make_future_frame


def predict(config_path: Path | str | None = None, run_date: str | None = None) -> Path:
    cfg = load_config(config_path)
    processed = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "lgbm_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Run `make train` first.")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    encoders = artifact["encoders"]

    long = pd.read_parquet(processed / "sales_long.parquet")
    horizon = cfg["forecast"]["horizon"]
    last_day = int(long["day_idx"].max())

    future = make_future_frame(long, horizon)
    combined = pd.concat([long, future], ignore_index=True)
    featured = build_features(
        combined,
        lags=cfg["features"]["lags"],
        rolling_windows=cfg["features"]["rolling_windows"],
        min_lag=cfg["features"]["min_lag"],
    )
    featured, _ = encode_categories(featured, encoders)
    future_rows = featured[featured["day_idx"] > last_day].copy()

    future_rows["y_pred"] = np.clip(model.predict(future_rows[feature_cols]), 0, None)
    result = future_rows[["id", "date", "day_idx", "y_pred"]].copy()
    result["run_date"] = run_date or datetime.now().strftime("%Y-%m-%d")

    out_path = reports_dir / f"forecasts_{result['run_date'].iloc[0]}.parquet"
    result.to_parquet(out_path, index=False)
    print(f"Forecasts saved → {out_path} ({len(result)} rows)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate forecasts")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-date", type=str, default=None)
    args = parser.parse_args()
    predict(args.config, args.run_date)


if __name__ == "__main__":
    main()
