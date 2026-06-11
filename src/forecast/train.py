"""Train production LightGBM on all data, save artifact, log to MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from forecast.config import DEFAULT_CONFIG_PATH, load_config
from forecast.features import build_features, get_feature_columns
from forecast.models import encode_categories, make_lgbm
from forecast.tracking import log_run_context, setup_mlflow


def train(config_path: Path | str | None = None) -> Path:
    cfg = load_config(config_path)
    processed = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    long = pd.read_parquet(processed / "sales_long.parquet")
    featured = build_features(
        long,
        lags=cfg["features"]["lags"],
        rolling_windows=cfg["features"]["rolling_windows"],
        min_lag=cfg["features"]["min_lag"],
    )
    featured, encoders = encode_categories(featured)
    feature_cols = get_feature_columns(cfg) + ["dept_id_enc", "item_id_enc"]

    first_lag = f"lag_{cfg['features']['lags'][0]}"
    train_df = featured.dropna(subset=[first_lag])

    setup_mlflow(cfg)
    model_path = models_dir / "lgbm_model.joblib"

    with mlflow.start_run(run_name="train"):
        log_run_context(cfg, processed / "sales_long.parquet")
        mlflow.log_params({
            "horizon": cfg["forecast"]["horizon"],
            "n_series": cfg["dataset"]["n_series"],
            **{f"lgbm_{k}": v for k, v in cfg["models"]["lgbm"].items()},
        })

        model = make_lgbm(cfg["models"]["lgbm"])
        model.fit(train_df[feature_cols], train_df["sales"])

        joblib.dump(
            {"model": model, "feature_cols": feature_cols, "encoders": encoders},
            model_path,
        )
        mlflow.log_artifact(str(model_path))

    print(f"Model saved → {model_path}")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
