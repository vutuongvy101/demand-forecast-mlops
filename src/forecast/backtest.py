"""Rolling-origin backtest: refit every model per fold, report mean ± std."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from forecast.config import DEFAULT_CONFIG_PATH, day_to_int, load_config
from forecast.features import assert_no_leakage, build_features, get_feature_columns
from forecast.metrics import compute_series_metrics
from forecast.models import baseline_forecast, encode_categories, fit_predict_lgbm
from forecast.tracking import log_run_context, setup_mlflow

MODELS = ["seasonal_naive", "moving_average", "lgbm"]


def _evaluate_fold(
    preds: pd.DataFrame,
    long: pd.DataFrame,
    train_end: str,
    weights: pd.Series,
) -> dict[str, float]:
    """Per-series RMSSE/MASE/sMAPE, then mean + sales-weighted aggregate."""
    cutoff = day_to_int(train_end)
    actuals = long[long["day_idx"] > cutoff][["id", "day_idx", "sales"]]
    merged = preds.merge(actuals, on=["id", "day_idx"], how="inner")

    train_sales = {
        sid: grp["sales"].to_numpy()
        for sid, grp in long[long["day_idx"] <= cutoff].groupby("id", sort=False)
    }

    rows = []
    for sid, grp in merged.groupby("id", sort=False):
        y_train = train_sales.get(sid)
        if y_train is None or len(y_train) < 2:
            continue
        m = compute_series_metrics(grp["sales"].to_numpy(), grp["y_pred"].to_numpy(), y_train)
        m["id"] = sid
        rows.append(m)

    per_series = pd.DataFrame(rows).set_index("id")
    w = weights.reindex(per_series.index).fillna(0)
    return {
        "rmsse": float(per_series["rmsse"].mean()),
        "wrmsse": float(np.average(per_series["rmsse"], weights=w / w.sum())),
        "smape": float(per_series["smape"].mean()),
        "mase": float(per_series["mase"].mean()),
    }


def run_backtest(config_path: Path | str | None = None, models: list[str] | None = None) -> Path:
    cfg = load_config(config_path)
    models = models or MODELS
    horizon = cfg["forecast"]["horizon"]

    processed = Path(cfg["paths"]["processed_dir"])
    long = pd.read_parquet(processed / "sales_long.parquet")
    meta = pd.read_parquet(processed / "series_meta.parquet")
    weights = meta.set_index("id")["weight"]

    # Features are computed once; every lag/rolling shift is >= horizon, so the
    # same frame is leakage-safe for all folds (verified per fold below).
    featured = build_features(
        long,
        lags=cfg["features"]["lags"],
        rolling_windows=cfg["features"]["rolling_windows"],
        min_lag=cfg["features"]["min_lag"],
    )
    featured, _ = encode_categories(featured)
    feature_cols = get_feature_columns(cfg) + ["dept_id_enc", "item_id_enc"]

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    setup_mlflow(cfg)
    results = []

    with mlflow.start_run(run_name="backtest"):
        log_run_context(cfg, processed / "sales_long.parquet")

        for model_name in models:
            folds = []
            train_time = 0.0
            for fold_idx, fold in enumerate(cfg["backtest"]["folds"]):
                train_end = fold["train_end"]

                if model_name == "lgbm":
                    assert_no_leakage(
                        featured, feature_cols, train_end, horizon,
                        min_lag=cfg["features"]["min_lag"],
                    )
                    preds, elapsed, _ = fit_predict_lgbm(
                        featured, feature_cols, train_end, horizon, cfg
                    )
                else:
                    preds, elapsed = baseline_forecast(
                        model_name, long, train_end, horizon, cfg
                    )

                fold_metrics = _evaluate_fold(preds, long, train_end, weights)
                fold_metrics["fold"] = fold_idx
                folds.append(fold_metrics)
                train_time += elapsed

            summary = {
                "model": model_name,
                "rmsse_mean": float(np.mean([f["rmsse"] for f in folds])),
                "rmsse_std": float(np.std([f["rmsse"] for f in folds])),
                "wrmsse": float(np.mean([f["wrmsse"] for f in folds])),
                "smape": float(np.mean([f["smape"] for f in folds])),
                "train_time_sec": round(train_time, 2),
                "folds": folds,
            }
            results.append(summary)

            for key in ("rmsse_mean", "rmsse_std", "wrmsse", "smape", "train_time_sec"):
                mlflow.log_metric(f"{model_name}_{key}", summary[key])

        lb_path = reports_dir / "backtest_results.json"
        pd.DataFrame(results).to_json(lb_path, orient="records", indent=2)
        mlflow.log_artifact(str(lb_path))

    print("\n=== Backtest Results (3 folds, H=28) ===")
    for r in results:
        print(
            f"{r['model']:18s} RMSSE={r['rmsse_mean']:.4f}±{r['rmsse_std']:.4f}  "
            f"WRMSSE={r['wrmsse']:.4f}  sMAPE={r['smape']:.1f}  time={r['train_time_sec']}s"
        )
    return lb_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling-origin backtest")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", nargs="+", default=None, choices=MODELS)
    args = parser.parse_args()
    run_backtest(args.config, args.models)


if __name__ == "__main__":
    main()
