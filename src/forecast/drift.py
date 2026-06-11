"""Drift monitoring: PSI on top features + performance vs. backtest reference.

Performance check reuses backtest folds: the last fold covers the most recent
28 observed days; its RMSSE is compared to the mean of the earlier folds.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.config import DEFAULT_CONFIG_PATH, load_config
from forecast.features import build_features, get_feature_columns


def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI between two distributions (0 = identical, > 0.25 = significant shift)."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, n_bins + 1)))
    if len(breakpoints) < 2:
        return 0.0

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def check_drift(config_path: Path | str | None = None) -> Path:
    cfg = load_config(config_path)
    processed = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    long = pd.read_parquet(processed / "sales_long.parquet")
    featured = build_features(
        long,
        lags=cfg["features"]["lags"],
        rolling_windows=cfg["features"]["rolling_windows"],
        min_lag=cfg["features"]["min_lag"],
    )

    # 1. Feature drift: training period vs. most recent horizon window
    horizon = cfg["forecast"]["horizon"]
    n_days = long["day_idx"].max()
    train_mask = featured["day_idx"] <= n_days - 2 * horizon
    recent_mask = featured["day_idx"] > n_days - horizon

    psi_results = {}
    for col in get_feature_columns(cfg)[: cfg["drift"]["top_features"]]:
        psi = population_stability_index(
            featured.loc[train_mask, col].to_numpy(dtype=float),
            featured.loc[recent_mask, col].to_numpy(dtype=float),
        )
        status = "ok"
        if psi >= cfg["drift"]["psi_flag"]:
            status = "flag"
        elif psi >= cfg["drift"]["psi_warn"]:
            status = "warn"
        psi_results[col] = {"psi": round(psi, 4), "status": status}

    # 2. Performance drift: last backtest fold (most recent 28 days) vs. earlier folds
    performance = {"recent_rmsse": None, "reference_rmsse": None, "ratio": None, "status": "ok"}
    backtest_path = reports_dir / "backtest_results.json"
    if backtest_path.exists():
        results = json.loads(backtest_path.read_text())
        lgbm = next((r for r in results if r["model"] == "lgbm"), None)
        if lgbm and len(lgbm.get("folds", [])) >= 2:
            folds = lgbm["folds"]
            recent = folds[-1]["rmsse"]
            reference = float(np.mean([f["rmsse"] for f in folds[:-1]]))
            ratio = recent / reference if reference else None
            performance = {
                "recent_rmsse": round(recent, 4),
                "reference_rmsse": round(reference, 4),
                "ratio": round(ratio, 4) if ratio else None,
                "status": (
                    "flag"
                    if ratio and ratio > cfg["drift"]["performance_ratio_flag"]
                    else "ok"
                ),
            }

    statuses = [v["status"] for v in psi_results.values()] + [performance["status"]]
    overall = "flag" if "flag" in statuses else "warn" if "warn" in statuses else "ok"

    report = {
        "timestamp": datetime.now().isoformat(),
        "feature_drift": psi_results,
        "performance": performance,
        "overall_status": overall,
    }
    out_path = reports_dir / "drift_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Drift report → {out_path} (status: {overall})")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run drift checks")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    check_drift(args.config)


if __name__ == "__main__":
    main()
