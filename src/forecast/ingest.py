"""Download M5 data (or generate synthetic fallback), slice, validate, save parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.config import DEFAULT_CONFIG_PATH, load_config


def _generate_synthetic_m5(
    n_days: int = 1969, n_items: int = 250, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate M5-shaped synthetic data for local dev / CI without Kaggle credentials."""
    rng = np.random.default_rng(seed)
    store_id = "CA_1"
    state_id = "CA"
    cat_id = "FOODS"

    depts = [f"FOODS_{i}" for i in range(1, 4)]
    items = [f"FOODS_{i:03d}" for i in range(1, n_items + 1)]

    # Calendar
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    calendar = pd.DataFrame({
        "date": dates,
        "d": [f"d_{i}" for i in range(1, n_days + 1)],
        "wm_yr_wk": ((dates - dates[0]).days // 7 + 1).astype(int),
        "weekday": dates.day_name(),
        "wday": dates.weekday + 1,
        "month": dates.month,
        "year": dates.year,
        "event_name_1": "",
        "event_type_1": "",
        "event_name_2": "",
        "event_type_2": "",
        "snap_CA": (dates.weekday == 6).astype(int),  # Sunday SNAP
        "snap_TX": 0,
        "snap_WI": 0,
    })
    # Add a few holiday events
    for idx in [100, 500, 900, 1300, 1700]:
        calendar.loc[idx, "event_name_1"] = "Holiday"
        calendar.loc[idx, "event_type_1"] = "National"

    day_cols = [f"d_{i}" for i in range(1, n_days + 1)]

    sales_rows = []
    price_rows = []
    for i, item_id in enumerate(items):
        dept_id = depts[i % len(depts)]
        base = rng.exponential(3 + (i % 5))
        trend = np.linspace(0, 0.5, n_days)
        weekly = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # weaker seasonality
        noise = rng.normal(0, 0.8, n_days)
        snap_boost = 1 + 0.6 * calendar["snap_CA"].values  # stronger SNAP effect
        event_boost = np.where(
            calendar["event_name_1"].astype(str).str.len() > 0, 1.5, 1.0
        )
        demand = np.maximum(0, base * weekly * snap_boost * event_boost + trend + noise)

        # Intermittent series: zero out ~30% of low-volume items
        if i % 4 == 3:
            mask = rng.random(n_days) > 0.4
            demand = demand * mask

        row = {
            "id": f"{item_id}_{store_id}",
            "item_id": item_id,
            "dept_id": dept_id,
            "cat_id": cat_id,
            "store_id": store_id,
            "state_id": state_id,
        }
        for j, col in enumerate(day_cols):
            row[col] = float(demand[j])
        sales_rows.append(row)

        base_price = 2.0 + (i % 10) * 0.3
        for wk in calendar["wm_yr_wk"].unique():
            price_rows.append({
                "store_id": store_id,
                "item_id": item_id,
                "wm_yr_wk": wk,
                "sell_price": base_price + rng.normal(0, 0.1),
            })

    sales = pd.DataFrame(sales_rows)
    prices = pd.DataFrame(price_rows)
    return sales, calendar, prices


def _try_kaggle_download(raw_dir: Path) -> bool:
    try:
        import kagglehub
        path = kagglehub.dataset_download("competitions/m5-forecasting-accuracy")
        src = Path(path)
        for name in ("sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"):
            src_file = src / name
            if not src_file.exists():
                return False
            (raw_dir / name).write_bytes(src_file.read_bytes())
        return True
    except Exception:
        return False


def _load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    return sales, calendar, prices


def _sample_series(sales: pd.DataFrame, n_series: int, seed: int) -> pd.DataFrame:
    """Stratified sample by sales volume quartile."""
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales["_total"] = sales[day_cols].sum(axis=1)
    sales["_quartile"] = pd.qcut(sales["_total"], q=4, labels=["low", "med", "high", "top"])
    parts = []
    per_q = max(1, n_series // 4)
    for _, grp in sales.groupby("_quartile", observed=True):
        parts.append(grp.sample(min(len(grp), per_q), random_state=seed))
    sampled = pd.concat(parts).head(n_series)
    return sampled.drop(columns=["_total", "_quartile"])


def _validate(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> None:
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    assert len(day_cols) >= 1969, f"Need >= 1969 days, got {len(day_cols)}"
    assert not sales[day_cols].isna().any().any(), "Sales contain NaN"
    assert len(calendar) >= 1969, "Calendar too short"
    assert len(prices) > 0, "Prices empty"


def ingest(config_path: Path | str | None = None, force: bool = False) -> Path:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "sales_slice.parquet"
    if output_path.exists() and not force:
        return output_path

    required = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
    if not all((raw_dir / f).exists() for f in required):
        if not _try_kaggle_download(raw_dir):
            print("Kaggle unavailable — generating synthetic M5-shaped data")
            sales, calendar, prices = _generate_synthetic_m5(seed=cfg["dataset"]["random_seed"])
            sales.to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
            calendar.to_csv(raw_dir / "calendar.csv", index=False)
            prices.to_csv(raw_dir / "sell_prices.csv", index=False)

    sales, calendar, prices = _load_raw(raw_dir)

    # Slice to store + category
    mask = (sales["store_id"] == cfg["dataset"]["store_id"]) & (
        sales["cat_id"] == cfg["dataset"]["category"]
    )
    sliced = sales[mask].copy()
    if len(sliced) == 0:
        raise ValueError(
            f"No series for {cfg['dataset']['store_id']} / {cfg['dataset']['category']}"
        )

    sliced = _sample_series(sliced, cfg["dataset"]["n_series"], cfg["dataset"]["random_seed"])
    _validate(sliced, calendar, prices)

    # Long format for modelling
    day_cols = [c for c in sliced.columns if c.startswith("d_")]
    long = sliced.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    from forecast.config import day_to_int as _day_to_int

    long["day_idx"] = long["d"].map(_day_to_int)
    long = long.merge(calendar, on="d", how="left")
    long = long.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    long["sell_price"] = long.groupby("id")["sell_price"].ffill().bfill()
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["id", "day_idx"]).reset_index(drop=True)

    meta = sliced[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    meta["weight"] = sliced[day_cols].sum(axis=1)

    long.to_parquet(processed_dir / "sales_long.parquet", index=False)
    meta.to_parquet(processed_dir / "series_meta.parquet", index=False)
    calendar.to_parquet(processed_dir / "calendar.parquet", index=False)
    sliced.to_parquet(output_path, index=False)

    print(f"Ingested {len(sliced)} series, {len(day_cols)} days → {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and slice M5 data")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ingest(args.config, force=args.force)


if __name__ == "__main__":
    main()
