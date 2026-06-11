"""Feature engineering: lags, rolling stats, calendar, price — leakage-safe shifts."""

from __future__ import annotations

import pandas as pd

from forecast.config import day_to_int


def build_features(
    df: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
    min_lag: int,
) -> pd.DataFrame:
    """
    Build ML features with shifts >= min_lag to prevent leakage.

    Direct forecasting strategy: with min_lag >= horizon, every lag/rolling
    feature for a test-period row references only training-period demand.
    """
    out = df.sort_values(["id", "day_idx"]).reset_index(drop=True)
    grouped = out.groupby("id", group_keys=False)

    for lag in lags:
        assert lag >= min_lag, f"Lag {lag} < min_lag {min_lag} — leakage risk"
        out[f"lag_{lag}"] = grouped["sales"].shift(lag)

    for window in rolling_windows:
        shift = min_lag
        out[f"roll_mean_{window}"] = grouped["sales"].transform(
            lambda s: s.shift(shift).rolling(window, min_periods=1).mean()
        )
        out[f"roll_std_{window}"] = grouped["sales"].transform(
            lambda s: s.shift(shift).rolling(window, min_periods=1).std()
        ).fillna(0)

    # Calendar features
    out["dow"] = out["wday"]
    out["month_feat"] = out["month"]
    out["snap"] = out["snap_CA"].fillna(0).astype(int)
    out["is_event"] = (
        out["event_name_1"].fillna("").astype(str).str.len() > 0
    ).astype(int)

    # Price features
    out["price"] = out["sell_price"]
    out["price_vs_4wk"] = grouped["sell_price"].transform(
        lambda s: s - s.shift(min_lag).rolling(28, min_periods=1).mean()
    )

    # Static categoricals
    out["dept_id"] = out["dept_id"].astype("category")
    out["item_id"] = out["item_id"].astype("category")

    return out


def get_feature_columns(cfg: dict) -> list[str]:
    lags = cfg["features"]["lags"]
    windows = cfg["features"]["rolling_windows"]
    cols = [f"lag_{lag}" for lag in lags]
    for w in windows:
        cols.extend([f"roll_mean_{w}", f"roll_std_{w}"])
    cols.extend(["dow", "month_feat", "snap", "is_event", "price", "price_vs_4wk"])
    return cols


def get_train_cutoff_mask(df: pd.DataFrame, train_end_day: str) -> pd.Series:
    cutoff = day_to_int(train_end_day)
    return df["day_idx"] <= cutoff


def assert_no_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_end_day: str,
    horizon: int,
    min_lag: int = 28,
) -> None:
    """
    Verify no test-period feature can use information after the train cutoff.

    Direct strategy: the worst case is the LAST test day (cutoff + horizon).
    A lag feature there references day cutoff + horizon - lag, which stays
    within the training period iff lag >= horizon. Rolling features are
    shifted by min_lag, so the same bound applies.
    """
    cutoff = day_to_int(train_end_day)
    last_test_day = cutoff + horizon

    for col in feature_cols:
        if col.startswith("lag_"):
            lag = int(col.split("_")[1])
            source_day = last_test_day - lag
            if source_day > cutoff:
                raise AssertionError(
                    f"Leakage in {col}: test day {last_test_day} would use "
                    f"source day {source_day} > cutoff {cutoff} (lag {lag} < horizon {horizon})"
                )
        if col.startswith("roll_") and min_lag < horizon:
            raise AssertionError(
                f"Leakage in {col}: rolling shift {min_lag} < horizon {horizon}"
            )
