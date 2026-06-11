"""Feature engineering boundary tests (direct strategy: shifts >= 28)."""

import pandas as pd

from forecast.features import build_features


def _make_series(n: int = 60, sales_val: float = 1.0, sid: str = "A") -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "id": sid,
            "day_idx": i,
            "sales": sales_val + i * 0.1,
            "wday": (i % 7) + 1,
            "month": 1,
            "snap_CA": 0,
            "event_name_1": "",
            "sell_price": 5.0,
            "dept_id": "FOODS_1",
            "item_id": "FOODS_001",
        })
    return pd.DataFrame(rows)


def test_lag_28_at_boundary():
    df = _make_series(60)
    out = build_features(df, lags=[28], rolling_windows=[7], min_lag=28)
    # Day 28 has no day-0 value to lag from
    row28 = out[out["day_idx"] == 28].iloc[0]
    assert pd.isna(row28["lag_28"])
    # Day 29 lags back to day 1
    row29 = out[out["day_idx"] == 29].iloc[0]
    expected = df[df["day_idx"] == 1]["sales"].iloc[0]
    assert abs(row29["lag_28"] - expected) < 1e-9


def test_rolling_shifted_by_min_lag():
    df = _make_series(60)
    out = build_features(df, lags=[28], rolling_windows=[7], min_lag=28)
    # shift(28).rolling(7) at day 35 averages original days 1-7
    row35 = out[out["day_idx"] == 35].iloc[0]
    window_sales = df[(df["day_idx"] >= 1) & (df["day_idx"] <= 7)]["sales"]
    assert abs(row35["roll_mean_7"] - window_sales.mean()) < 1e-9


def test_series_independence():
    df_a = _make_series(60, sales_val=1.0, sid="A")
    df_b = _make_series(60, sales_val=100.0, sid="B")
    df = pd.concat([df_a, df_b], ignore_index=True)
    out = build_features(df, lags=[28], rolling_windows=[7], min_lag=28)
    a_lag = out[(out["id"] == "A") & (out["day_idx"] == 35)]["lag_28"].iloc[0]
    b_lag = out[(out["id"] == "B") & (out["day_idx"] == 35)]["lag_28"].iloc[0]
    assert a_lag != b_lag
