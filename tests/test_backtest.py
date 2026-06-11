"""Backtest integrity tests — leakage is the critical one."""

import pandas as pd

from forecast.features import assert_no_leakage, build_features, get_feature_columns

CFG = {"features": {"lags": [28, 35, 42], "rolling_windows": [7, 28], "min_lag": 28}}


def _make_long_df(n_days: int = 2000) -> pd.DataFrame:
    rows = []
    for d in range(1, n_days + 1):
        rows.append({
            "id": "series_1",
            "day_idx": d,
            "sales": float(d),
            "wday": (d % 7) + 1,
            "month": 1,
            "snap_CA": 0,
            "event_name_1": "",
            "sell_price": 5.0,
            "dept_id": "FOODS_1",
            "item_id": "FOODS_001",
        })
    return pd.DataFrame(rows)


def test_no_leakage():
    """No test-period feature may use information after the train cutoff.

    Direct strategy with min_lag = horizon = 28: a feature at test day t
    references sales at t - 28 or earlier, which is always <= cutoff for
    any test day within the 28-day horizon.
    """
    df = _make_long_df()
    featured = build_features(
        df,
        lags=CFG["features"]["lags"],
        rolling_windows=CFG["features"]["rolling_windows"],
        min_lag=CFG["features"]["min_lag"],
    )
    feature_cols = get_feature_columns(CFG)

    # Must not raise for any fold
    for train_end in ("d1885", "d1913", "d1941"):
        assert_no_leakage(featured, feature_cols, train_end, horizon=28, min_lag=28)


def test_lag_values_come_from_training_period():
    """Sales are 1..N, so lag values are directly checkable: lag_28 at day t = t - 28."""
    df = _make_long_df()
    featured = build_features(
        df, lags=[28], rolling_windows=[7], min_lag=28
    )
    cutoff = 1885
    test = featured[(featured["day_idx"] > cutoff) & (featured["day_idx"] <= cutoff + 28)]
    assert len(test) == 28
    for _, row in test.iterrows():
        source_day = row["day_idx"] - 28
        assert source_day <= cutoff
        assert abs(row["lag_28"] - float(source_day)) < 1e-9


def test_leakage_detected_for_short_lags():
    """A lag shorter than the horizon must be caught by the leakage check."""
    df = _make_long_df()
    featured = build_features(df, lags=[7], rolling_windows=[7], min_lag=7)
    try:
        assert_no_leakage(featured, ["lag_7"], "d1885", horizon=28, min_lag=7)
    except AssertionError:
        return
    raise AssertionError("Expected leakage to be detected for lag_7 with horizon 28")
