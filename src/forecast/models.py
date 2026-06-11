"""All models in one place: two cheap baselines + direct-strategy LightGBM.

Direct strategy: every feature uses data >= 28 days (= horizon) old, so a single
model predicts all 28 future days in one shot. No recursive loop, no leakage.
"""

from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast.config import day_to_int

CATEGORICAL_COLS = ("dept_id", "item_id")


def baseline_forecast(
    name: str,
    df: pd.DataFrame,
    train_end: str,
    horizon: int,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    """Seasonal naive (repeat last week) or moving average (flat 28-day mean)."""
    cutoff = day_to_int(train_end)
    train = df[df["day_idx"] <= cutoff]
    future_days = np.arange(cutoff + 1, cutoff + horizon + 1)

    t0 = time.perf_counter()
    preds = []
    if name == "seasonal_naive":
        season = cfg["models"]["seasonal_naive"]["season_length"]
        tail = train.groupby("id", sort=False).tail(season)
        for sid, grp in tail.groupby("id", sort=False):
            vals = grp.sort_values("day_idx")["sales"].to_numpy()
            y = np.tile(vals, horizon // len(vals) + 1)[:horizon]
            preds.append(pd.DataFrame({"id": sid, "day_idx": future_days, "y_pred": y}))
    elif name == "moving_average":
        tail = train.groupby("id", sort=False).tail(28)
        means = tail.groupby("id", sort=False)["sales"].mean()
        for sid, mean_val in means.items():
            preds.append(pd.DataFrame({"id": sid, "day_idx": future_days, "y_pred": mean_val}))
    else:
        raise ValueError(f"Unknown baseline: {name}")

    elapsed = time.perf_counter() - t0
    out = pd.concat(preds, ignore_index=True)
    out["model"] = name
    return out, elapsed


def encode_categories(
    df: pd.DataFrame,
    encoders: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Integer-encode static categoricals. Categories are fixed per dataset,
    so encoding over the full frame introduces no leakage."""
    out = df.copy()
    if encoders is None:
        encoders = {}
        for col in CATEGORICAL_COLS:
            codes, uniques = pd.factorize(out[col].astype(str))
            out[f"{col}_enc"] = codes
            encoders[col] = {v: i for i, v in enumerate(uniques)}
    else:
        for col in CATEGORICAL_COLS:
            out[f"{col}_enc"] = (
                out[col].astype(str).map(encoders[col]).fillna(-1).astype(int)
            )
    return out, encoders


def make_lgbm(params: dict[str, Any]) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        verbose=-1,
    )


def fit_predict_lgbm(
    featured: pd.DataFrame,
    feature_cols: list[str],
    train_end: str,
    horizon: int,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, float, lgb.LGBMRegressor]:
    """Fit on rows <= train_end, predict the next `horizon` days directly."""
    cutoff = day_to_int(train_end)
    first_lag = f"lag_{cfg['features']['lags'][0]}"

    train = featured[featured["day_idx"] <= cutoff].dropna(subset=[first_lag])
    test = featured[
        (featured["day_idx"] > cutoff) & (featured["day_idx"] <= cutoff + horizon)
    ]

    model = make_lgbm(cfg["models"]["lgbm"])
    t0 = time.perf_counter()
    model.fit(train[feature_cols], train["sales"])
    elapsed = time.perf_counter() - t0

    preds = test[["id", "day_idx"]].copy()
    preds["y_pred"] = np.clip(model.predict(test[feature_cols]), 0, None)
    preds["model"] = "lgbm"
    return preds, elapsed, model


def make_future_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Skeleton rows for the next `horizon` days after the last observed day.
    Calendar values (snap/event) are copied from 28 days prior; price is carried
    forward. Lag/rolling features computed on these rows only touch real history."""
    last_day = int(df["day_idx"].max())
    rows = []
    for sid, grp in df.groupby("id", sort=False):
        grp = grp.sort_values("day_idx")
        last = grp.iloc[-1]
        hist = grp.set_index("day_idx")
        for k in range(1, horizon + 1):
            day = last_day + k
            date = last["date"] + pd.Timedelta(days=k)
            src = hist.loc[day - 28] if (day - 28) in hist.index else last
            rows.append({
                "id": sid,
                "item_id": last["item_id"],
                "dept_id": last["dept_id"],
                "day_idx": day,
                "date": date,
                "sales": np.nan,
                "wday": date.weekday() + 1,
                "month": date.month,
                "snap_CA": src["snap_CA"],
                "event_name_1": src["event_name_1"],
                "sell_price": last["sell_price"],
            })
    return pd.DataFrame(rows)
