"""Streamlit dashboard: forecast plot, leaderboard, drift status."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"
PROCESSED = ROOT / "data" / "processed"


@st.cache_data
def load_data():
    long = pd.read_parquet(PROCESSED / "sales_long.parquet")
    meta = pd.read_parquet(PROCESSED / "series_meta.parquet")
    return long, meta


def load_backtest() -> pd.DataFrame | None:
    path = REPORTS / "backtest_results.json"
    if not path.exists():
        return None
    return pd.DataFrame(json.loads(path.read_text()))


def load_forecasts() -> pd.DataFrame | None:
    files = sorted(REPORTS.glob("forecasts_*.parquet"))
    if not files:
        return None
    return pd.read_parquet(files[-1])


def load_drift() -> dict | None:
    path = REPORTS / "drift_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    st.set_page_config(page_title="Demand Forecast MLOps", layout="wide")
    st.title("Demand Forecast Dashboard")

    if not (PROCESSED / "sales_long.parquet").exists():
        st.warning("No data found. Run `make data && make backtest && make train && make predict`")
        return

    long, meta = load_data()
    tab1, tab2, tab3 = st.tabs(["Forecast vs Actuals", "Leaderboard", "Drift Status"])

    with tab1:
        series_ids = sorted(long["id"].unique())
        selected = st.selectbox("Select series", series_ids)
        hist = long[long["id"] == selected].tail(90)
        fig = px.line(hist, x="date", y="sales", title=f"History: {selected}")

        forecasts = load_forecasts()
        if forecasts is not None:
            fc = forecasts[forecasts["id"] == selected]
            if len(fc) > 0:
                fig.add_scatter(
                    x=fc["date"], y=fc["y_pred"], mode="lines+markers",
                    name="Forecast", line=dict(dash="dash"),
                )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        bt = load_backtest()
        if bt is not None:
            display_cols = ["model", "rmsse_mean", "rmsse_std", "wrmsse", "smape", "train_time_sec"]
            cols = [c for c in display_cols if c in bt.columns]
            st.dataframe(bt[cols].style.format(precision=4), use_container_width=True)
        else:
            st.info("Run `make backtest` to generate leaderboard.")

    with tab3:
        drift = load_drift()
        if drift is not None:
            status = drift.get("overall_status", "unknown")
            color = {"ok": "green", "warn": "orange", "flag": "red"}.get(status, "gray")
            st.markdown(f"**Overall status:** :{color}[{status.upper()}]")
            st.caption(f"Last check: {drift.get('timestamp', 'N/A')}")

            if drift.get("feature_drift"):
                psi_df = pd.DataFrame(drift["feature_drift"]).T.reset_index()
                psi_df.columns = ["feature", "psi", "status"]
                st.subheader("Feature Drift (PSI)")
                st.dataframe(psi_df, use_container_width=True)

            perf = drift.get("performance", {})
            st.subheader("Performance Drift")
            st.json(perf)
        else:
            st.info("Run `make drift` to generate drift report.")


if __name__ == "__main__":
    main()
