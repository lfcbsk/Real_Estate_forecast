"""Monitoring dashboard — drift, metrics, data health."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.app.utils import APP_CSS, get_raw_csv_paths, get_train_dir

try:
    import mlflow
except ImportError:
    mlflow = None

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")

st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=60, show_spinner=False)
def _api_get(path: str, params: Optional[Dict] = None) -> Optional[Dict]:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def _get_mlflow_runs(limit: int = 30) -> pd.DataFrame:
    if mlflow is None:
        return pd.DataFrame()
    try:
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("catboost_timeseries")
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit,
        )
        rows = []
        for r in runs:
            row = {
                "run_id": r.info.run_id,
                "start_time": r.info.start_time,
                "run_name": r.info.run_name,
            }
            row.update(r.data.metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
        return df.sort_values("start_time")
    except Exception:
        return pd.DataFrame()


def _drift_status(drift_report: Optional[Dict]) -> tuple[str, str]:
    if drift_report is None:
        return "gray", "Unknown"
    severity = drift_report.get("severity", "low").lower()
    if severity == "high":
        return "critical", "🔴 Drift Detected"
    if severity == "medium":
        return "warning", "🟡 Warning"
    return "ok", "🟢 OK"


def _render_status_pill(status: str, label: str) -> None:
    st.markdown(
        f'<span class="status-pill {status}"><span class="dot"></span>{label}</span>',
        unsafe_allow_html=True,
    )


st.title("📊 Monitoring Dashboard")
st.markdown("Track model performance, drift detection, and on-disk data status.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Monitoring")
    health = _api_get("/health")
    st.markdown("<div class='section-header'>API Status</div>", unsafe_allow_html=True)
    if health:
        st.markdown('<span class="badge badge-green">● Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="badge badge-red">● Offline (local mode)</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-header'>Data Files</div>", unsafe_allow_html=True)
    train_dir = get_train_dir()
    for key, path in get_raw_csv_paths().items():
        icon = "🟢" if path.exists() else "🔴"
        st.caption(f"{icon} `{path.name}`")

    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

drift_report = _api_get("/drift")
sectors_resp = _api_get("/sectors")
metrics_resp = _api_get("/metrics")
mlflow_df = _get_mlflow_runs()

# ── Overview ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>System Overview</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"<div class='metric-card {'good' if health else 'bad'}'><div class='val'>{'OK' if health else 'OFF'}</div>"
        f"<div class='lbl'>API</div></div>",
        unsafe_allow_html=True,
    )

with c2:
    status, label = _drift_status(drift_report)
    score = drift_report.get("drift_score", 0) if drift_report else 0
    cls = "bad" if status == "critical" else ("warn" if status == "warning" else "good")
    st.markdown(
        f"<div class='metric-card {cls}'><div class='val'>{score:.1%}</div>"
        f"<div class='lbl'>Drift Score</div></div>",
        unsafe_allow_html=True,
    )

with c3:
    csv_ok = all(p.exists() for p in get_raw_csv_paths().values())
    st.markdown(
        f"<div class='metric-card {'good' if csv_ok else 'warn'}'><div class='val'>{'OK' if csv_ok else 'MISS'}</div>"
        f"<div class='lbl'>Raw CSVs</div></div>",
        unsafe_allow_html=True,
    )
    st.caption(str(train_dir))

with c4:
    if metrics_resp and metrics_resp.get("status") == "success":
        m = {x["name"]: x["value"] for x in metrics_resp.get("metrics", [])}
        r2 = m.get("holdout_r2")
        val = f"{r2:.3f}" if r2 is not None else "–"
    else:
        val = "–"
    st.markdown(
        f"<div class='metric-card'><div class='val'>{val}</div><div class='lbl'>Latest R²</div></div>",
        unsafe_allow_html=True,
    )

tab_perf, tab_drift, tab_data = st.tabs(["📈 Performance", "🔬 Drift", "🏘️ Data"])

with tab_perf:
    if not mlflow_df.empty:
        available = [c for c in ["holdout_mae", "holdout_rmse", "holdout_r2"] if c in mlflow_df.columns]
        selected = st.multiselect("Metrics", available, default=available[:2] if available else [])
        if selected:
            fig = go.Figure()
            colors = {
                "holdout_mae": "#3b82f6",
                "holdout_rmse": "#ef4444",
                "holdout_r2": "#10b981",
            }
            for m in selected:
                fig.add_trace(
                    go.Scatter(
                        x=mlflow_df["start_time"],
                        y=mlflow_df[m],
                        mode="lines+markers",
                        name=m.replace("holdout_", "").upper(),
                        line=dict(color=colors.get(m, "#64748b")),
                    )
                )
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mlflow_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("No MLflow runs yet. Train the model first.")

with tab_drift:
    if drift_report:
        status, label = _drift_status(drift_report)
        _render_status_pill(status, label)
        st.markdown(f"*{drift_report.get('recommendation', '')}*")
        st.json(drift_report)
    else:
        st.warning("Drift report unavailable. Start the API or run orchestration locally.")

with tab_data:
    if sectors_resp:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Total sectors", sectors_resp["total_sectors"])
        with sc2:
            st.metric("Active", sectors_resp["active_sectors_count"])
        with sc3:
            st.metric("Zero sectors", sectors_resp["zero_sectors_count"])
        st.dataframe(pd.DataFrame(sectors_resp.get("sectors", [])), use_container_width=True)
    else:
        st.info("Sector stats require the API. CSV data is stored under `data/train/`.")

st.divider()
st.caption(f"Last refresh: {datetime.now():%Y-%m-%d %H:%M:%S} · API: `{API_BASE_URL}`")
