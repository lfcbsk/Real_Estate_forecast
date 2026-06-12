"""
streamlit_app.py – House Transaction Forecast · Home
Run: uv run streamlit run src/app/streamlit_app.py
"""

import mlflow
import streamlit as st

from src.app.utils import APP_CSS, get_raw_csv_paths, get_train_dir

st.set_page_config(
    page_title="House Forecast",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)


def _show_mlflow_sidebar():
    try:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("catboost_timeseries")
        if exp is None:
            st.caption("No MLflow experiment yet.")
            return
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.mlflow.runName = 'pipeline_run'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            st.caption("No pipeline run yet.")
            return
        run = runs[0]
        m = run.data.metrics

        def _fmt(key, default="–"):
            v = m.get(key)
            return f"{v:.4f}" if v is not None else default

        for lbl, key in [
            ("Competition Score", "test_competition_score"),
            ("MAE", "test_mae"),
            ("RMSE", "test_rmse"),
            ("MAPE", "test_mape"),
            ("R²", "test_r2"),
        ]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:13px;padding:4px 0;border-bottom:1px solid #1e293b'>"
                f"<span style='color:#64748b'>{lbl}</span>"
                f"<span style='color:#f1f5f9;font-family:JetBrains Mono'>{_fmt(key)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.caption(f"Run: `{run.info.run_id[:8]}…`")
    except Exception:
        st.caption("MLflow unavailable.")


with st.sidebar:
    st.markdown("## 🏠 House Forecast")
    st.markdown(
        "<div style='font-size:12px;color:#475569;margin-bottom:20px'>" "Real-estate transaction forecasting</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-header'>Baseline Metrics</div>", unsafe_allow_html=True)
    _show_mlflow_sidebar()

    st.markdown("<div class='section-header'>Data Store</div>", unsafe_allow_html=True)
    st.caption(f"`{get_train_dir()}`")
    for path in get_raw_csv_paths().values():
        icon = "✅" if path.exists() else "⬜"
        st.caption(f"{icon} {path.name}")

st.title("🏠 House Transaction Forecast")

st.markdown("""
### Workflow

1. **📤 Upload & Predict** — upload 3 raw CSV files → merged into `data/train/` → predict
2. **📈 Sector Forecast** — recursive multi-month forecast from merged data
3. **📊 Monitoring** — drift, metrics, data health

Use the **sidebar** to navigate between pages.
""")

st.markdown("<div class='section-header'>Quick Start</div>", unsafe_allow_html=True)

st.code(
    """# 1. Install & train (first time)
uv pip install -e ".[dev]"
uv run python -m src.pipeline.training

# 2. Launch dashboard
uv run streamlit run src/app/streamlit_app.py""",
    language="bash",
)

st.markdown("""
### Data storage
Uploaded raw files are **appended** to CSVs under `data/train/`.
Rows with the same `(month, sector)` are **overwritten** by the latest upload.
There is no separate database — CSV files are the source of truth.
""")
