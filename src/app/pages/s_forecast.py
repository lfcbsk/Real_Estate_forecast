"""Multi-month sector forecast from merged training data."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.utils import APP_CSS, get_raw_csv_paths, load_merged_training_data, load_production_model
from src.pipeline.predict import forecast_next_year

st.markdown(APP_CSS, unsafe_allow_html=True)

st.title("📈 Sector Forecast")
st.markdown(
    "Generate a **recursive multi-month forecast** using merged data from "
    "`data/train/` and the production ONNX model."
)

paths = get_raw_csv_paths()
missing = [p.name for p in paths.values() if not p.exists()]

if missing:
    st.warning(f"Missing CSV files: {', '.join(missing)}. " "Upload data on the **📤 Upload & Predict** page first.")

col1, col2 = st.columns([1, 2])
with col1:
    n_months = st.slider("Months to forecast", min_value=1, max_value=36, value=12)
with col2:
    sector_filter = st.text_input(
        "Filter sectors (comma-separated, empty = all)",
        placeholder="e.g. 1, 5, 12",
    )

if st.button("📊 Generate Forecast", type="primary", use_container_width=True):
    with st.spinner("Loading data → recursive forecast..."):
        try:
            df = load_merged_training_data()
            registry = load_production_model()

            results = {"zero_sectors": registry.zero_sectors}
            forecast_df = forecast_next_year(df, registry, results, n_months=n_months)
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            if sector_filter.strip():
                sectors = [int(s.strip()) for s in sector_filter.split(",") if s.strip()]
                forecast_df = forecast_df[forecast_df["sector"].isin(sectors)]

            st.success(f"Generated {len(forecast_df):,} predictions for {n_months} month(s).")

            st.markdown("<div class='section-header'>Forecast Chart</div>", unsafe_allow_html=True)

            chart_df = forecast_df
            if forecast_df["sector"].nunique() > 15:
                top = forecast_df.groupby("sector")["pred_amount"].sum().sort_values(ascending=False).head(15).index
                chart_df = forecast_df[forecast_df["sector"].isin(top)]
                st.caption("Showing top 15 sectors by total predicted volume.")

            fig = px.line(
                chart_df.sort_values("date"),
                x="date",
                y="pred_amount",
                color="sector",
                markers=True,
                labels={"pred_amount": "Predicted Amount", "date": "Date", "sector": "Sector"},
            )
            fig.update_layout(template="plotly_white", height=480)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='section-header'>Total Volume by Month</div>", unsafe_allow_html=True)
            monthly = forecast_df.groupby("date", as_index=False)["pred_amount"].sum()
            fig2 = px.bar(monthly, x="date", y="pred_amount")
            fig2.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("<div class='section-header'>Forecast Data</div>", unsafe_allow_html=True)
            st.dataframe(
                forecast_df.sort_values(["sector", "date"]).reset_index(drop=True),
                use_container_width=True,
            )

            st.download_button(
                "⬇️ Download forecast (CSV)",
                data=forecast_df.to_csv(index=False).encode("utf-8"),
                file_name=f"forecast_{n_months}m.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Forecast failed: {e}")
else:
    st.info("Configure options above and click **Generate Forecast**.")
