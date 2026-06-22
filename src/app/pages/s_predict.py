"""Upload 3 raw CSV files → merge into data/train → feature engineer → predict."""

import streamlit as st

from src.app.utils import (
    APP_CSS,
    RAW_FILE_SPECS,
    get_train_dir,
    predict_on_uploaded_months,
    read_uploaded_csv,
    save_uploaded_raw_files,
)

st.markdown(APP_CSS, unsafe_allow_html=True)

st.title("📤 Upload & Predict")
st.markdown("""
Upload **3 raw competition CSV files**. New rows are **merged into** `data/train/` —
duplicate `(month, sector)` keys are **overwritten** with the latest upload.

Then the app runs: **merge → feature engineering → load ONNX model → predict**.
""")

st.markdown(
    "<div class='upload-hint'>"
    "Required files (same format as Kaggle competition):<br>"
    "• <code>new_house_transactions.csv</code><br>"
    "• <code>new_house_transactions_nearby_sectors.csv</code><br>"
    "• <code>pre_owned_house_transactions.csv</code><br>"
    "Each file must have <code>month</code> and <code>sector</code> columns."
    "</div>",
    unsafe_allow_html=True,
)

train_dir = get_train_dir()
st.caption(f"Data directory: `{train_dir}`")

col1, col2, col3 = st.columns(3)
with col1:
    file_main = st.file_uploader(
        RAW_FILE_SPECS["main"]["label"],
        type=["csv", "xlsx", "xls"],
        key="upload_main",
    )
with col2:
    file_nearby = st.file_uploader(
        RAW_FILE_SPECS["nearby"]["label"],
        type=["csv", "xlsx", "xls"],
        key="upload_nearby",
    )
with col3:
    file_pre = st.file_uploader(
        RAW_FILE_SPECS["pre"]["label"],
        type=["csv", "xlsx", "xls"],
        key="upload_pre",
    )

all_uploaded = file_main and file_nearby and file_pre

if all_uploaded:
    st.markdown("<div class='section-header'>Preview</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    try:
        with p1:
            preview_main = read_uploaded_csv(file_main)
            st.caption(RAW_FILE_SPECS["main"]["filename"])
            st.dataframe(preview_main.head(5), use_container_width=True)
        with p2:
            preview_nearby = read_uploaded_csv(file_nearby)
            st.caption(RAW_FILE_SPECS["nearby"]["filename"])
            st.dataframe(preview_nearby.head(5), use_container_width=True)
        with p3:
            preview_pre = read_uploaded_csv(file_pre)
            st.caption(RAW_FILE_SPECS["pre"]["filename"])
            st.dataframe(preview_pre.head(5), use_container_width=True)

        for f in (file_main, file_nearby, file_pre):
            f.seek(0)
    except Exception as e:
        st.error(f"Could not read uploaded files: {e}")
        all_uploaded = False

if all_uploaded and st.button("🚀 Merge, Process & Predict", type="primary", use_container_width=True):
    with st.spinner("Saving to CSV → merging → feature engineering → predicting..."):
        try:
            main_df = read_uploaded_csv(file_main)
            nearby_df = read_uploaded_csv(file_nearby)
            pre_df = read_uploaded_csv(file_pre)

            save_stats = save_uploaded_raw_files(main_df, nearby_df, pre_df)

            st.success("Raw data saved to `data/train/` (append + overwrite duplicates).")

            st.markdown(
                "<div class='section-header'>CSV Update Summary</div>",
                unsafe_allow_html=True,
            )
            sc1, sc2, sc3 = st.columns(3)
            for col, key in zip([sc1, sc2, sc3], ["main", "nearby", "pre"]):
                s = save_stats[key]
                with col:
                    st.markdown(
                        f"<div class='metric-card'>"
                        f"<div class='val'>{s['total_rows']:,}</div>"
                        f"<div class='lbl'>{RAW_FILE_SPECS[key]['filename']}</div></div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"+{s['added']} new · {s['updated']} updated")

            upload_months = main_df["month"]
            result_df = predict_on_uploaded_months(upload_months)

            st.markdown("<div class='section-header'>Predictions</div>", unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(
                    f"<div class='metric-card'><div class='val'>{len(result_df):,}</div>"
                    f"<div class='lbl'>Rows Predicted</div></div>",
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    f"<div class='metric-card good'><div class='val'>{result_df['predicted_amount'].mean():,.0f}</div>"
                    f"<div class='lbl'>Avg Predicted Amount</div></div>",
                    unsafe_allow_html=True,
                )
            with mc3:
                n_months = result_df["date"].nunique()
                st.markdown(
                    f"<div class='metric-card'><div class='val'>{n_months}</div>"
                    f"<div class='lbl'>Months Covered</div></div>",
                    unsafe_allow_html=True,
                )

            st.dataframe(result_df, use_container_width=True)

            st.download_button(
                "⬇️ Download predictions (CSV)",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions_upload.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except FileNotFoundError as e:
            st.error(str(e))
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

elif not all_uploaded:
    st.info("Upload all 3 raw CSV files to enable prediction.")

st.divider()
with st.expander("ℹ️ How data is stored"):
    st.markdown("""
1. Each upload is **appended** to the matching file under `data/train/`.
2. If `(month, sector)` already exists, the **new row replaces** the old one.
3. On the next forecast or orchestration run, the updated CSVs are used automatically.
4. No separate database — **CSV files are the source of truth**.
        """)
