import streamlit as st
import pandas as pd
from utils import model_utils
REQUIRED_COLUMNS = model_utils.FEATURE_COLUMNS
def render(model):
    st.title("Batch Prediction")

    if model is None:
        st.warning("No trained model found. Please go to the **Train Model** page first.")
        return

    st.markdown(
        "Upload a CSV file containing multiple student records to get "
        "placement predictions for all of them at once."
    )

    with st.expander("Required CSV Columns"):
        st.code(", ".join(REQUIRED_COLUMNS))
        sample = pd.DataFrame([{
            "CGPA": 7.8, "AptitudeScore": 72, "CommunicationSkill": 7,
            "Attendance": 85, "Projects": 4, "Internship": 1,
            "Certifications": 3, "CodingSkill": 8,
        }])
        st.markdown("**Example row:**")
        st.dataframe(sample, use_container_width=True)

        sample_csv = sample.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Sample Template CSV",
            data=sample_csv,
            file_name="sample_template.csv",
            mime="text/csv",
        )

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to begin batch prediction.")
        return

    # --- Read uploaded file safely -------------------------------------
    try:
        batch_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file: {e}")
        return

    if batch_df.empty:
        st.error("The uploaded CSV file is empty.")
        return

    # --- Validate required columns --------------------------------------
    is_valid, missing_cols = model_utils.validate_columns(batch_df)
    if not is_valid:
        st.error(
            "The uploaded CSV is missing required column(s): "
            f"**{', '.join(missing_cols)}**. Please check the template above."
        )
        return

    # --- Handle missing values in required columns ----------------------
    if batch_df[REQUIRED_COLUMNS].isnull().any().any():
        st.warning("Some rows have missing values in required columns. Those rows will be dropped before prediction.")
        batch_df = batch_df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    if batch_df.empty:
        st.error("No valid rows remain after removing missing values.")
        return

    st.success(f"File uploaded successfully! {len(batch_df)} valid rows found.")
    st.dataframe(batch_df.head(10), use_container_width=True)

    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
        try:
            result_df = model_utils.predict_batch(model, batch_df)
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            return

        st.markdown("---")
        st.subheader("Prediction Results")

        placed_count = (result_df["Predicted_Placement"] == "Placed").sum()
        total = len(result_df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Predicted Placed", int(placed_count))
        col3.metric("Predicted Placement Rate", f"{(placed_count/total*100):.1f}%")

        st.dataframe(result_df, use_container_width=True, height=400)

        # --- Download results ---------------------------------------
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Results as CSV",
            data=csv_data,
            file_name="batch_prediction_results.csv",
            mime="text/csv",
            type="primary",
        )
