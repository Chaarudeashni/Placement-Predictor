import streamlit as st
import pandas as pd
def render(df: pd.DataFrame):
    st.title("Dataset Explorer")
    st.markdown("Browse, filter, and inspect the placement dataset used to train the model.")
    st.markdown("---")
    with st.expander("Filter Data", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            cgpa_range = st.slider(
                "CGPA range", float(df["CGPA"].min()), float(df["CGPA"].max()),
                (float(df["CGPA"].min()), float(df["CGPA"].max())),
            )
        with col2:
            placement_filter = st.selectbox(
                "Placement Status", options=["All", "Placed", "Not Placed"]
            )
        with col3:
            internship_filter = st.selectbox(
                "Internship", options=["All", "Yes", "No"]
            )

    filtered_df = df[(df["CGPA"] >= cgpa_range[0]) & (df["CGPA"] <= cgpa_range[1])]

    if placement_filter == "Placed":
        filtered_df = filtered_df[filtered_df["Placement"] == 1]
    elif placement_filter == "Not Placed":
        filtered_df = filtered_df[filtered_df["Placement"] == 0]

    if internship_filter == "Yes":
        filtered_df = filtered_df[filtered_df["Internship"] == 1]
    elif internship_filter == "No":
        filtered_df = filtered_df[filtered_df["Internship"] == 0]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} records**")
    st.dataframe(filtered_df, use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe().T, use_container_width=True)

    st.markdown("---")
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_placement_data.csv",
        mime="text/csv",
    )
