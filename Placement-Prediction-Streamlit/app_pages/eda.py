import streamlit as st
import pandas as pd
from utils import visualizations as viz
def render(df: pd.DataFrame):
    st.title("Exploratory Data Analysis")
    st.markdown("Visual insights into what drives student placement outcomes.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.placement_distribution_pie(df), use_container_width=True)
    with col2:
        st.plotly_chart(viz.placement_rate_by_category(df, "Internship"), use_container_width=True)

    st.markdown("---")
    st.plotly_chart(viz.correlation_heatmap(df), use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Feature Distribution Explorer")
    numeric_cols = ["CGPA", "AptitudeScore", "CommunicationSkill", "Attendance",
                     "Projects", "Certifications", "CodingSkill"]
    selected_col = st.selectbox("Choose a feature to explore", numeric_cols)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.numeric_histogram(df, selected_col), use_container_width=True)
    with col2:
        st.plotly_chart(viz.box_plot_by_placement(df, selected_col), use_container_width=True)

    st.markdown("---")
    st.subheader("Relationship Explorer")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis feature", numeric_cols, index=0, key="x_axis")
    with col2:
        y_axis = st.selectbox("Y-axis feature", numeric_cols, index=1, key="y_axis")

    st.plotly_chart(viz.scatter_by_placement(df, x_axis, y_axis), use_container_width=True)
