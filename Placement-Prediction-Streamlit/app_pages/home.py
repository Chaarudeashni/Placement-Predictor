import streamlit as st
import pandas as pd
def render(df: pd.DataFrame, model_exists: bool):
    st.title("Placement Prediction Dashboard")
    st.markdown("---")
    total_students = len(df)
    placed_count = int(df["Placement"].sum())
    placement_rate = (placed_count / total_students * 100) if total_students else 0
    avg_cgpa = df["CGPA"].mean() if total_students else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", f"{total_students:,}")
    with col2:
        st.metric("Students Placed", f"{placed_count:,}")
    with col3:
        st.metric("Placement Rate", f"{placement_rate:.1f}%")
    with col4:
        st.metric("Average CGPA", f"{avg_cgpa:.2f}")

    st.markdown("---")
    if model_exists:
        st.success("A trained model is available. You can head to the **Prediction** or **Batch Prediction** page.")
    else:
        st.warning("No trained model found yet. Please visit the **Train Model** page first to train one.")

    st.markdown(
        """
        Predict student placement outcomes instantly using Machine Learning — powered by academic performance, skills, and experience.
        """
    )