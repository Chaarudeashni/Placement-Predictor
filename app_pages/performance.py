import streamlit as st
import pandas as pd

from utils import model_utils
from utils import visualizations as viz


def render(df: pd.DataFrame, model):
    st.title("Model Performance")

    if model is None:
        st.warning("No trained model found. Please go to the **Train Model** page first.")
        return

    st.markdown(
        "Evaluation metrics for the currently trained Random Forest model."
    )
    st.markdown("---")
    cached = st.session_state.get("last_training_results")

    if cached is not None:
        metrics = cached["metrics"]
        confusion = cached["confusion"]
        report = cached["report"]
   
        st.caption("Showing metrics from your most recent training run in this session.")
    else:
        st.caption("No training run found in this session — evaluating the saved model on a fresh test split.")
        results = model_utils.train_model(df)
        metrics = results["metrics"]
        confusion = results["confusion"]
        report = results["report"]
  
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.confusion_matrix_heatmap(confusion), use_container_width=True)

    st.markdown("---")
    with st.expander("Full Classification Report"):
        st.text(report)

