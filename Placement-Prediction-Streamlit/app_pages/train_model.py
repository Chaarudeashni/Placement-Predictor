import time
import streamlit as st
import pandas as pd
from pathlib import Path

from utils import model_utils
from utils import visualizations as viz

def render(df: pd.DataFrame, model_path: Path):
    st.title("Train Model")
    st.markdown(
        "Train a **Random Forest Classifier** on the placement dataset. "
        "You can adjust hyperparameters below before training."
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20, step=5) / 100
    with col2:
        n_estimators = st.slider("Number of trees (n_estimators)", 50, 500, 300, step=50)
    with col3:
        max_depth = st.slider("Max tree depth", 3, 30, 10, step=1)

    st.markdown("---")

    if st.button("Train Model Now", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Preparing data...")
        for pct in range(0, 40, 10):
            time.sleep(0.05)
            progress_bar.progress(pct, text="Preparing data...")

        progress_bar.progress(40, text="Splitting train/test sets...")
        time.sleep(0.1)

        progress_bar.progress(60, text="Training Random Forest model...")
        try:
            results = model_utils.train_model(
                df,
                test_size=test_size,
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
        except Exception as e:
            progress_bar.empty()
            st.error(f"Training failed: {e}")
            return

        progress_bar.progress(85, text="Saving trained model...")
        model_utils.save_model(results["model"], model_path)
        time.sleep(0.1)

        progress_bar.progress(100, text="Done!")
        time.sleep(0.2)
        progress_bar.empty()

        st.success("Model trained and saved successfully!")
        st.session_state["model_version"] = st.session_state.get("model_version", 0) + 1

        # Cache results in session_state so Performance page can reuse them
        st.session_state["last_training_results"] = {
            "metrics": results["metrics"],
            "confusion": results["confusion"],
            "report": results["report"],
            "feature_importance": results["feature_importance"],
        }
        st.markdown("---")
        st.subheader("Training Results")

        m = results["metrics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{m['precision']*100:.2f}%")
        col3.metric("Recall", f"{m['recall']*100:.2f}%")
        col4.metric("F1-Score", f"{m['f1']*100:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(viz.confusion_matrix_heatmap(results["confusion"]), use_container_width=True)
        with col2:
            st.plotly_chart(viz.feature_importance_bar(results["feature_importance"]), use_container_width=True)

        with st.expander("Full Classification Report"):
            st.text(results["report"])

    else:
        st.info("Configure the hyperparameters above and click **Train Model Now** to begin.")
