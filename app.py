from pathlib import Path
import streamlit as st
from utils.data_generator import load_or_generate_data
from utils import model_utils
from app_pages import (
    home,
    dataset,
    eda,
    train_model,
    prediction,
    batch_prediction,
    performance,
    about,
)
st.set_page_config(
    page_title="Placement Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "placement_data.csv"
MODEL_PATH = BASE_DIR / "models" / "placement_model.pkl"
CSS_PATH = BASE_DIR / "assets" / "style.css"
def load_css(css_path: Path):
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css(CSS_PATH)
@st.cache_data(show_spinner="Loading dataset...")
def get_dataset(path: Path):
    return load_or_generate_data(path, n_rows=2000)
@st.cache_resource(show_spinner="Loading trained model...")
def get_model(path: Path, _cache_buster: int = 0):
    """
    _cache_buster lets us force a cache refresh (e.g. right after
    training a new model) by passing a different integer value.
    """
    return model_utils.load_model(path)
if "model_version" not in st.session_state:
    st.session_state["model_version"] = 0

df = get_dataset(DATA_PATH)
model = get_model(MODEL_PATH, st.session_state["model_version"])
st.sidebar.title("Placement Predictor")
st.sidebar.markdown("---")
PAGES = {
    "Home": "home",
    "EDA": "eda",
    "Prediction": "prediction",
    "Batch Prediction": "batch_prediction",
    "Model Performance": "performance",
    "About": "about",
}
selected_label = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
selected_page = PAGES[selected_label]

if selected_page == "home":
    home.render(df, model_exists=(model is not None))

elif selected_page == "dataset":
    dataset.render(df)

elif selected_page == "eda":
    eda.render(df)

elif selected_page == "train_model":
    train_model.render(df, MODEL_PATH)
    if MODEL_PATH.exists():
        pass

elif selected_page == "prediction":
    prediction.render(model)

elif selected_page == "batch_prediction":
    batch_prediction.render(model)

elif selected_page == "performance":
    performance.render(df, model)

elif selected_page == "about":
    about.render()
