# 🎓 Placement Prediction Streamlit App

A complete, ready-to-run **Machine Learning web application** built with
Streamlit that predicts whether a student is likely to be placed based
on academic performance, skills, and experience.

---

## 📌 Project Overview

This project demonstrates a full end-to-end ML workflow inside an
interactive multi-page dashboard:

- Synthetic dataset generation with realistic, correlated features
- Exploratory Data Analysis (EDA) with interactive Plotly charts
- Model training (Random Forest Classifier) with live progress feedback
- Single-student prediction via a simple form
- Batch prediction via CSV upload with downloadable results
- Full model performance evaluation (accuracy, precision, recall, F1,
  confusion matrix, feature importance)

> ⚠️ **Disclaimer:** The dataset is synthetically generated for
> demonstration purposes only. Do not use these predictions for real
> academic or hiring decisions.

---

## ✨ Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | Modern multi-page dashboard | Sidebar navigation across 8 pages |
| 2 | Home page | Project overview + KPI cards |
| 3 | Dataset Explorer | Filter and browse the dataset |
| 4 | EDA | Correlation heatmaps, histograms, scatter plots, box plots |
| 5 | Train Model | Configurable hyperparameters + live progress bar |
| 6 | Prediction | Single-student form-based prediction |
| 7 | Batch Prediction | CSV upload + bulk predictions |
| 8 | Model Performance | Metrics, confusion matrix, feature importance |
| 9 | About | Project & tech stack info |
| 10 | CSV Downloads | Download filtered data and prediction results |

---

## 🧰 Technologies Used

- **Python 3.13.7**
- **Streamlit** — web app framework
- **scikit-learn** — Random Forest Classifier
- **pandas / numpy** — data manipulation
- **Plotly** — interactive visualizations
- **joblib** — model persistence

---

## 📦 Installation

1. **Clone or download this project folder.**

2. **(Recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

From the project root directory, run:

```bash
streamlit run app.py
```

Then open the URL shown in your terminal (usually `http://localhost:8501`)
in your browser.

**First-time run behavior:**
- If `data/placement_data.csv` doesn't exist, it will be **auto-generated**
  (2000 synthetic student records).
- If `models/placement_model.pkl` doesn't exist, visit the **Train Model**
  page in the app to train and save one — it only takes a few seconds.

---

## 🖼️ Screenshots

> _(Add screenshots of your running app here)_

- **Home Page:** `screenshots/home.png`
- **EDA Page:** `screenshots/eda.png`
- **Train Model Page:** `screenshots/train_model.png`
- **Prediction Page:** `screenshots/prediction.png`
- **Batch Prediction Page:** `screenshots/batch_prediction.png`
- **Model Performance Page:** `screenshots/performance.png`

---

## 📁 Folder Structure

```
Placement-Prediction-Streamlit/
│
├── app.py                     # Main entry point & page router
├── requirements.txt            # Pinned dependencies
├── README.md                   # This file
│
├── data/
│   └── placement_data.csv      # Auto-generated synthetic dataset
│
├── models/
│   └── placement_model.pkl     # Saved trained model (created after training)
│
├── utils/
│   ├── data_generator.py       # Synthetic dataset generation logic
│   ├── model_utils.py          # Training, saving, loading, prediction logic
│   └── visualizations.py       # Reusable Plotly chart functions
│
├── pages/
│   ├── home.py                 # Home / overview page
│   ├── dataset.py               # Dataset explorer page
│   ├── eda.py                   # Exploratory data analysis page
│   ├── train_model.py           # Model training page
│   ├── prediction.py            # Single student prediction page
│   ├── batch_prediction.py      # Batch CSV prediction page
│   ├── performance.py           # Model performance/metrics page
│   └── about.py                  # About page
│
└── assets/
    └── style.css                # Custom CSS styling
```

---

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier (`scikit-learn`)
- **Target variable:** `Placement` (0 = Not Placed, 1 = Placed)
- **Input features:**
  - `CGPA` (0–10)
  - `AptitudeScore` (0–100)
  - `CommunicationSkill` (1–10)
  - `Attendance` (0–100)
  - `Projects` (0–10)
  - `Internship` (0/1)
  - `Certifications` (0–10)
  - `CodingSkill` (1–10)
- **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Explainability:** Feature importance chart included

---

## 📝 Notes for Developers

- All file paths use `pathlib.Path` for cross-platform compatibility.
- Data loading is cached with `st.cache_data`; the model is cached with
  `st.cache_resource` — both are cleared/refreshed automatically after
  retraining.
- The dataset and model are **auto-created if missing**, so the app
  never crashes on a fresh clone — just run `streamlit run app.py`.

---

Made with ❤️ using Streamlit.
