"""
model_utils.py
--------------
All machine-learning logic lives here: training, evaluation, saving,
loading, and running predictions (single + batch).

Keeping this logic separate from the Streamlit pages keeps the UI code
clean and makes the ML code independently testable/reusable.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# The exact feature order the model expects. Keeping this as a single
# source of truth avoids "duplicate column" / "wrong order" bugs.
FEATURE_COLUMNS = [
    "CGPA",
    "AptitudeScore",
    "CommunicationSkill",
    "Attendance",
    "Projects",
    "Internship",
    "Certifications",
    "CodingSkill",
]
TARGET_COLUMN = "Placement"


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
                 n_estimators: int = 300, max_depth: int = 10):
    """
    Train a RandomForestClassifier on the placement dataset.

    Returns
    -------
    dict with keys:
        model         -> trained sklearn model
        metrics       -> dict of accuracy/precision/recall/f1
        confusion     -> confusion matrix (numpy array)
        report        -> full classification report (string)
        feature_importance -> DataFrame of feature importances
        X_test, y_test, y_pred, y_proba -> for further analysis/plots
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"])

    feature_importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "metrics": metrics,
        "confusion": cm,
        "report": report,
        "feature_importance": feature_importance,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def save_model(model, model_path: Path):
    """Persist a trained model to disk using joblib."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def load_model(model_path: Path):
    """
    Load a trained model from disk.
    Returns None if the file doesn't exist (caller should handle this
    gracefully, e.g. by prompting the user to train a model first).
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def predict_single(model, input_dict: dict):
    """
    Predict placement for a single student.

    Parameters
    ----------
    model : trained sklearn model
    input_dict : dict
        Must contain all keys in FEATURE_COLUMNS.

    Returns
    -------
    (prediction, probability) -> (int 0/1, float 0-1)
    """
    row = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    return prediction, probability


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict placement for a batch of students (e.g. from an uploaded CSV).

    Returns a copy of the input DataFrame with two new columns:
    'Predicted_Placement' (Placed / Not Placed) and 'Placement_Probability'.
    """
    X = df[FEATURE_COLUMNS].copy()
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["Predicted_Placement"] = np.where(preds == 1, "Placed", "Not Placed")
    result["Placement_Probability"] = np.round(probs * 100, 2)
    return result


def validate_columns(df: pd.DataFrame):
    """
    Check that an uploaded DataFrame has all required feature columns.

    Returns
    -------
    (is_valid: bool, missing_columns: list)
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    return (len(missing) == 0, missing)
