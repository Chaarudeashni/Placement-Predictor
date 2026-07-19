"""
data_generator.py
------------------
Generates a realistic synthetic dataset for student placement prediction.

Why synthetic data?
Many students/companies don't have a ready-made placement dataset. This
module builds one with realistic value ranges AND meaningful correlations
(e.g., higher CGPA + more projects + internships => higher placement
probability) so that a machine learning model can actually learn patterns
from it, instead of random noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Fixed seed => same dataset every time we regenerate it (reproducibility)
RANDOM_SEED = 42


def generate_synthetic_data(n_rows: int = 2000) -> pd.DataFrame:
    """
    Generate a synthetic placement dataset.

    Parameters
    ----------
    n_rows : int
        Number of student records to generate (default 2000).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        CGPA, AptitudeScore, CommunicationSkill, Attendance, Projects,
        Internship, Certifications, CodingSkill, Placement
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Feature generation with realistic distributions -------------------
    # CGPA: most students cluster around 7, scale 0-10
    cgpa = rng.normal(loc=7.0, scale=1.1, size=n_rows)
    cgpa = np.clip(cgpa, 4.0, 10.0)

    # Aptitude test score out of 100
    aptitude = rng.normal(loc=62, scale=15, size=n_rows)
    aptitude = np.clip(aptitude, 0, 100)

    # Communication skill rated 1-10 by placement cell / mock interviews
    communication = rng.integers(low=1, high=11, size=n_rows)

    # Attendance percentage
    attendance = rng.normal(loc=78, scale=12, size=n_rows)
    attendance = np.clip(attendance, 40, 100)

    # Number of academic/mini projects completed (0-10)
    projects = rng.integers(low=0, high=11, size=n_rows)

    # Internship: 0 = No, 1 = Yes (about 40% of students have one)
    internship = rng.binomial(n=1, p=0.4, size=n_rows)

    # Number of certifications completed (0-10)
    certifications = rng.integers(low=0, high=11, size=n_rows)

    # Coding skill self/assessed rating (1-10)
    coding_skill = rng.integers(low=1, high=11, size=n_rows)

    # --- Build a weighted "placement readiness score" -----------------------
    # Each feature is normalized to a 0-1 range and combined with weights
    # that reflect real-world hiring priorities (technical + soft skills).
    score = (
        0.25 * (cgpa / 10.0)
        + 0.20 * (aptitude / 100.0)
        + 0.13 * (communication / 10.0)
        + 0.10 * (attendance / 100.0)
        + 0.10 * (projects / 10.0)
        + 0.07 * internship
        + 0.07 * (certifications / 10.0)
        + 0.08 * (coding_skill / 10.0)
    )

    # Add a small amount of random noise so the relationship isn't
    # perfectly deterministic (real life always has some randomness /
    # unmeasured factors), while still keeping a strong, learnable signal.
    noise = rng.normal(loc=0, scale=0.02, size=n_rows)
    score_noisy = score + noise

    # Convert the continuous readiness score into a placement PROBABILITY
    # using a logistic (sigmoid) function centered at ~0.55 readiness.
    # A steeper slope (25) makes the decision boundary sharper, giving the
    # model clearer, more learnable patterns (mirrors how, in reality,
    # strong all-round candidates are placed very consistently).
    probability = 1 / (1 + np.exp(-25 * (score_noisy - 0.55)))

    # Finally sample the binary Placement outcome from that probability
    placement = rng.binomial(n=1, p=probability)

    df = pd.DataFrame({
        "CGPA": np.round(cgpa, 2),
        "AptitudeScore": np.round(aptitude, 1),
        "CommunicationSkill": communication,
        "Attendance": np.round(attendance, 1),
        "Projects": projects,
        "Internship": internship,
        "Certifications": certifications,
        "CodingSkill": coding_skill,
        "Placement": placement,
    })

    return df


def load_or_generate_data(csv_path: Path, n_rows: int = 2000) -> pd.DataFrame:
    """
    Load the dataset from disk if it already exists; otherwise generate
    a fresh synthetic dataset and save it to `csv_path`.

    This gives the app a "self-healing" behaviour: even if someone deletes
    the data folder, the app will recreate it automatically on next run.
    """
    csv_path = Path(csv_path)

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Basic sanity check: make sure expected columns are present
            expected_cols = {
                "CGPA", "AptitudeScore", "CommunicationSkill", "Attendance",
                "Projects", "Internship", "Certifications", "CodingSkill",
                "Placement",
            }
            if expected_cols.issubset(set(df.columns)):
                return df
        except Exception:
            # If the file is corrupted/unreadable, fall through and regenerate
            pass

    # File missing or invalid -> generate a new dataset
    df = generate_synthetic_data(n_rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    # Allows running this file directly to (re)generate the dataset:
    # python utils/data_generator.py
    out_path = Path(__file__).resolve().parent.parent / "data" / "placement_data.csv"
    data = generate_synthetic_data(2000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"Generated {len(data)} rows -> {out_path}")
