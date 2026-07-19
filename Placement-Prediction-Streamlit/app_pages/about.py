import streamlit as st
def render():
    st.title("About This Project")
    st.markdown(
        """
        ## Placement Prediction Streamlit App

        This project is an end-to-end **Machine Learning web application**
        that predicts whether a student is likely to be placed based on
        academic performance, skills, and experience.

        ### Purpose
        Built as a demonstration of a complete ML workflow — from data
        generation, through exploratory analysis and model training, to
        real-time and batch prediction — all wrapped in an interactive
        Streamlit dashboard.

        ### Technologies Used
        | Technology | Purpose |
        |---|---|
        | **Python 3.13** | Core programming language |
        | **Streamlit** | Interactive web app framework |
        | **scikit-learn** | Machine learning (Random Forest Classifier) |
        | **pandas / numpy** | Data manipulation & numerical computing |
        | **Plotly** | Interactive charts and visualizations |
        | **joblib** | Model persistence (saving/loading) |

        ### Model Details
        - **Algorithm:** Random Forest Classifier
        - **Target:** Binary placement outcome (Placed / Not Placed)
        - **Features:** CGPA, Aptitude Score, Communication Skill,
          Attendance, Projects, Internship, Certifications, Coding Skill

        ### Disclaimer
        The dataset used in this app is **synthetically generated** for
        demonstration purposes. Predictions are illustrative and should
        **not** be used for real academic or hiring decisions.
        ---
        
        """
    )
