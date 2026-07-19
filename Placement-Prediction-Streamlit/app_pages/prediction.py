import streamlit as st
from utils import model_utils
from utils import visualizations as viz
def render(model):
    st.title("Placement Prediction (Single Student)")

    if model is None:
        st.warning(" No trained model found. Please go to the **Train Model** page first.")
        return

    st.markdown("Fill in the student's details below to predict their placement outcome.")
    st.markdown("---")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, step=0.1)
            aptitude = st.slider("Aptitude Score", 0, 100, 65)
            communication = st.slider("Communication Skill (1-10)", 1, 10, 6)
            attendance = st.slider("Attendance (%)", 0, 100, 80)

        with col2:
            projects = st.slider("Number of Projects", 0, 10, 3)
            internship = st.radio("Internship Completed?", options=["Yes", "No"], horizontal=True)
            certifications = st.slider("Number of Certifications", 0, 10, 2)
            coding_skill = st.slider("Coding Skill (1-10)", 1, 10, 6)

        submitted = st.form_submit_button("Predict Placement", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not (0 <= cgpa <= 10):
            errors.append("CGPA must be between 0 and 10.")
        if not (0 <= aptitude <= 100):
            errors.append("Aptitude Score must be between 0 and 100.")
        if not (0 <= attendance <= 100):
            errors.append("Attendance must be between 0 and 100.")

        if errors:
            for e in errors:
                st.error(f" {e}")
            return

        input_dict = {
            "CGPA": cgpa,
            "AptitudeScore": aptitude,
            "CommunicationSkill": communication,
            "Attendance": attendance,
            "Projects": projects,
            "Internship": 1 if internship == "Yes" else 0,
            "Certifications": certifications,
            "CodingSkill": coding_skill,
        }

        try:
            prediction, probability = model_utils.predict_single(model, input_dict)
        except Exception as e:
            st.error(f" Prediction failed: {e}")
            return

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2 = st.columns([1, 1])

        with col1:
            if prediction == 1:
                st.success(" **Result: PLACED**")
            else:
                st.error(" **Result: NOT PLACED**")

            st.metric("Placement Probability", f"{probability*100:.1f}%")

            # --- Recommendation message ------------------------------
            st.markdown("### Recommendation")
            if probability >= 0.75:
                st.info("Excellent profile! Keep up the strong performance and start applying to top companies.")
            elif probability >= 0.5:
                st.info("Good chances of placement. Consider improving certifications, coding practice, and mock interviews to boost your odds further.")
            elif probability >= 0.3:
                st.warning("Moderate risk. Focus on improving CGPA, aptitude score, and gaining internship experience.")
            else:
                st.warning("Low placement probability currently. Prioritize academic performance, projects, and communication skills training.")

        with col2:
            st.plotly_chart(viz.probability_gauge(probability), use_container_width=True)
