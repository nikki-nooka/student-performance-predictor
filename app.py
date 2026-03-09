import streamlit as st
import pickle
import numpy as np

# Page setup
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# Load trained model
model = pickle.load(open("student_model.pkl", "rb"))

st.title("🎓 Student Performance Prediction System")
st.write("Enter student details below to predict performance.")

st.divider()

# -----------------------------
# Academic Scores
# -----------------------------
st.subheader("📘 Academic Scores")

col1, col2 = st.columns(2)

with col1:
    math_score = st.number_input("Math Score", 0, 100)
    science_score = st.number_input("Science Score", 0, 100)

with col2:
    english_score = st.number_input("English Score", 0, 100)
    assignment_score = st.number_input("Assignment Score", 0, 100)

st.divider()

# -----------------------------
# Student Activity
# -----------------------------
st.subheader("📊 Student Activity")

col3, col4 = st.columns(2)

with col3:
    attendance_percentage = st.slider("Attendance Percentage", 0, 100)

with col4:
    study_hours_per_week = st.slider("Study Hours per Week", 0, 40)

st.divider()

# -----------------------------
# Student Background
# -----------------------------
st.subheader("📈 Student Background")

col5, col6 = st.columns(2)

with col5:
    previous_gpa = st.number_input("Previous GPA", 0.0, 10.0)

with col6:
    participation = st.selectbox(
        "Participation Level",
        ["Low", "Medium", "High"]
    )

# Encode participation exactly like training
if participation == "Low":
    participation_level = 0
elif participation == "Medium":
    participation_level = 1
else:
    participation_level = 2

# Derived feature
avg_score = (math_score + science_score + english_score + assignment_score) / 4

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Student Performance", use_container_width=True):

    # IMPORTANT: Feature order must match training dataset
    features = np.array([[
        math_score,
        science_score,
        english_score,
        assignment_score,
        attendance_percentage,
        study_hours_per_week,
        previous_gpa,
        participation_level,
        avg_score
    ]])

    prediction = model.predict(features)

    # Show probabilities (helps debugging)
    proba = model.predict_proba(features)

    st.subheader("📢 Prediction Result")

    if prediction[0] == 0:
        st.error("⚠️ Predicted Performance: LOW")

    elif prediction[0] == 1:
        st.warning("⚡ Predicted Performance: MEDIUM")

    else:
        st.success("🏆 Predicted Performance: HIGH")

    st.write("Prediction Probabilities:", proba)