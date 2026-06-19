import streamlit as st
import pandas as pd
import joblib

# Load saved model and columns
model = joblib.load("rf_model.pkl")
training_columns = joblib.load("columns.pkl")

# Page settings
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Fake Job Posting Detector")
st.write("Enter job details to check if a job posting is Genuine or Fake.")

# User inputs (same as notebook prediction)
title = st.text_input("Job Title", "Data Analyst")
location = st.text_input("Location", "New York")
department = st.text_input("Department", "Analytics")

company_profile = st.text_area(
    "Company Profile",
    "Tech Company"
)

requirements = st.text_area(
    "Requirements",
    "Python, SQL, Excel"
)

benefits = st.text_area(
    "Benefits",
    "Health insurance, Paid leave"
)

has_company_logo = st.selectbox(
    "Has Company Logo",
    [0, 1]
)

employment_type = st.selectbox(
    "Employment Type",
    [
        "Full-time",
        "Part-time",
        "Contract",
        "Temporary",
        "Other"
    ]
)

required_experience = st.selectbox(
    "Required Experience",
    [
        "Internship",
        "Entry level",
        "Associate",
        "Mid-Senior level",
        "Director",
        "Executive",
        "1-3 years"
    ]
)

required_education = st.selectbox(
    "Required Education",
    [
        "High School",
        "Bachelor's Degree",
        "Master's Degree",
        "Doctorate"
    ]
)

industry = st.text_input("Industry", "IT")
function = st.text_input("Function", "Analytics")

# Predict Button
if st.button("Predict Job Authenticity"):

    # Create input dataframe
    new_job = {
        'title': title,
        'location': location,
        'department': department,
        'company_profile': company_profile,
        'requirements': requirements,
        'benefits': benefits,
        'has_company_logo': has_company_logo,
        'employment_type': employment_type,
        'required_experience': required_experience,
        'required_education': required_education,
        'industry': industry,
        'function': function
    }

    new_job_df = pd.DataFrame([new_job])

    # One-hot encoding
    new_job_encoded = pd.get_dummies(new_job_df)

    # Match training columns
    new_job_encoded = new_job_encoded.reindex(
        columns=training_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(new_job_encoded)
    prediction_proba = model.predict_proba(new_job_encoded)

    # Result
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ This Job Posting is likely FAKE")
    else:
        st.success("✅ This Job Posting seems GENUINE")

    st.write(
        f"Confidence Score: {max(prediction_proba[0])*100:.2f}%"
    )