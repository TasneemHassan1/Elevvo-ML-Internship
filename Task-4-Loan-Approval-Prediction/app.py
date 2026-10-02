import streamlit as st
import pandas as pd
import joblib
import os


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💳",
    layout="centered"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

    .stApp {
        background: linear-gradient(135deg, #0f172a, #111827, #1e1b4b);
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .section-title {
        color: #e2e8f0;
        font-size: 22px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(12px);
    }

    label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }

    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: white !important;
        color: #111827 !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 12px;
        font-size: 18px;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.35);
    }

    /* Prediction Box */

    .prediction-box {
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        margin-top: 25px;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
    }

    .prediction-box.approved {
        border: 1px solid rgba(34, 197, 94, 0.5);
        background: rgba(34, 197, 94, 0.12);
        box-shadow: 0 8px 25px rgba(34, 197, 94, 0.08);
    }

    .prediction-box.rejected {
        border: 1px solid rgba(239, 68, 68, 0.5);
        background: rgba(239, 68, 68, 0.12);
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.08);
    }

    .prediction-title {
        font-size: 17px;
        color: #cbd5e1 !important;
        margin-bottom: 5px;
    }

    .prediction-result {
        font-size: 32px;
        font-weight: 800;
        margin-top: 8px;
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    base_dir,
    "artifacts",
    "loan_approval_model.pkl"
)

artifact = joblib.load(model_path)

model = artifact["model"]
features = artifact["features"]

education_encoder = artifact["education_encoder"]
self_employed_encoder = artifact["self_employed_encoder"]


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">💳 Loan Approval Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict loan approval using a machine learning model'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Applicant Information
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Applicant Information</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:

        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=5,
            value=2,
            step=1
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        income_annum = st.number_input(
            "Annual Income",
            min_value=200000,
            max_value=9900000,
            value=5000000,
            step=100000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=300000,
            max_value=39500000,
            value=10000000,
            step=100000
        )

    with col2:

        loan_term = st.number_input(
            "Loan Term (Years)",
            min_value=2,
            max_value=20,
            value=10,
            step=2
        )

        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=700,
            step=1
        )

        residential_assets_value = st.number_input(
            "Residential Assets Value",
            min_value=0,
            max_value=29100000,
            value=5000000,
            step=100000
        )

        commercial_assets_value = st.number_input(
            "Commercial Assets Value",
            min_value=0,
            max_value=19400000,
            value=2000000,
            step=100000
        )

        luxury_assets_value = st.number_input(
            "Luxury Assets Value",
            min_value=300000,
            max_value=39200000,
            value=5000000,
            step=100000
        )

        bank_asset_value = st.number_input(
            "Bank Asset Value",
            min_value=0,
            max_value=14700000,
            value=2000000,
            step=100000
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Loan Approval"):

    education_encoded = education_encoder.transform(
        [education]
    )[0]

    self_employed_encoded = self_employed_encoder.transform(
        [self_employed]
    )[0]

    input_data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education_encoded,
        "self_employed": self_employed_encoded,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    input_data = input_data[features]

    prediction = model.predict(input_data)[0]

    if prediction == "Approved":

        st.markdown(
            """
            <div class="prediction-box approved">
                <div class="prediction-title">Prediction</div>
                <div class="prediction-result">
                    ✅ Loan Approved
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="prediction-box rejected">
                <div class="prediction-title">Prediction</div>
                <div class="prediction-result">
                    ❌ Loan Rejected
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )