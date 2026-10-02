import os
import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ==================================================
       Main App
       ================================================== */

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(124, 58, 237, 0.22), transparent 28%),
            radial-gradient(circle at 90% 20%, rgba(236, 72, 153, 0.16), transparent 25%),
            radial-gradient(circle at 50% 100%, rgba(59, 130, 246, 0.12), transparent 30%),
            #09090f;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* ==================================================
       Animated Background
       ================================================== */

    .stApp::before {
        content: "";
        position: fixed;
        inset: -30%;
        background:
            radial-gradient(circle, rgba(124, 58, 237, 0.10), transparent 35%),
            radial-gradient(circle, rgba(236, 72, 153, 0.08), transparent 35%);
        animation: aurora 12s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: -1;
    }

    @keyframes aurora {
        0%   { transform: translate(-3%, -2%) scale(1); }
        100% { transform: translate(3%, 2%) scale(1.08); }
    }

    /* ==================================================
       Main Container
       ================================================== */

    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 2rem;
    }

    /* ==================================================
       Header
       ================================================== */

    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #c084fc, #f472b6, #818cf8, #c084fc);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientMove 6s ease infinite;
    }

    @keyframes gradientMove {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .subtitle {
        text-align: center;
        color: #a1a1aa;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 2.2rem;
    }

    /* ==================================================
       Section Titles
       ================================================== */

    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #f8fafc;
        margin-top: 1.3rem;
        margin-bottom: 1rem;
    }

    /* ==================================================
       Input Labels
       ================================================== */

    .stNumberInput label {
        color: #d4d4d8 !important;
        font-weight: 500 !important;
    }

    /* ==================================================
       Input Fields
       ================================================== */

    .stNumberInput input {
        background-color: #ffffff !important;
        color: #111827 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: 1px solid #e5e7eb !important;
    }

    .stNumberInput input:focus {
        border: 2px solid #a855f7 !important;
        box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.15) !important;
    }

    /* ==================================================
       Prediction Button
       ================================================== */

    .stButton > button {
        width: 100%;
        margin-top: 1rem;
        padding: 0.85rem 1rem;
        border: none;
        border-radius: 12px;
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        background: linear-gradient(90deg, #7c3aed, #db2777, #7c3aed);
        background-size: 200% auto;
        transition: transform 0.25s ease, box-shadow 0.25s ease, background-position 0.5s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        background-position: right center;
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.35);
    }

    /* ==================================================
       Prediction Result
       ================================================== */

    .result-card {
        margin-top: 2rem;
        padding: 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.22), rgba(219, 39, 119, 0.18));
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25);
        text-align: center;
    }

    .result-label {
        color: #c4b5fd;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    .result-score {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.8rem;
        font-weight: 700;
        line-height: 1.1;
        background: linear-gradient(90deg, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .result-description {
        color: #d4d4d8;
        margin-top: 0.7rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .model-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.10);
        color: #c4b5fd;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* ==================================================
       Model Note
       ================================================== */

    .model-note {
        margin-top: 1rem;
        padding: 0.85rem 1.1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.07);
        color: #a1a1aa;
        font-size: 0.8rem;
        line-height: 1.5;
        text-align: center;
    }

    /* ==================================================
       Feature Cards
       ================================================== */

    .feature-card {
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.2s ease, background 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.08);
    }

    .feature-name {
        color: #f4f4f5;
        font-weight: 600;
        font-size: 0.93rem;
    }

    .feature-value {
        color: #a1a1aa;
        font-size: 0.84rem;
        margin-top: 0.25rem;
    }

    /* ==================================================
       Divider
       ================================================== */

    .divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin: 2rem 0;
    }

    /* ==================================================
       Footer
       ================================================== */

    .footer {
        text-align: center;
        color: #71717a;
        font-size: 0.8rem;
        margin-top: 2.5rem;
        line-height: 1.6;
    }

    /* ==================================================
       Hide Streamlit Branding
       ================================================== */

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ==================================================
       Scrollbar
       ================================================== */

    ::-webkit-scrollbar { width: 7px; }
    ::-webkit-scrollbar-track { background: #09090f; }
    ::-webkit-scrollbar-thumb {
        background: #52525b;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load Model Artifact
# --------------------------------------------------

artifact_path = os.path.join(
    os.path.dirname(__file__),
    "artifacts",
    "student_score_model.pkl"
)

artifact = joblib.load(artifact_path)

model = artifact["model"]
features = artifact["features"]


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎓 Student Score Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "Predict a student's exam score using academic "
    "and lifestyle factors."
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Student Information
# --------------------------------------------------

st.markdown(
    '<div class="section-title">Student Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    hours_studied = st.number_input(
        "Hours Studied",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=1.0
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

with col2:

    previous_scores = st.number_input(
        "Previous Scores",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )

    tutoring_sessions = st.number_input(
        "Tutoring Sessions",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=1.0
    )

    physical_activity = st.number_input(
        "Physical Activity (hours/week)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=1.0
    )


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

predict_button = st.button(
    "Predict Exam Score",
    type="primary"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if predict_button:

    # Create input DataFrame using the exact
    # feature order used during model training.
    input_data = pd.DataFrame(
        [[
            hours_studied,
            attendance,
            sleep_hours,
            previous_scores,
            tutoring_sessions,
            physical_activity
        ]],
        columns=features
    )

    # Generate prediction
    predicted_score = model.predict(input_data)[0]

    # Keep prediction within the logical exam-score range
    predicted_score = max(0, min(100, predicted_score))


    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                Predicted Exam Score
            </div>
            <div class="result-score">
                {predicted_score:.1f}
            </div>
            <div class="result-description">
                Estimated exam score based on the
                academic and lifestyle information provided.
            </div>
            <div class="model-badge">
                Multiple Linear Regression
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # Model Note
    # --------------------------------------------------

    st.markdown(
        """
        <div class="model-note">
            The prediction is generated from the same six
            numerical features used to train the model.
            It is an estimate and not a guaranteed result.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------
    # Features Used
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">Features Used</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Input values used by the prediction model."
    )

    # Keep the feature display in the exact same
    # order as the trained model.
    feature_values = {
        "Hours Studied": f"{hours_studied:.1f} hours",
        "Attendance": f"{attendance:.0f}%",
        "Sleep Hours": f"{sleep_hours:.1f} hours",
        "Previous Scores": f"{previous_scores:.1f}",
        "Tutoring Sessions": f"{tutoring_sessions:.0f}",
        "Physical Activity": f"{physical_activity:.1f} hours/week"
    }

    col1, col2 = st.columns(2)

    feature_items = list(feature_values.items())

    for i, (name, value) in enumerate(feature_items):

        with col1 if i % 2 == 0 else col2:

            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-name">
                        {name}
                    </div>
                    <div class="feature-value">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    '<div class="divider"></div>'
    '<div class="footer">'
    'Student Score Prediction using Multiple Linear Regression'
    '<br>'
    'Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)