import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# Custom Dark Styling with Animations
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', 'Space Grotesk', sans-serif;
    }

    /* ===== App background with animated aurora ===== */
    .stApp {
        background: radial-gradient(circle at 20% 10%, #1a1f3a 0%, #0a0e27 40%, #05070f 100%);
        background-attachment: fixed;
        color: #e6e9f5;
        overflow-x: hidden;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(124, 58, 237, 0.15) 0%, transparent 40%),
                    radial-gradient(circle, rgba(236, 72, 153, 0.12) 0%, transparent 45%),
                    radial-gradient(circle, rgba(34, 211, 238, 0.10) 0%, transparent 40%);
        animation: auroraMove 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes auroraMove {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33%      { transform: translate(-5%, 3%) rotate(3deg); }
        66%      { transform: translate(4%, -3%) rotate(-3deg); }
    }

    .main {
        padding-top: 2rem;
        position: relative;
        z-index: 1;
    }

    /* ===== Main title ===== */
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(120deg, #a78bfa, #ec4899, #22d3ee, #a78bfa);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 6s ease infinite, fadeInDown 1s ease;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50%      { background-position: 100% 50%; }
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ===== Subtitle ===== */
    .subtitle {
        text-align: center;
        color: #8b92b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        animation: fadeIn 1.4s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }

    /* ===== Section titles ===== */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e6e9f5;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-left: 0.9rem;
        border-left: 4px solid #a78bfa;
        animation: slideUp 0.7s ease both;
    }

    /* ===== Prediction result card ===== */
    .result-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(236, 72, 153, 0.12));
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(167, 139, 250, 0.35);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 10px 40px rgba(124, 58, 237, 0.25),
                    inset 0 1px 0 rgba(255, 255, 255, 0.06);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        animation: slideUp 0.8s ease both, pulseGlow 3s ease-in-out infinite;
        transition: transform 0.35s ease, box-shadow 0.35s ease;
    }

    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 50px rgba(236, 72, 153, 0.35),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 10px 40px rgba(124, 58, 237, 0.25); }
        50%      { box-shadow: 0 10px 50px rgba(236, 72, 153, 0.4); }
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(25px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .result-label {
        color: #a5acd0;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
        background: linear-gradient(120deg, #a78bfa, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .result-description {
        color: #c7cbe0;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* ===== Metric cards ===== */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        height: 100%;
        backdrop-filter: blur(10px);
        transition: transform 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease;
        animation: slideUp 0.9s ease both;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(167, 139, 250, 0.45);
        box-shadow: 0 10px 30px rgba(124, 58, 237, 0.25);
    }

    .metric-label {
        color: #8b92b8;
        font-size: 0.8rem;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #a78bfa;
        font-size: 1.25rem;
        font-weight: 800;
    }

    /* ===== Button ===== */
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 14px;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: white !important;
        background: linear-gradient(120deg, #7c3aed, #ec4899);
        background-size: 200% 200%;
        border: none;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.35);
        animation: gradientShift 4s ease infinite;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 12px 32px rgba(236, 72, 153, 0.5);
        color: white !important;
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.99);
    }

    .stButton > button::after {
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
        transition: left 0.6s ease;
    }
    .stButton > button:hover::after {
        left: 100%;
    }

    /* ===== Inputs (WHITE background, DARK numbers) ===== */
    .stNumberInput input,
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid rgba(167, 139, 250, 0.4) !important;
        border-radius: 12px !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
    }

    /* Make typed numbers dark and bold */
    .stNumberInput input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #7c3aed !important;
    }

    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.25) !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }

    /* +/- buttons */
    .stNumberInput button {
        background-color: #f3f4f6 !important;
        color: #111827 !important;
        border: 1px solid rgba(167, 139, 250, 0.35) !important;
        font-weight: 800 !important;
    }

    .stNumberInput button:hover {
        background-color: #a78bfa !important;
        color: #ffffff !important;
    }

    /* Labels */
    label, .stMarkdown p, .stCaption {
        color: #c7cbe0 !important;
    }

    label {
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }

    /* ===== Alerts (info / success / warning) — WHITE background ===== */
    .stAlert {
        background: #ffffff !important;
        border-radius: 14px;
        border: 1px solid rgba(167, 139, 250, 0.4);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.15);
        animation: slideUp 0.5s ease both;
    }

    .stAlert p,
    .stAlert div,
    .stAlert span {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        font-weight: 700 !important;
    }

    .stAlert strong {
        color: #7c3aed !important;
        -webkit-text-fill-color: #7c3aed !important;
        font-weight: 900 !important;
    }

    /* ===== Divider ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.4), transparent);
        margin: 2rem 0;
    }

    /* ===== Footer ===== */
    .footer {
        text-align: center;
        color: #5a6088;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-bottom: 1rem;
        letter-spacing: 0.5px;
    }

    /* ===== Hide Streamlit branding ===== */
    #MainMenu, footer, header { visibility: hidden; }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0e27; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7c3aed, #ec4899);
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Load Saved Model and Data
# ============================================================

artifact_path = os.path.join(
    os.path.dirname(__file__),
    "artifacts",
    "kmeans_customer_segmentation.pkl"
)

artifact = joblib.load(artifact_path)

model = artifact["model"]
scaler = artifact["scaler"]
cluster_summary = artifact["cluster_summary"]
cluster_counts = artifact["cluster_counts"]
cluster_names = artifact["cluster_names"]
cluster_descriptions = artifact["cluster_descriptions"]


data_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "Mall_Customers.csv"
)

df = pd.read_csv(data_path)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">🛍️ Customer Segmentation</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover the customer segment based on annual income and spending behavior.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# Customer Information
# ============================================================

st.markdown(
    '<div class="section-title">Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input(
        "Annual Income (k$)",
        min_value=0.0,
        max_value=200.0,
        value=60.0,
        step=1.0
    )

with col2:
    spending_score = st.number_input(
        "Spending Score (1-100)",
        min_value=1.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )


# ============================================================
# Prediction Button
# ============================================================

predict_button = st.button(
    "Predict Customer Segment",
    type="primary"
)


# ============================================================
# Prediction
# ============================================================

if predict_button:

    customer_data = [[
        annual_income,
        spending_score
    ]]

    customer_scaled = scaler.transform(customer_data)

    predicted_cluster = model.predict(customer_scaled)[0]

    predicted_name = cluster_names[predicted_cluster]
    predicted_description = cluster_descriptions[predicted_cluster]

    average_income = cluster_summary.loc[
        predicted_cluster,
        "Annual Income (k$)"
    ]

    average_spending = cluster_summary.loc[
        predicted_cluster,
        "Spending Score (1-100)"
    ]

    customer_count = cluster_counts.loc[
        predicted_cluster
    ]


    # ========================================================
    # Prediction Result
    # ========================================================

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted Customer Segment</div>
            <div class="result-title">{predicted_name}</div>
            <div class="result-description">
                {predicted_description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # Segment Metrics
    # ========================================================

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Cluster ID</div>
                <div class="metric-value">{predicted_cluster}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Income</div>
                <div class="metric-value">{average_income:.2f} k$</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Spending</div>
                <div class="metric-value">{average_spending:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("")


    # ========================================================
    # Segment Size
    # ========================================================

    st.info(
        f"This segment contains **{customer_count} customers** "
        f"in the original dataset."
    )


    # ========================================================
    # Customer Position
    # ========================================================

    st.markdown(
        '<div class="section-title">Customer Position</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "The X marker shows the position of the new customer "
        "within the existing customer segments."
    )


    all_customers = df[
        ["Annual Income (k$)", "Spending Score (1-100)"]
    ]

    all_customers_scaled = scaler.transform(
        all_customers
    )

    all_customer_clusters = model.predict(
        all_customers_scaled
    )


    # ========================================================
    # Create visualization with LIGHT theme
    # ========================================================

    fig, ax = plt.subplots(figsize=(8, 5))

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=all_customer_clusters,
        cmap="viridis",
        alpha=0.75,
        s=45,
        edgecolors="white",
        linewidths=0.5
    )

    ax.scatter(
        annual_income,
        spending_score,
        marker="X",
        s=280,
        c="#dc2626",
        edgecolors="black",
        linewidths=1.5,
        label="New Customer"
    )

    ax.set_xlabel(
        "Annual Income (k$)",
        color="#1f2937",
        fontweight="600"
    )
    ax.set_ylabel(
        "Spending Score (1-100)",
        color="#1f2937",
        fontweight="600"
    )
    ax.set_title(
        "Customer Position within Segments",
        color="#111827",
        fontweight="bold",
        fontsize=13
    )

    ax.tick_params(colors="#1f2937")

    for spine in ax.spines.values():
        spine.set_color("#d1d5db")

    legend = ax.legend(
        facecolor="#ffffff",
        edgecolor="#d1d5db",
        loc="upper right"
    )
    for text in legend.get_texts():
        text.set_color("#111827")

    ax.grid(alpha=0.2, color="#9ca3af")

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# Footer
# ============================================================

st.markdown(
    """
    <div class="footer">
        Customer Segmentation using K-Means Clustering
        <br>
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)