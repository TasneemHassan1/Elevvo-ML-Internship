import json

import pandas as pd
import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)


st.title("⚙️ Industrial Predictive Maintenance")
st.write(
    "Upload a CSV or JSON file containing machine data "
    "to predict whether the machine is likely to fail."
)


uploaded_file = st.file_uploader(
    "Upload CSV or JSON",
    type=["csv", "json"]
)


if uploaded_file is not None:

    try:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            data = json.load(uploaded_file)

            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)

        st.subheader("Uploaded Data")
        st.dataframe(df)

        if len(df) == 0:
            st.error("The uploaded file is empty.")

        else:

            row = df.iloc[0].to_dict()

            st.info(
                "The first row will be sent to the prediction API."
            )

            if st.button("Predict"):

                response = requests.post(
                    API_URL,
                    json=row,
                    timeout=10
                )

                if response.status_code == 200:

                    result = response.json()

                    st.subheader("Prediction Result")

                    if result["failure"]:
                        st.error("⚠️ Machine Failure Predicted")
                    else:
                        st.success("✅ No Machine Failure Predicted")

                    st.metric(
                        "Failure Probability",
                        f"{result['failure_probability'] * 100:.2f}%"
                    )

                    st.write(
                        "Model threshold:",
                        result["threshold"]
                    )

                else:

                    st.error(
                        f"API Error ({response.status_code})"
                    )

                    try:
                        st.json(response.json())
                    except Exception:
                        st.write(response.text)

    except Exception as e:

        st.error(
            f"Could not process the uploaded file: {e}"
        )