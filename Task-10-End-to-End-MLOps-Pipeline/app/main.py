from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACT_PATH = (
    BASE_DIR
    / "artifacts"
    / "predictive_maintenance_model.pkl"
)


# --------------------------------------------------
# Load saved model artifact
# --------------------------------------------------

artifact = joblib.load(ARTIFACT_PATH)

model = artifact["model"]
threshold = artifact["threshold"]

print("Model artifact loaded successfully.")
print("Model type:", artifact["model_type"])
print("Threshold:", threshold)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description=(
        "FastAPI service for predicting machine failure "
        "using the saved Task 9 predictive maintenance model."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Pydantic validation error handler
# --------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input data",
            "details": exc.errors()
        }
    )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": artifact["model_type"],
        "threshold": threshold
    }


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def engineer_features(data: dict) -> pd.DataFrame:

    df = pd.DataFrame([{
        "Type": data["Type"],

        "Air temperature [K]":
            data["Air temperature [K]"],

        "Process temperature [K]":
            data["Process temperature [K]"],

        "Rotational speed [rpm]":
            data["Rotational speed [rpm]"],

        "Torque [Nm]":
            data["Torque [Nm]"],

        "Tool wear [min]":
            data["Tool wear [min]"]
    }])

    # Same feature engineering used in Task 9

    df["Temperature Difference [K]"] = (
        df["Process temperature [K]"]
        - df["Air temperature [K]"]
    )

    df["Torque-Speed Interaction"] = (
        df["Torque [Nm]"]
        * df["Rotational speed [rpm]"]
    )

    df["Tool Wear-Torque Interaction"] = (
        df["Tool wear [min]"]
        * df["Torque [Nm]"]
    )

    return df


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

from .schemas import MachineInput


@app.post("/predict")
def predict_machine_failure(input_data: MachineInput):

    # Convert validated Pydantic data to dictionary
    data = input_data.model_dump(
        by_alias=True
    )

    # Create the same engineered features
    features = engineer_features(data)

    # Predict probability of machine failure
    failure_probability = model.predict_proba(
        features
    )[:, 1][0]

    # Apply the threshold selected in Task 9
    prediction = int(
        failure_probability >= threshold
    )

    return {
        "prediction": prediction,
        "failure": bool(prediction),
        "failure_probability": round(
            float(failure_probability),
            4
        ),
        "threshold": threshold
    }