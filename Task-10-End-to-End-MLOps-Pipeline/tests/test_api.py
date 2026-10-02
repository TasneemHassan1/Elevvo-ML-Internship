from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


VALID_INPUT = {
    "Type": "M",
    "Air temperature [K]": 300.1,
    "Process temperature [K]": 310.2,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 100
}


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=VALID_INPUT
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "failure" in data
    assert "failure_probability" in data
    assert "threshold" in data


def test_invalid_numeric_input():
    invalid_input = VALID_INPUT.copy()
    invalid_input["Torque [Nm]"] = "hello"

    response = client.post(
        "/predict",
        json=invalid_input
    )

    assert response.status_code == 400
    assert response.json()["error"] == "Invalid input data"