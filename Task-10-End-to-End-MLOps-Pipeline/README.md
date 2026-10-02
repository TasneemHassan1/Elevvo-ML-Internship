# Task 10 - End-to-End MLOps Pipeline

An end-to-end deployment pipeline for an Industrial Predictive Maintenance machine learning model.

The project takes the trained model artifact from Task 9 and turns it into a deployable API service using FastAPI and Docker. It also includes input validation, Postman API testing, a Streamlit frontend, automated tests, and GitHub Actions CI.

## Project Structure

```text
Task-10-End-to-End-MLOps-Pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── artifacts/
│   └── predictive_maintenance_model.pkl
├── frontend/
│   └── streamlit_app.py
├── postman/
│   └── Predictive Maintenance API.postman_collection.json
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pytest.ini
├── README.md
└── requirements.txt
```

## Model

The API uses the saved model artifact from Task 9:

* Model: Cost-sensitive Random Forest
* Decision threshold: 0.55
* Artifact format: `.pkl`

The service recreates the feature engineering used during model development, including:

* Temperature Difference
* Torque-Speed Interaction
* Tool Wear-Torque Interaction

## FastAPI Service

The application provides two main endpoints.

### Health Check

```text
GET /health
```

This endpoint verifies that the service is running and that the model artifact was loaded successfully.

### Prediction

```text
POST /predict
```

The endpoint receives machine information and returns:

* Prediction
* Failure status
* Failure probability
* Decision threshold

Example request:

```json
{
  "Type": "M",
  "Air temperature [K]": 300.1,
  "Process temperature [K]": 310.2,
  "Rotational speed [rpm]": 1500,
  "Torque [Nm]": 40.0,
  "Tool wear [min]": 100
}
```

## Input Validation

Pydantic is used to validate incoming request data.

For example, numerical fields such as `Torque [Nm]` must contain valid numerical values.

Invalid input is rejected with:

```text
HTTP 400 Bad Request
```

Example invalid value:

```json
{
  "Torque [Nm]": "hello"
}
```

The API returns an error response containing the validation details.

## Running the API Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI service:

```bash
uvicorn app.main:app --reload
```

The API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build the Docker image:

```bash
docker build -t predictive-maintenance-api .
```

Run the container:

```bash
docker run -d \
  --name predictive-maintenance-container \
  -p 8000:8000 \
  predictive-maintenance-api
```

The API can then be accessed through:

```text
http://127.0.0.1:8000
```

## Postman Testing

A Postman collection is included in:

```text
postman/Predictive Maintenance API.postman_collection.json
```

The collection contains:

* Health Check
* Valid Prediction Request
* Invalid Prediction Request

The invalid request verifies that incorrect numerical input is rejected with HTTP 400.

## Streamlit Frontend

A simple Streamlit frontend is included in:

```text
frontend/streamlit_app.py
```

It supports uploading:

* CSV files
* JSON files

The uploaded machine data is sent to the FastAPI prediction endpoint, and the frontend displays:

* Prediction result
* Failure probability
* Model threshold

Run the frontend with:

```bash
streamlit run frontend/streamlit_app.py
```

## Automated Testing

The project includes automated API tests using Pytest.

The tests cover:

* Health check
* Valid prediction request
* Invalid numerical input

Run the tests with:

```bash
pytest
```

Current test result:

```text
3 passed
```

## GitHub Actions CI

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The workflow automatically runs the test suite on every push and pull request.

The CI process:

```text
Checkout repository
        ↓
Set up Python 3.12
        ↓
Install dependencies
        ↓
Run Pytest
        ↓
Pass / Fail
```

## Covered Topics

* Model Serving
* Containerization
* API Development
* Input Schema Validation
* API Testing
* Frontend Integration
* Automated Testing
* CI/CD
