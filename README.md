# Elevvo ML Internship

A collection of machine learning projects completed during the **Elevvo ML Internship**, covering regression, clustering, classification, recommendation systems, audio classification, time series forecasting, predictive maintenance, and end-to-end MLOps.

Each task has its own README with detailed explanations, methodology, results, and implementation details.

## Projects

| Task                                                                                     | Project                           | Level          | Bonuses                                                                               |
| ---------------------------------------------------------------------------------------- | --------------------------------- | -------------- | ------------------------------------------------------------------------------------- |
| [Task 1 — Student Score Prediction](./Task-1-Student-Score-Prediction)                   | Student Score Prediction          | Level 1        | Polynomial Regression, Feature Combinations, Streamlit Interface                      |
| [Task 2 — Customer Segmentation](./Task-2-Customer-Segmentation)                         | Customer Segmentation             | Level 1        | DBSCAN, Average Spending per Cluster, Streamlit Interface                             |
| [Task 3 — Forest Cover Type Classification](./Task-3-Forest-Cover-Classification)        | Forest Cover Type Classification  | Level 2        | Random Forest vs XGBoost, Hyperparameter Tuning                                       |
| [Task 4 — Loan Approval Prediction](./Task-4-Loan-Approval-Prediction)                   | Loan Approval Prediction          | Level 2        | SMOTE, Logistic Regression vs Decision Tree, Streamlit Interface                      |
| [Task 5 — Movie Recommendation System](./Task-5-Movie-Recommendation)                    | Movie Recommendation System       | Level 2        | Item-Based Collaborative Filtering, SVD, Streamlit Interface                          |
| [Task 6 — Music Genre Classification](./Task-6-Music-Genre-Classification)               | Music Genre Classification        | Level 3        | —                                                                                     |
| [Task 7 — Sales Forecasting](./Task-7-Sales-Forecasting)                                 | Sales Forecasting                 | Level 3        | Rolling Averages, Seasonal Decomposition, XGBoost/LightGBM with Time-Aware Validation |
| [Task 9 — Industrial Predictive Maintenance](./Task-9-Industrial-Predictive-Maintenance) | Industrial Predictive Maintenance | Industry Level | Time-to-Failure / RUL, Lead-Indicator Correlation Analysis                            |
| [Task 10 — End-to-End MLOps Pipeline](./Task-10-End-to-End-MLOps-Pipeline)               | End-to-End MLOps Pipeline         | Industry Level | Streamlit Frontend, GitHub Actions CI                                                 |

## Machine Learning & MLOps Highlights

* **Regression:** Linear and Polynomial Regression
* **Clustering:** K-Means and DBSCAN
* **Classification:** Binary and Multi-Class Classification
* **Recommendation Systems:** Similarity-Based and Collaborative Filtering, SVD
* **Audio ML:** Music genre classification using extracted audio features
* **Time Series:** Lag features, rolling averages, seasonal decomposition, and time-aware validation
* **Predictive Maintenance:** Cost-sensitive modeling and threshold optimization with a focus on reducing false positives
* **Model Comparison:** Random Forest, XGBoost, Logistic Regression, Decision Trees, and other models
* **Hyperparameter Tuning:** Applied to multiple machine learning tasks
* **Interactive Applications:** Streamlit interfaces for Tasks 1, 2, 4, and 5
* **Model Serving:** FastAPI with Pydantic validation
* **Deployment:** Docker containerization
* **Testing & CI:** Pytest, Postman, and GitHub Actions

## Tech Stack

**Programming & Data**

* Python
* Pandas
* NumPy
* SQL

**Machine Learning**

* Scikit-learn
* XGBoost
* K-Means
* DBSCAN
* Regression & Classification Models
* Recommendation Systems

**Visualization & Applications**

* Matplotlib
* Seaborn
* Streamlit

**MLOps & Deployment**

* FastAPI
* Docker
* Pydantic
* Pytest
* Postman
* GitHub Actions

**Audio Processing**

* Librosa

## Repository Structure

```text
Elevvo-ML-Internship/
│
├── Task-1-Student-Score-Prediction/
├── Task-2-Customer-Segmentation/
├── Task-3-Forest-Cover-Classification/
├── Task-4-Loan-Approval-Prediction/
├── Task-5-Movie-Recommendation/
├── Task-6-Music-Genre-Classification/
├── Task-7-Sales-Forecasting/
├── Task-9-Industrial-Predictive-Maintenance/
├── Task-10-End-to-End-MLOps-Pipeline/
│
├── .gitattributes
└── .gitignore
```

Each task folder contains its own README with the full project details.
