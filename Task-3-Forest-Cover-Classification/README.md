# Task 3 - Forest Cover Type Classification

## Project Overview

This project predicts the type of forest cover based on cartographic and environmental features using multi-class classification.

The project uses the **Covertype dataset from UCI**, which contains 581,012 samples, 54 input features, and 7 forest cover types.

## Dataset

The dataset includes features such as:

* Elevation
* Aspect and Slope
* Distances to hydrology, roads, and fire points
* Hillshade measurements
* Wilderness area indicators
* Soil type indicators

The target variable is `Cover_Type`, with values from 1 to 7.

## Data Preprocessing

The dataset was checked for:

* Missing values
* Duplicate rows
* Invalid target values
* Feature ranges

No missing values or duplicate rows were found, and all target values were valid.

The dataset already contains wilderness area and soil type information as binary indicator features, so no additional categorical encoding was required.

A stratified train/test split was used to keep a similar class distribution in both sets.

## Models

### Random Forest

Random Forest was used as the baseline classification model.

Results:

* Accuracy: **95.42%**
* Macro Precision: **94.57%**
* Macro Recall: **90.52%**
* Macro F1-Score: **92.38%**

### XGBoost

XGBoost was tested as an additional model.

Results:

* Accuracy: **80.93%**
* Macro Precision: **82.11%**
* Macro Recall: **71.66%**
* Macro F1-Score: **74.97%**

Random Forest performed better in this experiment.

### Hyperparameter Tuning

Random Forest hyperparameters were tuned using `RandomizedSearchCV` with macro F1-score as the evaluation metric.

The tuned model achieved **94.70% accuracy**, which was slightly lower than the baseline Random Forest.

Therefore, the baseline Random Forest was selected as the final model.

## Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Feature importance was also used to identify the most important features for the Random Forest model.

## Project Structure

```text
Task-3-Forest-Cover-Classification/
│
├── data/
│   └── covtype.data
│
├── artifacts/
│   └── forest_cover_model.pkl
│
├── forest_cover_classification.ipynb
├── README.md
└── requirements.txt
```

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Open the notebook:

```bash
jupyter notebook forest_cover_classification.ipynb
```

3. Run the notebook cells from top to bottom.

The final trained Random Forest model is saved as:

```text
artifacts/forest_cover_model.pkl
```

## Tools

Python, Pandas, Matplotlib, Scikit-learn, XGBoost, Joblib
