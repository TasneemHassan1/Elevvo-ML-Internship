# Student Score Prediction

## Project Overview

This project is part of the **Elevvo Machine Learning Internship Program**.

The goal of this project is to predict students' exam scores using student performance factors.

The project starts with a simple **Linear Regression** model using `Hours_Studied` as the main feature. Then, additional numerical features are added using **Multiple Linear Regression**.

Polynomial Regression is also tested to compare whether increasing model complexity improves the predictions.

## Dataset

The project uses the **Student Performance Factors** dataset.

The dataset contains information about students, including:

* Hours Studied
* Attendance
* Sleep Hours
* Previous Scores
* Tutoring Sessions
* Physical Activity
* Parental Involvement
* Access to Resources
* Motivation Level
* and other student-related factors

The target variable is:

`Exam_Score`

The dataset contains **6,607 student records and 20 columns**.

## Project Workflow

The project follows these main steps:

1. Load the dataset
2. Inspect the data
3. Check duplicates and missing values
4. Clean missing values
5. Check invalid values
6. Detect and inspect outliers
7. Perform Exploratory Data Analysis (EDA)
8. Build a baseline Linear Regression model
9. Build a Multiple Linear Regression model
10. Test Polynomial Regression
11. Compare model performance
12. Draw a final conclusion
13. Save the final model for prediction

## Data Cleaning

The dataset was checked for:

* Missing values
* Duplicate rows
* Invalid values
* Statistical outliers

Missing values in categorical columns were filled using the **mode**.

One invalid `Exam_Score` value of `101` was found. Since exam scores cannot exceed 100, it was corrected to `100`.

Statistical outliers were inspected rather than automatically removed. Values that were unusual but still realistic were kept in the dataset.

## Exploratory Data Analysis

Several visualizations were used to understand the data:

* Hours Studied vs Exam Score
* Exam Score distribution
* Study Hours distribution
* Average Exam Score by Study Hours

The analysis showed a **moderate positive relationship** between study hours and exam scores.

## Models

### 1. Linear Regression

The baseline model uses only:

`Hours_Studied`

Performance:

| Metric | Result |
| ------ | -----: |
| MAE    |   2.45 |
| MSE    |  10.86 |
| RMSE   |   3.29 |
| R²     |   0.23 |

The model shows that study hours alone provide useful information, but they are not enough to explain most of the variation in exam scores.

### 2. Multiple Linear Regression

The second model uses six numerical features:

* Hours Studied
* Attendance
* Sleep Hours
* Previous Scores
* Tutoring Sessions
* Physical Activity

Performance:

| Metric | Result |
| ------ | -----: |
| MAE    |   1.27 |
| MSE    |   5.07 |
| RMSE   |   2.25 |
| R²     |   0.64 |

The Multiple Linear Regression model performed significantly better than the baseline model.

### 3. Polynomial Regression

Polynomial Regression was tested using:

* Degree 2
* Degree 3

The polynomial models produced only a very small improvement compared with the baseline Linear Regression model.

## Model Comparison

| Model                          |      MAE |      MSE |     RMSE |       R² |
| ------------------------------ | -------: | -------: | -------: | -------: |
| Linear Regression              |     2.45 |    10.86 |     3.29 |     0.23 |
| Polynomial Degree 2            |     2.44 |    10.85 |     3.29 |     0.23 |
| Polynomial Degree 3            |     2.44 |    10.84 |     3.29 |     0.23 |
| **Multiple Linear Regression** | **1.27** | **5.07** | **2.25** | **0.64** |

## Conclusion

The **Multiple Linear Regression** model achieved the best performance.

The results show that using several relevant student-related features provides much better predictions than using study hours alone.

Polynomial Regression did not provide a meaningful improvement, which suggests that adding useful features was more effective than increasing the complexity of the relationship between study hours and exam scores.

The final Multiple Linear Regression model was saved as an artifact and used in the Streamlit application for making predictions on new student inputs.

## Streamlit Application

A Streamlit web application was created to allow users to enter the six numerical features and receive a predicted exam score.

The application uses the saved **Multiple Linear Regression** model and displays:

* Predicted exam score
* Model used
* Input features used for prediction

## Tools and Libraries

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook
* Streamlit

## Project Files

```text
Task-1-Student-Score-Prediction/

│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── artifacts/
│   └── student_score_model.pkl
│
├── README.md
├── requirements.txt
├── student_score_prediction.ipynb
└── app.py
```

## How to Run

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

Open the notebook:

```bash
jupyter notebook student_score_prediction.ipynb
```

Run the notebook cells from top to bottom to reproduce the data cleaning, analysis, model training, evaluation, and model artifact creation.

### 3. Run the Streamlit Application

From the project root directory, run:

```bash
streamlit run Task-1-Student-Score-Prediction/app.py
```

The application will open in the browser and allow you to enter student information and generate an exam score prediction.
