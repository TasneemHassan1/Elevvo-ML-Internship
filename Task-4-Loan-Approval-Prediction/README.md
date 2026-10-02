# Loan Approval Prediction

A machine learning project that predicts whether a loan application is likely to be **Approved** or **Rejected** based on applicant financial and personal information.

The project includes data cleaning, exploratory data analysis, binary classification, imbalance handling with SMOTE, model comparison, overfitting analysis, and a Streamlit web application for interactive predictions.

## Project Overview

Loan approval decisions can depend on several factors such as income, loan amount, credit score, loan term, and asset values.

In this project, machine learning models are trained to predict the loan status of an applicant using a publicly available loan approval dataset.

The project focuses on:

* Cleaning and validating the dataset
* Exploring relationships between applicant features and loan approval
* Encoding categorical features
* Handling class imbalance using SMOTE
* Training and comparing classification models
* Evaluating models using precision, recall, F1-score, and accuracy
* Checking for decision tree overfitting
* Saving the final trained model
* Building an interactive Streamlit prediction app

## Dataset

**Dataset:** Loan Approval Prediction Dataset

**File:** `loan_approval_dataset.csv`

The dataset contains **4,269 loan applications** and **13 columns**.

### Features

| Feature                    | Description                                 |
| -------------------------- | ------------------------------------------- |
| `loan_id`                  | Unique identifier for each loan application |
| `no_of_dependents`         | Number of dependents                        |
| `education`                | Applicant education level                   |
| `self_employed`            | Whether the applicant is self-employed      |
| `income_annum`             | Annual income                               |
| `loan_amount`              | Requested loan amount                       |
| `loan_term`                | Loan repayment term in years                |
| `cibil_score`              | Applicant's CIBIL credit score              |
| `residential_assets_value` | Value of residential assets                 |
| `commercial_assets_value`  | Value of commercial assets                  |
| `luxury_assets_value`      | Value of luxury assets                      |
| `bank_asset_value`         | Value of bank assets                        |
| `loan_status`              | Target variable: Approved or Rejected       |

## Data Cleaning

The dataset was checked for common data quality issues before modeling.

The following steps were performed:

* Removed leading and trailing spaces from column names
* Checked for missing values
* Checked for duplicate rows
* Validated categorical values
* Removed leading and trailing spaces from categorical values
* Cleaned the target labels
* Investigated invalid negative asset values
* Replaced invalid negative values in `residential_assets_value` with `0`
* Kept valid zero asset values
* Excluded `loan_id` from the machine learning features because it is an identifier rather than a meaningful predictive feature

After cleaning, the dataset contained no missing values or duplicate rows.

## Exploratory Data Analysis

Several visualizations and analyses were performed to understand the dataset.

### Loan Status Distribution

The target variable contains:

* **Approved:** 2,656 applications (62.22%)
* **Rejected:** 1,613 applications (37.78%)

This represents a moderate class imbalance.

### CIBIL Score vs Loan Status

CIBIL score showed a strong relationship with loan approval in this dataset.

Average CIBIL scores were approximately:

* Approved: **703.46**
* Rejected: **429.47**

This indicates that credit score is an important feature for distinguishing between the two classes.

### Income vs Loan Amount

Annual income and loan amount showed a strong positive relationship, with a correlation of approximately **0.927**.

This means applicants with higher annual income generally tended to request larger loans in this dataset.

### Other Relationships

The analysis also examined:

* Loan term vs approval status
* Education vs approval rate
* Self-employment status vs approval rate
* Correlations between numerical features
* Feature relationships and distributions

## Data Preprocessing

The target variable was separated from the input features.

`loan_status` was used as the target, while `loan_id` was excluded from modeling.

The categorical features:

* `education`
* `self_employed`

were encoded using `LabelEncoder`.

The data was then divided into training and testing sets using an **80/20 stratified split**.

Feature scaling was applied for Logistic Regression using `StandardScaler`.

Decision Tree was trained using the original unscaled features because tree-based models do not require feature scaling.

## Models

Three classification approaches were evaluated:

### 1. Logistic Regression

The baseline Logistic Regression model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 92.27% |
| Precision | 92.66% |
| Recall    | 95.10% |
| F1-score  | 93.87% |

### 2. SMOTE + Logistic Regression

SMOTE was applied to the training data to balance the two classes.

The SMOTE model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 93.21% |
| Precision | 94.71% |
| Recall    | 94.35% |
| F1-score  | 94.53% |

SMOTE improved the overall performance compared with the baseline Logistic Regression model.

### 3. Decision Tree

An unrestricted Decision Tree initially achieved very high performance but showed signs of overfitting because its training accuracy reached 100%.

Different values of `max_depth` were tested.

The best balance between performance and overfitting was achieved with:

```text
max_depth = 7
```

The final Decision Tree achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 97.78% |
| Precision | 97.06% |
| Recall    | 99.44% |
| F1-score  | 98.23% |

## Model Comparison

| Model                     |   Accuracy |  Precision |     Recall |   F1-score |
| ------------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression       |     92.27% |     92.66% |     95.10% |     93.87% |
| SMOTE Logistic Regression |     93.21% |     94.71% |     94.35% |     94.53% |
| Decision Tree             | **97.78%** | **97.06%** | **99.44%** | **98.23%** |

The **Decision Tree with `max_depth=7`** was selected as the final model because it achieved the best overall performance while reducing the overfitting observed in the unrestricted tree.

## Streamlit Application

A Streamlit web application was created to make the trained model interactive.

The application allows users to enter applicant information including:

* Number of dependents
* Education
* Self-employment status
* Annual income
* Loan amount
* Loan term
* CIBIL score
* Residential asset value
* Commercial asset value
* Luxury asset value
* Bank asset value

After clicking **Predict Loan Approval**, the application displays either:

* **✅ Loan Approved**
* **❌ Loan Rejected**

The application uses the saved model artifact and the same encoders used during training.

## Project Structure

```text
Task-4-Loan-Approval-Prediction/
│
├── data/
│   └── loan_approval_dataset.csv
│
├── artifacts/
│   └── loan_approval_model.pkl
│
├── loan_approval_prediction.ipynb
│
├── app.py
│
├── README.md
│
└── requirements.txt
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

From the project directory:

```bash
streamlit run app.py
```

The application will open in your browser.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Imbalanced-learn
* Joblib
* Streamlit
* Jupyter Notebook

## Key Takeaways

* CIBIL score showed a strong relationship with loan approval in this dataset.
* Income and loan amount had a very strong positive correlation.
* SMOTE improved Logistic Regression performance on the imbalanced training data.
* The Decision Tree performed better than both Logistic Regression approaches.
* Limiting the tree depth to 7 helped reduce overfitting while maintaining strong test performance.
* The final model was saved as a reusable artifact and integrated into a Streamlit application.
