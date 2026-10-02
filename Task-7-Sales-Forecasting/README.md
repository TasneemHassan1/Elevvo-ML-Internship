# Task 7 - Sales Forecasting

## Project Overview

This project focuses on forecasting weekly sales using the Walmart Sales Forecast dataset.

The main goal is to use historical sales data, time-based features, and previous sales values to predict future weekly sales.

## Dataset

The dataset used in this project is the Walmart Sales Forecast dataset from Kaggle.

It contains information about:

* Store and department
* Weekly sales
* Date
* Holiday information
* Temperature
* Fuel price
* MarkDown features
* CPI
* Unemployment
* Store type and size

The dataset files used in the project are:

* `train.csv`
* `features.csv`
* `stores.csv`
* `test.csv`

## Data Cleaning and Preprocessing

The data was checked for missing values and duplicate rows.

The following preprocessing steps were performed:

* Missing values in the MarkDown columns were filled with `0`.
* Missing CPI and Unemployment values were interpolated separately for each store.
* Negative weekly sales values were inspected and retained because they may represent returns or sales adjustments.
* The datasets were merged using the Store and Date columns.
* Duplicate feature columns created during the merge were removed.

## Feature Engineering

Several time-based and historical sales features were created.

### Time-Based Features

* Year
* Month
* Week
* Day

### Lag Features

* `Lag_1` - sales from the previous week
* `Lag_2` - sales from two weeks earlier
* `Lag_4` - sales from four weeks earlier

### Rolling Feature

* `Rolling_Mean_4` - four-week rolling average based only on previous sales values

The current week's sales were excluded from the rolling average to avoid data leakage.

## Train-Test Split

A chronological train-test split was used instead of a random split.

The data was split based on unique dates, with approximately 80% of the dates used for training and the remaining 20% used for testing.

The training period was:

`2010-04-02` to `2012-04-20`

The testing period was:

`2012-04-27` to `2012-10-26`

This preserves the time order and prevents future dates from being used to train the model before earlier dates are predicted.

## Models

Two main regression models were trained:

* Linear Regression
* Random Forest Regression

The models were evaluated using:

* MAE
* MSE
* RMSE
* R²

Random Forest achieved the following results on the final testing period:

* MAE: `1359.97`
* RMSE: `2852.11`
* R²: `0.9832`

## Actual vs Predicted Sales

The actual and predicted weekly sales were plotted over time to visually compare the model's predictions with the real sales values.

## Feature Importance

Feature importance was examined using the Random Forest model.

The most important features were mainly related to previous sales values, especially:

* `Lag_1`
* `Rolling_Mean_4`
* `Lag_4`
* `Week`
* `Lag_2`

This shows that recent historical sales were important for predicting future weekly sales.

## Bonus Analysis

### Rolling Average

A four-week rolling average was created to capture recent sales trends for each Store and Department.

### Seasonal Decomposition

Seasonal decomposition was performed on the overall weekly sales.

A period of 52 weeks was used to represent an approximate yearly seasonal cycle.

The decomposition showed:

* An overall sales trend over time
* Recurring yearly seasonal patterns
* Larger residual variations around some major sales spikes

### XGBoost

XGBoost was also tested using time-aware validation with `TimeSeriesSplit` and three chronological folds.

The average validation results were:

* Mean MAE: `2071.22`
* Mean RMSE: `5913.29`
* Mean R²: `0.9280`

These results are based on time-aware cross-validation, while the Random Forest results are based on the final chronological test period. Therefore, the two sets of results are not directly comparable.

## Saved Model

The final Random Forest model was saved as:

`artifacts/sales_forecasting_model.pkl`

The saved model was successfully loaded again, and its predictions matched the original predictions.

## Tools and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Statsmodels
* XGBoost
* Joblib
* Jupyter Notebook