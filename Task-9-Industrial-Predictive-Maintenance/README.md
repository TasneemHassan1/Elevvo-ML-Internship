# Task 9 - Industrial Predictive Maintenance

## Project Overview

This project focuses on predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset.

The goal is to predict whether a machine failure is likely to occur based on machine and sensor readings such as temperature, rotational speed, torque, and tool wear.

The project also analyzes different failure modes and applies anomaly detection, feature engineering, cost-sensitive learning, and prediction threshold tuning.

A key industrial requirement of this project is minimizing false failure alerts. Therefore, **False Discovery Rate (FDR)** is used as the main evaluation constraint when selecting the final operating point.

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains 10,000 machine records and includes machine type, air temperature, process temperature, rotational speed, torque, tool wear, machine failure, and five failure-mode indicators:

* Tool Wear Failure (TWF)
* Heat Dissipation Failure (HDF)
* Power Failure (PWF)
* Overstrain Failure (OSF)
* Random Failure (RNF)

The dataset contains no missing values and no duplicate records.

Machine failure is highly imbalanced:

* No Failure: 9,661 records
* Failure: 339 records

## Project Objectives

The main objectives are:

* Predict machine failure using sensor readings.
* Predict individual failure modes using a multi-label classification approach.
* Detect unusual machine operating conditions using anomaly detection.
* Engineer additional features from sensor interactions.
* Apply cost-sensitive learning to reduce false-positive failure alerts.
* Tune the prediction threshold according to the industrial FDR requirement.
* Analyze which sensors and engineered features contribute most to failure predictions.
* Perform correlation analysis as a bonus task.
* Investigate the feasibility of Time-to-Failure / Remaining Useful Life estimation.

## Feature Engineering

Three additional features were created:

* `Temperature Difference [K]`
* `Torque-Speed Interaction`
* `Tool Wear-Torque Interaction`

These features are calculated only from sensor measurements and do not use failure labels, preventing target leakage.

The interaction features were particularly useful for the final Random Forest model.

## Anomaly Detection

Isolation Forest was used as an unsupervised anomaly detection method.

The model was trained without using the machine failure target.

The anomaly detector achieved high recall but produced a high False Discovery Rate:

* Validation FDR: 87.83%
* Test FDR: 85.20%

This demonstrates that unusual operating conditions do not necessarily correspond to actual machine failures.

Therefore, Isolation Forest is treated as a complementary anomaly-detection technique rather than the final failure classifier.

## Failure Type Classification

The dataset contains multiple failure-mode indicators, and some machine records can contain more than one failure mode.

Therefore, failure-type prediction was formulated as a **multi-label classification problem**.

A separate binary prediction was made for each failure mode:

* TWF
* HDF
* PWF
* OSF
* RNF

A Random Forest model with class weighting was used for the multi-label prediction.

Performance varied across the failure modes. HDF, PWF, and OSF showed strong validation performance, while the rare TWF and RNF classes were more difficult to predict.

The rare failure modes contain very few positive examples, so their metrics should be interpreted cautiously.

## Classification Models

Several supervised models and configurations were evaluated.

### Logistic Regression

The Logistic Regression baseline achieved:

* Accuracy: 86.20%
* Precision: 16.95%
* Recall: 78.43%
* F1-score: 27.87%
* FDR: 83.05%

The model produced many false-positive failure alerts and was therefore not suitable for the main industrial operating point.

### Random Forest

The standard Random Forest baseline achieved:

* Accuracy: 99.27%
* Precision: 93.48%
* Recall: 84.31%
* F1-score: 88.66%
* FDR: 6.52%

This substantially improved false-alarm behavior compared with Logistic Regression.

### Cost-Sensitive Random Forest

Cost-sensitive Random Forest models were evaluated by increasing the weight assigned to the negative class.

Increasing the negative-class weight made the model more conservative when predicting failure.

A negative-class weight of 2 achieved:

* Precision: 97.56%
* Recall: 78.43%
* FDR: 2.44%

This configuration was selected for further threshold tuning.

### XGBoost

XGBoost was also evaluated using cost-sensitive sample weighting.

The tested configurations achieved FDR values between 13.04% and 16.28%.

The cost-sensitive Random Forest produced a lower FDR on the validation set, so it was used for the final model.

## Threshold Tuning

The classification threshold was tuned using the validation set rather than the test set.

Because the industrial requirement prioritizes minimizing false-positive alerts, FDR was used as the main selection criterion.

A project-level constraint of **Recall >= 50%** was used to prevent reducing FDR simply by predicting almost every observation as a non-failure.

A threshold of **0.55** was selected.

Validation performance at this threshold:

* Accuracy: 99.20%
* Precision: 100%
* Recall: 76.47%
* F1-score: 86.67%
* FDR: 0%
* False Positives: 0
* False Negatives: 12

Higher thresholds also produced zero FDR, but resulted in progressively lower recall.

Therefore, 0.55 provided the selected operating point while maintaining the highest recall among the zero-FDR thresholds tested.

## Final Test Results

The final model was evaluated on the untouched test set using the selected threshold of 0.55.

| Metric    |  Result |
| --------- | ------: |
| Accuracy  |  99.07% |
| Precision | 100.00% |
| Recall    |  72.55% |
| F1-score  |  84.09% |
| FDR       |   0.00% |

Confusion matrix:

```text
[[1449    0]
 [  14   37]]
```

This corresponds to:

* True Negatives: 1,449
* False Positives: 0
* False Negatives: 14
* True Positives: 37

The model produced no false failure alerts on the unseen test set.

However, it did not detect every actual failure, with a recall of 72.55%.

Therefore, the final model prioritizes reliable failure alerts and a very low false-alarm rate rather than maximizing failure detection at all costs.

## Feature Importance

The most important features in the final Random Forest were:

| Feature                      | Importance |
| ---------------------------- | ---------: |
| Torque-Speed Interaction     |     21.48% |
| Tool Wear-Torque Interaction |     19.22% |
| Rotational speed             |     15.98% |
| Torque                       |     13.20% |
| Temperature Difference       |     10.50% |
| Tool wear                    |      5.84% |

The engineered interaction features ranked among the most important predictors, supporting the usefulness of feature engineering.

Feature importance indicates how useful a feature was to the trained model and does not establish a causal relationship with machine failure.

## Bonus — Correlation Analysis

Correlation analysis was performed between the sensor measurements and machine failure.

The strongest correlations were:

| Feature             | Correlation with Machine Failure |
| ------------------- | -------------------------------: |
| Torque              |                            0.191 |
| Tool wear           |                            0.105 |
| Air temperature     |                            0.083 |
| Rotational speed    |                           -0.044 |
| Process temperature |                            0.036 |

Torque showed the strongest direct association with machine failure among the analyzed sensor variables.

However, correlation does not establish causation.

The available AI4I 2020 CSV does not contain an explicit timestamp or a temporal sequence of sensor readings leading up to each failure. Therefore, the analysis cannot prove that a sensor is a temporal lead indicator that changes before failure.

## Bonus — Time-to-Failure / Remaining Useful Life

A true Time-to-Failure or Remaining Useful Life (RUL) regression model could not be reliably implemented using the available dataset.

The available AI4I 2020 CSV does not provide:

* An explicit timestamp for each observation
* Failure times
* Remaining useful life values
* An explicit RUL target

Creating an artificial RUL target from the available columns would require unsupported assumptions.

Therefore, this bonus requirement is documented as a dataset limitation rather than using a fabricated regression target.

## Final Model Artifact

The final model was saved as:

```text
artifacts/predictive_maintenance_model.pkl
```

The artifact contains:

* Trained cost-sensitive Random Forest
* Selected prediction threshold
* Feature column information
* Target definition
* Engineered feature information
* Model configuration

The saved artifact was successfully reloaded and tested to verify that it can be used for future inference.

## Project Structure

```text
Task-9-Industrial-Predictive-Maintenance/
├── artifacts/
│   └── predictive_maintenance_model.pkl
├── data/
│   └── ai4i2020.csv
├── Industrial-Predictive-Maintenance.ipynb
├── README.md
└── requirements.txt
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Joblib

## Main Concepts

* Predictive Maintenance
* Binary Classification
* Multi-Label Classification
* Anomaly Detection
* Isolation Forest
* Random Forest
* XGBoost
* Cost-Sensitive Learning
* Feature Engineering
* Threshold Tuning
* False Discovery Rate (FDR)
* Correlation Analysis
* Model Deployment and Artifact Saving

## Conclusion

This project demonstrates a predictive maintenance workflow that goes beyond maximizing accuracy.

The final cost-sensitive Random Forest was tuned specifically for the industrial requirement of minimizing false failure alerts.

On the unseen test set, the selected operating point achieved **0% FDR**, **100% precision**, and **72.55% recall**.

The results show that the model can provide highly reliable failure alerts while still detecting a substantial proportion of actual failures.

The project also demonstrates the importance of feature engineering, cost-sensitive learning, threshold selection, and careful interpretation of imbalanced failure data.
