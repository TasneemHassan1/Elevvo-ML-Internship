# Task 2 - Customer Segmentation

## Overview

This project applies unsupervised machine learning to segment mall customers based on their **Annual Income** and **Spending Score**.

The main clustering method is **K-Means**, with **DBSCAN** implemented as a bonus comparison.

A Streamlit application is also included to demonstrate how the trained K-Means model can be used to predict the customer segment of a new customer.

---

## Dataset

**Dataset:** Mall Customer Segmentation Data

The dataset contains information about mall customers, including:

* Customer ID
* Gender
* Age
* Annual Income (k$)
* Spending Score (1-100)

The dataset contains **200 customers** and **5 columns**.

For the main clustering analysis, **Annual Income** and **Spending Score** were selected because the task focuses on identifying customer segments based on income and spending behavior.

`CustomerID` was excluded because it is an identifier, while `Gender` and `Age` were not used as clustering features.

---

## Project Workflow

```text
Load Dataset
      ↓
Initial Data Inspection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Selection
      ↓
Feature Scaling
      ↓
Find Optimal K
      ↓
K-Means Clustering
      ↓
Cluster Analysis
      ↓
Business Interpretation
      ↓
DBSCAN Bonus
      ↓
Save Trained Model
      ↓
Streamlit Prediction App
```

---

## Data Cleaning

The dataset was checked for:

* Missing values
* Duplicate rows
* Invalid values
* Reasonable ranges for the selected features

No missing values or duplicate rows were found.

The values in **Age**, **Annual Income**, and **Spending Score** were within reasonable ranges, so no rows needed to be removed or imputed.

---

## Exploratory Data Analysis

Several visualizations were used to understand the customer data before clustering:

* Customer age distribution
* Annual income distribution
* Spending score distribution
* Annual income vs spending score

The scatter plot showed noticeable differences in customer behavior based on income and spending score, suggesting that meaningful customer segments could be identified.

---

## Feature Scaling

K-Means is based on distance calculations, so the selected features were standardized using `StandardScaler`.

The two features used for clustering were:

```text
Annual Income (k$)
Spending Score (1-100)
```

---

## K-Means Clustering

Different values of `K` from 2 to 10 were evaluated using:

* Elbow Method
* Silhouette Score

The highest Silhouette Score was obtained with:

```text
K = 5
Silhouette Score ≈ 0.555
```

Therefore, **5 clusters** were selected for the final K-Means model.

---

## K-Means Results

The five customer segments were analyzed using their average income and spending score.

| Segment                             | Average Income (k$) | Average Spending Score | Customers |
| ----------------------------------- | ------------------: | ---------------------: | --------: |
| Average Customers                   |               55.30 |                  49.52 |        81 |
| High-Income High-Spending Customers |               86.54 |                  82.13 |        39 |
| Low-Income High-Spending Customers  |               25.73 |                  79.36 |        22 |
| High-Income Low-Spending Customers  |               88.20 |                  17.11 |        35 |
| Low-Income Low-Spending Customers   |               26.30 |                  20.91 |        23 |

### Segment Interpretation

**Average Customers**

Medium-income and medium-spending customers. This is the largest segment in the dataset.

**High-Income High-Spending Customers**

Customers with both high income and high spending scores. They may be an important target for premium products, exclusive offers, and personalized marketing.

**Low-Income High-Spending Customers**

Customers with lower income but relatively high spending scores. Promotions and value-oriented offers may be suitable for this segment.

**High-Income Low-Spending Customers**

Customers with high income but relatively low spending scores. More targeted engagement strategies may help increase their spending.

**Low-Income Low-Spending Customers**

Customers with both low income and low spending scores. Affordable products and value-oriented promotions may be more suitable for this segment.

These interpretations are based only on **Annual Income** and **Spending Score**. Additional customer information would be needed to understand the reasons behind these behaviors.

---

## Business Interpretation

Customer segmentation can help a mall understand different groups of customers and design more targeted marketing strategies.

For example:

* High-income, high-spending customers may be targeted with premium offers.
* Low-income, high-spending customers may respond well to attractive promotions.
* High-income, low-spending customers may benefit from engagement campaigns.
* Low-income, low-spending customers may be more interested in affordable and value-oriented products.

The purpose of these segments is to support better customer understanding and more targeted marketing decisions.

---

## Bonus: DBSCAN

DBSCAN was implemented as a bonus clustering method.

Unlike K-Means, DBSCAN:

* Does not require the number of clusters to be specified in advance.
* Groups points based on density.
* Can identify points that do not belong to sufficiently dense regions as noise.

Several `eps` values were tested while keeping:

```text
min_samples = 5
```

The tested `eps` values were:

```text
0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

The best Silhouette Score among the valid DBSCAN configurations was obtained with:

```text
eps = 0.3
Silhouette Score ≈ 0.5243
```

This configuration produced:

```text
7 clusters
35 noise points
```

The Silhouette Score was calculated using the non-noise points because DBSCAN labels noise as `-1`, which does not represent a regular cluster.

### DBSCAN vs K-Means

K-Means produced five clearly interpretable customer segments without noise points.

DBSCAN produced a more detailed density-based structure, with seven clusters and 35 noise points.

For the main business segmentation task, K-Means was preferred because its five segments were simpler and easier to interpret.

---

## Model Artifact

The final K-Means model and preprocessing objects were saved using `joblib`.

The saved artifact contains:

* Trained K-Means model
* Fitted StandardScaler
* Cluster summary
* Cluster counts
* Cluster names
* Cluster descriptions

Artifact:

```text
artifacts/kmeans_customer_segmentation.pkl
```

The Streamlit application loads this artifact instead of retraining the model.

---

## Streamlit Application

The project includes a Streamlit application that provides a simple interface for predicting the segment of a new customer.

The user enters:

* Annual Income
* Spending Score

The application then:

1. Receives the customer input.
2. Applies the saved `StandardScaler`.
3. Uses the saved K-Means model to predict the cluster.
4. Displays the customer segment.
5. Shows the average income and spending score of that segment.
6. Shows the number of customers in the segment.
7. Visualizes the new customer's position among the existing customer segments.

The application is designed for **model inference and demonstration**, while the complete data analysis and clustering workflow remains in the Jupyter Notebook.

---

## Project Structure

```text
Task-2-Customer-Segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── artifacts/
│   └── kmeans_customer_segmentation.pkl
│
├── customer_segmentation.ipynb
├── app.py
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook
* Streamlit

---

## How to Run

### 1. Install the required libraries

```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook

Open:

```text
customer_segmentation.ipynb
```

and run the cells to reproduce the analysis and clustering workflow.

### 3. Run the Streamlit Application

From the repository root:

```bash
streamlit run Task-2-Customer-Segmentation/app.py
```

The application will open in the browser.

---

## Conclusion

This project demonstrates how unsupervised machine learning can be used to identify meaningful customer segments from income and spending behavior.

K-Means clustering was selected as the main method after evaluating different values of `K` using the Elbow Method and Silhouette Score. The final model identified five customer segments with distinct income and spending patterns.

DBSCAN was also implemented as a bonus to compare a density-based clustering approach with K-Means.

Finally, the trained K-Means model was saved as an artifact and integrated into a Streamlit application for interactive customer segment prediction.
