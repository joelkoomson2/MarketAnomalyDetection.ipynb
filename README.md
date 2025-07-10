## Overview

This repository contains a machine learning pipeline that detects market anomalies using Bloomberg financial data. The project utilizes a **RandomForest Classifier** to identify anomalies based on historical market indicators. The primary focus is on **data preprocessing, feature imputation, and model training** to ensure robust detection of unusual market behaviors. 

This README provides an **explanation** of the methodology used rather than a step-by-step guide, helping developers and financial analysts understand the reasoning behind the approach.

## Table of Contents

- [Market Anomaly Detection with Python and Bloomberg Data](#market-anomaly-detection-with-python-and-bloomberg-data)
  - [Overview of Market Anomaly Detection](#overview-of-market-anomaly-detection)
  - [Code Demo using Google Colab](#code-demo-using-google-colab)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Glossary](#glossary)
  - [Data Preparation](#data-preparation)
  - [Handling Missing Data](#handling-missing-data)
  - [RandomForest for Anomaly Detection](#randomforest-for-anomaly-detection)
    - [Model Evaluation](#model-evaluation)
  - [Model Persistence and Deployment](#model-persistence-and-deployment)
    - [Reference](#reference)

## Glossary

- **Market Anomaly**: A deviation from expected market behavior that might indicate investment opportunities or risks.
- **RandomForest Classifier**: An ensemble learning technique that builds multiple decision trees and combines their predictions for improved accuracy.
- **Imputation**: Filling in missing data to ensure the dataset is complete.
- **KNN Imputation**: Replacing missing values with those from the nearest neighbors in the dataset.
- **Median Imputation**: Filling missing values with the median of the column, useful for handling outliers.
- **Forward Fill**: Propagating the last known valid data point forward to fill in missing values.

## Data Preparation

The project begins by loading a **financial dataset from Bloomberg**. The dataset includes a variety of market indicators such as:

- **Commodities:** Gold prices (`XAU BGNL`), crude oil (`Cl1`)
- **Exchange Rates:** Dollar index (`DXY`), Japanese Yen (`JPY`), British Pound (`GBP`)
- **Market Volatility:** VIX (Volatility Index)
- **Interest Rates:** Various US and global bond yields
- **Equity Indexes:** Market indexes from different regions (`MXUS`, `MXEU`, `MXJP`, etc.)

These indicators serve as input features, while the target variable (`Y`) indicates whether a market anomaly has occurred. The data is then **split into training and testing sets** to evaluate model performance.

## Handling Missing Data

Financial datasets often contain **missing or incomplete data**, which can negatively impact model performance. To address this, three imputation techniques are used:

1. **KNN Imputation**  
   - Applied to features like **bond yields and interest rates**.
   - Uses nearest neighbors to predict missing values.
   
2. **Median Imputation**  
   - Used for features like **economic surplus and overnight interest rates (EONIA)**.
   - A stable method that mitigates the impact of outliers.
   
3. **Forward Fill**  
   - Applied to **commodity prices, volatility indexes, and stock market indices**.
   - Ensures missing values are replaced with the latest available data.

This **multi-strategy approach** enhances data quality, ensuring the machine learning model receives a complete dataset for training.

## RandomForest for Anomaly Detection

After preprocessing, a **RandomForest Classifier** is trained to detect market anomalies. The key advantages of using RandomForest include:

- **Handling high-dimensional data**: Works well with multiple market indicators.
- **Robustness to overfitting**: Since multiple decision trees are used, the model generalizes well.
- **Feature importance analysis**: Helps identify which factors contribute most to market anomalies.

### Model Evaluation

After training, the model is evaluated using:

- **Accuracy Score**: Measures overall correctness.
- **Confusion Matrix**: Displays the distribution of predictions.
- **Classification Report**: Shows precision, recall, and F1-score for anomaly detection.

Results from testing indicate **high accuracy (~93%)**, meaning the model effectively identifies anomalies.
<img width="358" alt="Screenshot 2025-02-13 at 5 46 55 PM" src="https://github.com/user-attachments/assets/ea9b0eac-214d-4305-ad03-b5d3153167b2" />

## Model Persistence and Deployment

To make the trained model reusable, **Joblib** is used to save both the model and imputation transformers. This enables deployment without retraining:

```python
import joblib

# Save the trained RandomForest model
