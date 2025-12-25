Telecom Customer Churn Prediction using Machine Learning
Overview

Customer churn is a major business problem in the telecom industry, where retaining existing customers is far more cost-effective than acquiring new ones.
This project builds a machine learning–based churn prediction system to identify customers who are likely to leave, enabling proactive retention strategies.

The solution follows a complete ML lifecycle, from data preprocessing and exploratory analysis to model training, hyperparameter tuning, and evaluation.

Business Problem

Predict whether a customer will churn (leave the service)

Identify patterns and drivers behind churn

Support data-driven retention decisions

Machine Learning Problem

Type: Binary Classification

Target Variable: Churn

0 → Not Churned

1 → Churned

Dataset Description

The dataset contains customer-level information such as:

Customer tenure

Contract type

Internet and phone services

Billing and payment details

Monthly and total charges

Churn label (target)

Project Workflow
Data Preprocessing

Dropped non-informative columns (e.g., customerID)

Encoded categorical features

Scaled numerical variables

Split data into training and test sets

Exploratory Data Analysis (EDA)

Churn distribution analysis

Relationship between churn and customer tenure

Impact of contract type and services on churn

Identification of important churn indicators

Model Development

The following ensemble models were trained and evaluated:

Random Forest Classifier

AdaBoost Classifier

Baseline models were compared to identify the best-performing approach.

Model Optimization

Hyperparameter tuning was performed using GridSearchCV.

Optimized AdaBoost parameters:

n_estimators = 200
learning_rate = 0.5


This improved performance from the baseline model.

Model Evaluation

Evaluation metrics used:

Accuracy

Confusion Matrix

Precision, Recall, F1-score

Churn probability predictions

Best Model Performance:

Cross-validated accuracy ≈ 80.6%

Improved over baseline accuracy (~79%)

Key Insights

Ensemble methods outperform simpler models

AdaBoost achieved a good bias–variance balance

Churn probability scores enable prioritization of high-risk customers

Feature importance analysis helps understand churn drivers

Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

RandomForestClassifier

GridSearchCV

Classification metrics