Problem Statement
Across Southern Africa, small and medium-sized enterprises (SMEs) are vital to employment
and economic growth, yet many remain financially fragile and excluded from formal financial systems.
This project builds a machine learning model to predict the Financial Health Index (FHI)
of SMEs — classifying businesses into Low, Medium, or High financial health based on:

Savings & Assets
Debt & Repayment Ability
Resilience to Shocks
Access to Credit & Financial Services

Data spans 4 Southern African countries: Eswatini, Lesotho, Zimbabwe, and Malawi.

Project Structure

SME-Financial-Health-Prediction/

solution_v4.py       # Full ML pipeline (LightGBM + SMOTE + Ensemble)

submission_v4.csv    # Latest submission file


README.md            # This file

Approach
1. Feature Engineering — 75 Features Created
Financial Ratios -Profit, expense ratio, profit margin, income-to-turnover
Log Transforms - Log turnover, log income, log expenses
FHI Dimension 1 — Savings & AssetsSavings proxy, financial inclusion score
FHI Dimension 2 — Debt & RepaymentDebt burden, income stability
FHI Dimension 3 — ResilienceVulnerability flag, resilience risk score
FHI Dimension 4 — Credit AccessCredit diversity, total access score
Interaction Features - fin × attitude, access × profit, age × fin score
Business Age - Total months, age group (new/young/mature/established)

3. Preprocessing
OrdinalEncoder for categorical features
MedianImputer for numeric features
Full NaN imputation before SMOTE (most_frequent strategy)
SMOTE oversampling — boosts High class to 50% of Low, Medium to 70% of Low

4.  Cross Validation
5-Fold Stratified CV with country-aware stratification
Ensures all 4 countries represented in every fold
Stable scores (std dev = 0.0105)


How to Run
Install Dependencies
bashpip install lightgbm xgboost catboost imbalanced-learn scikit-learn pandas numpy
Update File Paths
Open solution_v4.py and update these lines with your own paths:

train = pd.read_csv(r"C:\Your\Path\Train.csv")

test  = pd.read_csv(r"C:\Your\Path\Test.csv")

Run

bashpython solution_v4.py
Expected Output
LightGBM found
XGBoost found
CatBoost found
SMOTE found

Mean MACRO F1: 0.8038
submission_v4.csv saved!

Key Learnings

Metric matters — optimizing weighted F1 hides minority class weakness; macro F1 forces the model to perform well on the rare High class

SMOTE needs clean data — impute ALL NaNs (including from OrdinalEncoder) before applying SMOTE or it crashes

Threshold tuning gave a significant boost without changing the model at all

Country stratification in CV gave more reliable estimates across different economic contexts

5-model ensemble (LGB + XGB + CatBoost + RF + ET) is more robust than any single model


Dataset

Source: Zindi — data.org Financial Health Prediction Challenge

License: CC-BY SA 4.0

Countries: Eswatini, Lesotho, Zimbabwe, Malawi

Provided by: FinMark Trust

Dataset not included per competition rules. Download from the Zindi challenge page.

Tech Stack
Python 3.8+ -Language
LightGBM - Primary gradient boosting model

XGBoost - Secondary gradient boosting model


CatBoost - Tertiary gradient boosting model

scikit-learn - RF, ET, preprocessing, CV

imbalanced-learn - SMOTE oversampling
Pandas + NumPy - Data manipulation

Author
Phumlani Mbatha

