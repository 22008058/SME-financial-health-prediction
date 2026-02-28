📌 Problem Statement

Across Southern Africa, small and medium-sized enterprises (SMEs) are vital to employment
and economic growth, yet many remain financially fragile and excluded from formal financial systems.
This project builds a machine learning model to predict the Financial Health Index (FHI)
of SMEs — classifying businesses into Low, Medium, or High financial health based on:

💾 Savings & Assets

💳 Debt & Repayment Ability

🛡️ Resilience to Shocks

🏦 Access to Credit & Financial Services

Data spans 4 Southern African countries: Eswatini, Lesotho, Zimbabwe, and Malawi.

📁 Project Structure

sme-financial-health-prediction/

solution.py                  # Full ML pipeline (LightGBM + SMOTE + Ensemble)

explore.py                   # Country-level financial health analysis

sme_dashboard_final.xlsx     # Interactive Excel dashboard (6 sections + 3 sheets)

submission_v4.csv            # Latest submission file

README.md                    # This file

🌍 Country-Level Insight (from explore.py)

Real data extracted directly from Train.csv using explore.py:
CountryHigh %Low %Medium %Priority🇸🇿 Eswatini11.5%51.4%37.1%
✅ Regional success model🇱🇸 Lesotho0.3%60.4%39.3%
🔴 Glass ceiling effect🇲🇼 Malawi4.0%81.2%14.7%
🔴 Highest priority — 81% Low🇿🇼 Zimbabwe2.3%68.6%29.1%
🟡 Moderate — improving
Key Finding: Eswatini has 11.5% High-health SMEs — nearly 5× the regional average.
Malawi's 81.2% Low rate signals the most urgent need for financial inclusion intervention.

💼 What This Means For Business

Financial Institutions:

Can identify 94% of financially vulnerable SMEs (Low class recall = 0.93)
Banks can use predictions for targeted lending and financial inclusion programs
Reduces manual SME assessment cost for microfinance institutions

Development Partners & NGOs:

Malawi needs most urgent intervention — 81.2% of SMEs are Low health
Lesotho shows a glass ceiling — 39.3% Medium but only 0.3% reach High
Eswatini is a replicable success model for the region

Government & Policy:

Country-level health scores enable targeted subsidy allocation
Model can be retrained annually to track financial inclusion progress
Zimbabwe and Malawi show highest need for credit access programs


🧠 Approach

1. Feature Engineering — 75 Features Created

Financial Ratios - Profit, expense ratio, profit margin, income-to-turnover
Log TransformsLog turnover, log income, log expenses
FHI Dimension 1 — Savings & AssetsSavings proxy, financial inclusion score
FHI Dimension 2 — Debt & RepaymentDebt burden, income stability
FHI Dimension 3 — ResilienceVulnerability flag, resilience risk score
FHI Dimension 4 — Credit AccessCredit diversity, total access score
Interaction Featuresfin × attitude, access × profit, age × fin score
Business AgeTotal months, age group (new/young/mature/established)

2. Preprocessing

OrdinalEncoder for categorical features
MedianImputer for numeric features
Full NaN imputation before SMOTE (most_frequent strategy)
SMOTE oversampling — boosts High class to 50% of Low, Medium to 70% of Low

3. Ensemble Model — 5 Models

Model          Weight        Role
LightGBM ✅    35%          Primary learner — fastest & strongest
XGBoost ✅     25%          Diverse boosting patterns
CatBoost ✅    20%          Handles categoricals natively
Random Forest  10%          Variance reductionExtra Trees10%Adds diversity

4. Threshold Tuning

Grid search over High threshold (0.10–0.60) and Medium threshold (0.15–0.60)
Optimized for Macro F1 to treat all 3 classes equally
Best thresholds: High = 0.56, Medium = 0.45

5. Cross Validation

5-Fold Stratified CV with country-aware stratification
Ensures all 4 countries represented in every fold
Stable scores (std dev = 0.0105)


📊 Excel Dashboard

The file sme_dashboard_final.xlsx contains a full interactive dashboard with 3 sheets:

🏠 Sheet 1 — Dashboard

KPI cards (Macro F1, features, samples, countries, models)
5-fold CV performance table
Per-class precision, recall, F1 breakdown
Class distribution with imbalance analysis
Country comparison with real data
Top 10 feature importance ranked table
Business impact section
Interpretation of all key decisions

📈 Sheet 2 — Charts

Fold F1 bar chart
Class distribution pie chart
Country financial health stacked bar chart (real data)
Model progression v1 → v4
Feature importance horizontal bar chart
Per-class metrics comparison chart

🌍 Sheet 3 — Country Deep Dive

Full country comparison table
Individual insight cards with policy recommendations per country
Identifies Malawi as highest priority, Eswatini as regional benchmark


🚀 How to Run

Install Dependencies

bashpip install lightgbm xgboost catboost imbalanced-learn scikit-learn pandas numpy

Update File Paths

Open solution.py and update these lines with your own paths:

pythontrain = pd.read_csv(r"C:\Your\Path\Train.csv")

test  = pd.read_csv(r"C:\Your\Path\Test.csv")

Run Main Model
bashpython solution.py

Run Country Analysis
bashpython explore.py

Expected Output (solution.py)

✅ LightGBM found
✅ XGBoost found
✅ CatBoost found
✅ SMOTE found


🔑 Key Learnings

Metric matters — optimizing weighted F1 hides minority class weakness; macro F1 forces the model to perform well on the rare High class

SMOTE needs clean data — impute ALL NaNs (including from OrdinalEncoder) before applying SMOTE or it crashes with a ValueError

Threshold tuning gave a significant boost without changing the model at all

Country stratification in CV gave more reliable estimates across different economic contexts

5-model ensemble (LGB + XGB + CatBoost + RF + ET) is more robust than any single model

Real country analysis revealed Malawi (81.2% Low) and Lesotho (0.3% High) need the most urgent intervention — a finding invisible without the explore.py breakdown


📦 Dataset

Source: Zindi — data.org Financial Health Prediction Challenge

License: CC-BY SA 4.0

Countries: Eswatini, Lesotho, Zimbabwe, Malawi


Provided by: FinMark Trust

⚠️ Dataset not included per competition rules. Download from the Zindi challenge page.


🛠️ Tech Stack

ToolPurposePython 3.14Core language  LightGBMPrimary gradient boosting model  XGBoostSecondary gradient boosting model  CatBoostTertiary gradient boosting modelscikit-learnRF, ET, preprocessing, CVimbalanced-learnSMOTE oversamplingpandas + NumPyData manipulation & analysisopenpyxlExcel dashboard creation

👤 Author
Phumlani Mbatha

