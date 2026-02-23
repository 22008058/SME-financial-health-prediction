# SME-financial-health-prediction
 Problem Statement
Across Southern Africa, small and medium-sized enterprises (SMEs) are vital to employment and economic growth, yet many remain financially fragile and excluded from formal financial systems.
This project builds a machine learning model to predict the Financial Health Index (FHI) of SMEs — classifying businesses into Low, Medium, or High financial health based on:

💾 Savings & Assets
💳 Debt & Repayment Ability
🛡️ Resilience to Shocks
🏦 Access to Credit & Financial Services

Data spans 4 Southern African countries: Eswatini, Lesotho, Zimbabwe, and Malawi.

📁 Project Structure
sme-financial-health-prediction/
│
├── solution_v3.py          # Best submission (0.8929 public F1)
├── solution_v4.py          # Enhanced version (macro F1 optimized)
├── submission_v3.csv       # Best submission file
├── submission_v4.csv       # v4 submission file
└── README.md               # This file

🧠 Approach
1. Exploratory Data Analysis

9,618 training samples, 2,405 test samples
39 raw features (demographics, business data, attitudes, insurance)
Heavy class imbalance: Low 65.3% / Medium 29.8% / High 4.9%
Up to 47% missing values in some columns

2. Feature Engineering
Created 75 features across the 4 FHI dimensions:
CategoryFeatures CreatedFinancial RatiosProfit, expense ratio, profit margin, income-to-turnoverLog TransformsLog turnover, log income, log expensesFHI Dimension 1Savings proxy, financial inclusion scoreFHI Dimension 2Debt burden, income stabilityFHI Dimension 3Vulnerability flag, resilience risk scoreFHI Dimension 4Credit diversity, total access scoreInteractionsfin × attitude, access × profit, age × fin scoreBusiness AgeTotal months, age group (new/young/mature/established)
3. Preprocessing

OrdinalEncoder for categorical features (handles unknowns + NaN natively)
MedianImputer for numeric features
SMOTE oversampling to boost minority High class

4. Ensemble Model
Weighted soft-voting ensemble of 3 models:
ModelWeightRoleHistGradientBoosting / LightGBM35%Primary learnerRandom Forest33%Variance reductionExtra Trees33%Diversity
5. Threshold Tuning
Multi-class threshold optimization:

Default argmax predicts too few High businesses
Grid search over High threshold (0.10–0.60) and Medium threshold (0.15–0.60)
Optimized for macro F1 to treat all classes equally
Best thresholds: High = 0.34, Medium = 0.53

6. Cross Validation

5-Fold Stratified CV with country-aware stratification
Ensures all 4 countries represented in every fold
Stable scores (std dev < 0.009)


📊 Model Performance
VersionModelF1 ScoreHigh RecallBaselineLogistic Regression0.62710.72v2Random Forest0.86490.66v3RF + ET + HGB Ensemble0.87000.67v4Full Ensemble + Macro F10.8929*0.67
*Public leaderboard score

🚀 How to Run
Install Dependencies
bashpip install scikit-learn pandas numpy lightgbm xgboost catboost imbalanced-learn
Run Best Solution
bash# Place Train.csv and Test.csv in the same folder
python solution_v4.py
Output

submission_v4.csv — ready to submit to Zindi


🔑 Key Learnings

Metric matters — optimizing weighted F1 hides minority class weakness; macro F1 is stricter and more honest
Threshold tuning gave a significant boost without changing the model at all
SMOTE on the High class (only 4.9% of data) directly improved recall from 0.62 → 0.67+
Country stratification in CV gave more reliable estimates across different economic contexts
Simple ensembles (soft voting) often outperform complex stacking on tabular data


📦 Dataset

Source: data.org x Zindi Challenge
License: CC-BY SA 4.0
Countries: Eswatini, Lesotho, Zimbabwe, Malawi
Provided by: FinMark Trust


Note: Dataset not included in this repo per competition rules. Download from the Zindi challenge page.
