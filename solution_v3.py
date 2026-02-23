"""
data.org Financial Health Prediction Challenge
Solution v3 — LightGBM + SMOTE + Threshold Tuning
====================================================
Requirements (install before running):
    pip install lightgbm imbalanced-learn scikit-learn pandas numpy

Run:
    python solution_v3.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ── Try importing optional heavy libs ────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
    print("✅ LightGBM found")
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not found — falling back to HistGradientBoosting")
    from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
    print("✅ imbalanced-learn (SMOTE) found")
except ImportError:
    HAS_SMOTE = False
    print("⚠️  imbalanced-learn not found — SMOTE will be skipped")

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════
# ⚠️  Update these paths if running locally
train = pd.read_csv("Train.csv")
test  = pd.read_csv("Test.csv")

print(f"\nTrain: {train.shape}  |  Test: {test.shape}")
print("Target distribution:\n", train['Target'].value_counts(), "\n")

X      = train.drop(columns=["Target", "ID"])
y      = train["Target"]
X_test = test.drop(columns=["ID"])

# ══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
def engineer(df):
    df = df.copy()

    # Financial ratios
    df['profit']           = df['business_turnover'] - df['business_expenses']
    df['expense_ratio']    = df['business_expenses'] / (df['business_turnover'] + 1)
    df['income_to_turn']   = df['personal_income']   / (df['business_turnover'] + 1)
    df['profit_margin']    = df['profit']             / (df['business_turnover'] + 1)

    # Log transforms (handle skewness)
    for col in ['business_turnover', 'personal_income', 'business_expenses']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0).fillna(0))

    # Business age in total months
    df['biz_age_months'] = (
        df['business_age_years'].fillna(0) * 12 +
        df['business_age_months'].fillna(0)
    )

    # Binary: has this product NOW?
    def has_now(val):
        if pd.isna(val): return 0
        s = str(val).lower()
        return 1 if ('have now' in s or s == 'yes') else 0

    fin_cols = ['has_mobile_money', 'has_credit_card', 'has_loan_account',
                'has_internet_banking', 'has_debit_card']
    ins_cols = ['has_insurance', 'motor_vehicle_insurance',
                'medical_insurance', 'funeral_insurance']
    att_cols = ['attitude_stable_business_environment',
                'attitude_satisfied_with_achievement',
                'attitude_more_successful_next_year']

    for c in fin_cols + ins_cols + att_cols:
        if c in df.columns:
            df[f'{c}_bin'] = df[c].apply(has_now)

    df['fin_score']      = df[[f'{c}_bin' for c in fin_cols if f'{c}_bin' in df.columns]].sum(axis=1)
    df['ins_score']      = df[[f'{c}_bin' for c in ins_cols if f'{c}_bin' in df.columns]].sum(axis=1)
    df['attitude_score'] = df[[f'{c}_bin' for c in att_cols if f'{c}_bin' in df.columns]].sum(axis=1)
    df['total_access']   = df['fin_score'] + df['ins_score']

    # Worried + cash flow = vulnerability flag
    def is_yes(v):
        return 1 if str(v).lower() == 'yes' else 0

    if 'attitude_worried_shutdown' in df.columns:
        df['worried_bin'] = df['attitude_worried_shutdown'].apply(is_yes)
    if 'current_problem_cash_flow' in df.columns:
        df['cashflow_problem_bin'] = df['current_problem_cash_flow'].apply(is_yes)
    if 'worried_bin' in df.columns and 'cashflow_problem_bin' in df.columns:
        df['vulnerability_flag'] = df['worried_bin'] + df['cashflow_problem_bin']

    return df

X      = engineer(X)
X_test = engineer(X_test)

# ══════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Ordinal encode categoricals (keeps NaN as NaN — LightGBM handles it)
oe = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=np.nan
)
X_cat      = oe.fit_transform(X[cat_cols])
X_test_cat = oe.transform(X_test[cat_cols])

# Numeric: keep NaN for LightGBM; impute for sklearn models
imp = SimpleImputer(strategy='median')
X_num      = imp.fit_transform(X[num_cols])
X_test_num = imp.transform(X_test[num_cols])

X_proc      = np.hstack([X_cat, X_num])
X_test_proc = np.hstack([X_test_cat, X_test_num])

# Encode target
le    = LabelEncoder()
y_enc = le.fit_transform(y)
classes = le.classes_
print("Classes:", classes)   # ['High', 'Low', 'Medium']

HIGH_IDX   = list(classes).index('High')
LOW_IDX    = list(classes).index('Low')
MEDIUM_IDX = list(classes).index('Medium')

# ══════════════════════════════════════════════════════════
# 4. BUILD MODELS
# ══════════════════════════════════════════════════════════
def make_lgb():
    if HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        return HistGradientBoostingClassifier(
            max_iter=800, learning_rate=0.03, max_depth=8,
            min_samples_leaf=10, l2_regularization=0.01,
            class_weight='balanced', random_state=42,
            early_stopping=False,
        )

rf = RandomForestClassifier(
    n_estimators=800, max_depth=None, min_samples_leaf=1,
    max_features='sqrt', class_weight='balanced_subsample',
    random_state=42, n_jobs=-1
)

et = ExtraTreesClassifier(
    n_estimators=800, max_depth=None, min_samples_leaf=1,
    max_features='sqrt', class_weight='balanced_subsample',
    random_state=43, n_jobs=-1
)

# Weights: LGB is strongest
WEIGHTS = {'lgb': 0.50, 'rf': 0.25, 'et': 0.25}

# ══════════════════════════════════════════════════════════
# 5. OOF TRAINING WITH OPTIONAL SMOTE
# ══════════════════════════════════════════════════════════
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_cls    = len(classes)
oof_prob = np.zeros((len(X_proc), n_cls))
tst_prob = np.zeros((len(X_test_proc), n_cls))
f1_folds = []

print("\n" + "="*60)
print("5-FOLD CV — Soft-Vote Ensemble (LGB + RF + ET)")
if HAS_SMOTE:
    print("SMOTE: ON  (oversampling High class in each fold)")
else:
    print("SMOTE: OFF (install imbalanced-learn to enable)")
print("="*60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_proc, y_enc), 1):
    X_tr, X_val = X_proc[tr_idx], X_proc[val_idx]
    y_tr, y_val = y_enc[tr_idx],  y_enc[val_idx]

    # Apply SMOTE only on training fold
    if HAS_SMOTE:
        sm = SMOTE(
            sampling_strategy={HIGH_IDX: int(sum(y_tr == LOW_IDX) * 0.15)},
            k_neighbors=5,
            random_state=42
        )
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    else:
        X_tr_sm, y_tr_sm = X_tr, y_tr

    fold_val_prob  = np.zeros((len(val_idx), n_cls))
    fold_test_prob = np.zeros((len(X_test_proc), n_cls))

    # LGB / HGB
    lgb_model = make_lgb()
    if HAS_LGB:
        lgb_model.fit(
            X_tr_sm, y_tr_sm,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
    else:
        lgb_model.fit(X_tr_sm, y_tr_sm)

    fold_val_prob  += WEIGHTS['lgb'] * lgb_model.predict_proba(X_val)
    fold_test_prob += WEIGHTS['lgb'] * lgb_model.predict_proba(X_test_proc)

    # Random Forest
    rf.fit(X_tr_sm, y_tr_sm)
    fold_val_prob  += WEIGHTS['rf'] * rf.predict_proba(X_val)
    fold_test_prob += WEIGHTS['rf'] * rf.predict_proba(X_test_proc)

    # Extra Trees
    et.fit(X_tr_sm, y_tr_sm)
    fold_val_prob  += WEIGHTS['et'] * et.predict_proba(X_val)
    fold_test_prob += WEIGHTS['et'] * et.predict_proba(X_test_proc)

    oof_prob[val_idx] = fold_val_prob
    tst_prob         += fold_test_prob / skf.n_splits

    preds_idx = fold_val_prob.argmax(axis=1)
    preds     = le.inverse_transform(preds_idx)
    y_val_str = le.inverse_transform(y_val)
    score     = f1_score(y_val_str, preds, average='weighted')
    f1_folds.append(score)
    high_mask_val = (y_val == HIGH_IDX)
    high_recall   = (preds_idx[high_mask_val] == HIGH_IDX).mean() if high_mask_val.sum() > 0 else 0
    print(f"  Fold {fold} | F1: {score:.4f} | High recall: {high_recall:.2f}")

print(f"\nFold scores: {[round(s,4) for s in f1_folds]}")
print(f"Std Dev:     {np.std(f1_folds):.4f}")
print(f"\n{'='*60}")
print(f"✅ Mean Weighted F1 (before threshold): {np.mean(f1_folds):.4f}")
print(f"{'='*60}")

# ══════════════════════════════════════════════════════════
# 6. THRESHOLD TUNING FOR HIGH CLASS
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("THRESHOLD TUNING — boosting High class recall")
print("="*60)

best_f1    = 0
best_thresh = 0.5

# Try different thresholds for the High class probability
for thresh in np.arange(0.10, 0.55, 0.01):
    preds_thresh = oof_prob.argmax(axis=1).copy()

    # Override: if P(High) > thresh, predict High
    high_mask = oof_prob[:, HIGH_IDX] >= thresh
    preds_thresh[high_mask] = HIGH_IDX

    preds_labels = le.inverse_transform(preds_thresh)
    score = f1_score(y, preds_labels, average='weighted')

    if score > best_f1:
        best_f1     = score
        best_thresh = thresh

print(f"Best High-class threshold: {best_thresh:.2f}")
print(f"Best OOF F1 after tuning:  {best_f1:.4f}")

# Apply best threshold to OOF for report
preds_oof = oof_prob.argmax(axis=1).copy()
preds_oof[oof_prob[:, HIGH_IDX] >= best_thresh] = HIGH_IDX
preds_oof_labels = le.inverse_transform(preds_oof)

print("\nFull OOF Classification Report (with threshold):")
print(classification_report(y, preds_oof_labels))

# ══════════════════════════════════════════════════════════
# 7. GENERATE FINAL SUBMISSION
# ══════════════════════════════════════════════════════════
final_preds = tst_prob.argmax(axis=1).copy()
final_preds[tst_prob[:, HIGH_IDX] >= best_thresh] = HIGH_IDX
final_labels = le.inverse_transform(final_preds)

submission = pd.DataFrame({
    "ID":     test["ID"],
    "Target": final_labels
})
submission.to_csv("submission_v3.csv", index=False)

print("\n" + "="*60)
print("✅ submission_v3.csv saved!")
print("Prediction distribution:")
print(submission['Target'].value_counts())
print("="*60)
