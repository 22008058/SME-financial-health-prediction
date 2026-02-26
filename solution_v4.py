"""
data.org Financial Health Prediction Challenge
Solution v4 — All Improvements Applied
========================================
Changes from v3:
  1. Metric changed to MACRO F1 (matches competition)
  2. Multi-threshold tuning (High + Medium)
  3. Enhanced feature engineering (4 FHI dimensions)
  4. XGBoost + CatBoost added to ensemble
  5. Better SMOTE strategy (boost High + Medium)
  6. Stratified CV by country
  7. Class index debug verification

Requirements:
    pip install lightgbm xgboost catboost imbalanced-learn scikit-learn pandas numpy
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

# ── Optional heavy libs ───────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True; print("✅ LightGBM found")
except ImportError:
    HAS_LGB = False; print("⚠️  LightGBM not found — using HistGradientBoosting")
    from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import xgboost as xgb
    HAS_XGB = True; print("✅ XGBoost found")
except ImportError:
    HAS_XGB = False; print("⚠️  XGBoost not found — skipping")

try:
    import catboost as cb
    HAS_CB = True; print("✅ CatBoost found")
except ImportError:
    HAS_CB = False; print("⚠️  CatBoost not found — skipping")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True; print("✅ SMOTE found")
except ImportError:
    HAS_SMOTE = False; print("⚠️  imbalanced-learn not found — SMOTE skipped")

np.random.seed(42)

# ══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════
train = pd.read_csv(r"C:\Users\pc1\OneDrive - University of Venda\Desktop\SME_Project\Train.csv")
test  = pd.read_csv(r"C:\Users\pc1\OneDrive - University of Venda\Desktop\SME_Project\Test.csv")

print(f"\nTrain: {train.shape}  |  Test: {test.shape}")
print("Target distribution:\n", train['Target'].value_counts(), "\n")

X      = train.drop(columns=["Target", "ID"])
y      = train["Target"]
X_test = test.drop(columns=["ID"])

# ══════════════════════════════════════════════════════════
# 2. ENHANCED FEATURE ENGINEERING (all 4 FHI dimensions)
# ══════════════════════════════════════════════════════════
def engineer(df):
    df = df.copy()

    # ── Core financial ratios ─────────────────────────────
    df['profit']        = df['business_turnover'] - df['business_expenses']
    df['expense_ratio'] = df['business_expenses'] / (df['business_turnover'] + 1)
    df['income_to_turn']= df['personal_income']   / (df['business_turnover'] + 1)
    df['profit_margin'] = df['profit']             / (df['business_turnover'] + 1)
    df['income_stability'] = df['profit_margin'].clip(-1, 1)

    # Log transforms (handle skewness)
    for col in ['business_turnover', 'personal_income', 'business_expenses']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0).fillna(0))

    # Business age
    df['biz_age_months_total'] = (
        df['business_age_years'].fillna(0) * 12 +
        df['business_age_months'].fillna(0)
    )
    df['biz_age_group'] = pd.cut(
        df['biz_age_months_total'],
        bins=[0, 12, 36, 120, 9999],
        labels=[0, 1, 2, 3]        # new / young / mature / established
    ).astype(float)

    # ── Binary helpers ────────────────────────────────────
    def has_now(val):
        if pd.isna(val): return 0
        s = str(val).lower()
        return 1 if ('have now' in s or s == 'yes') else 0

    def is_yes(val):
        return 1 if str(val).lower() == 'yes' else 0

    fin_cols = ['has_mobile_money', 'has_credit_card', 'has_loan_account',
                'has_internet_banking', 'has_debit_card']
    ins_cols = ['has_insurance', 'motor_vehicle_insurance',
                'medical_insurance', 'funeral_insurance']
    att_pos  = ['attitude_stable_business_environment',
                'attitude_satisfied_with_achievement',
                'attitude_more_successful_next_year']

    for c in fin_cols + ins_cols + att_pos:
        if c in df.columns:
            df[f'{c}_bin'] = df[c].apply(has_now)

    # ── FHI Dimension 1: Savings & Assets ─────────────────
    df['fin_score']    = df[[f'{c}_bin' for c in fin_cols if f'{c}_bin' in df.columns]].sum(axis=1)
    df['savings_proxy']= (df['personal_income'] - df['business_expenses']).clip(lower=0)
    df['log_savings']  = np.log1p(df['savings_proxy'].fillna(0))

    # ── FHI Dimension 2: Debt & Repayment ─────────────────
    df['ins_score']    = df[[f'{c}_bin' for c in ins_cols if f'{c}_bin' in df.columns]].sum(axis=1)
    has_loan           = df['has_loan_account_bin'] if 'has_loan_account_bin' in df.columns else 0
    df['debt_burden']  = has_loan * df['expense_ratio']

    # ── FHI Dimension 3: Resilience ───────────────────────
    if 'attitude_worried_shutdown' in df.columns:
        df['worried_bin'] = df['attitude_worried_shutdown'].apply(is_yes)
    if 'current_problem_cash_flow' in df.columns:
        df['cashflow_problem_bin'] = df['current_problem_cash_flow'].apply(is_yes)

    res_cols = [c for c in ['worried_bin', 'cashflow_problem_bin'] if c in df.columns]
    df['vulnerability_flag']   = df[res_cols].sum(axis=1) if res_cols else 0
    df['resilience_risk_score']= df['vulnerability_flag']

    # ── FHI Dimension 4: Access to Credit ─────────────────
    has_ins            = df['has_insurance_bin'] if 'has_insurance_bin' in df.columns else 0
    df['credit_diversity'] = df['fin_score'] * (1 + has_ins)
    df['total_access'] = df['fin_score'] + df['ins_score']

    # ── Attitude score ─────────────────────────────────────
    df['attitude_score']= df[[f'{c}_bin' for c in att_pos if f'{c}_bin' in df.columns]].sum(axis=1)

    # ── Risk flags ─────────────────────────────────────────
    df['high_risk_flag'] = (
        (df['expense_ratio'] > 0.9).astype(int) +
        (df['profit_margin'] < 0).astype(int) +
        df['vulnerability_flag']
    ).clip(0, 3)

    # ── Interaction features ───────────────────────────────
    df['fin_x_attitude']   = df['fin_score']  * df['attitude_score']
    df['access_x_profit']  = df['total_access'] * df['profit_margin'].clip(-1, 1)
    df['age_x_fin']        = df['biz_age_months_total'] * df['fin_score']

    return df

X      = engineer(X)
X_test = engineer(X_test)

print(f"Features after engineering: {X.shape[1]}")

# ══════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

oe = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=np.nan
)
X_cat      = oe.fit_transform(X[cat_cols])
X_test_cat = oe.transform(X_test[cat_cols])

imp = SimpleImputer(strategy='median')
X_num      = imp.fit_transform(X[num_cols])
X_test_num = imp.transform(X_test[num_cols])

X_proc      = np.hstack([X_cat, X_num])
X_test_proc = np.hstack([X_test_cat, X_test_num])

# Fix: impute ALL NaNs so SMOTE works (categorical NaNs from OrdinalEncoder)
imp_all     = SimpleImputer(strategy='most_frequent')
X_proc      = imp_all.fit_transform(X_proc)
X_test_proc = imp_all.transform(X_test_proc)
print(f"NaNs remaining after imputation: {np.isnan(X_proc).sum()}")

# Encode target
le      = LabelEncoder()
y_enc   = le.fit_transform(y)
classes = le.classes_

# ── DEBUG: verify class indices ───────────────────────────
print("\nClass index verification:")
print(dict(zip(classes, range(len(classes)))))
HIGH_IDX   = list(classes).index('High')
LOW_IDX    = list(classes).index('Low')
MEDIUM_IDX = list(classes).index('Medium')
print(f"HIGH_IDX={HIGH_IDX}  LOW_IDX={LOW_IDX}  MEDIUM_IDX={MEDIUM_IDX}")

# ══════════════════════════════════════════════════════════
# 4. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════
def make_lgb():
    if HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, num_leaves=63,
            max_depth=-1, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=800, learning_rate=0.03, max_depth=8,
            min_samples_leaf=10, l2_regularization=0.01,
            class_weight='balanced', random_state=42, early_stopping=False,
        )

def make_xgb():
    if HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.03, max_depth=6,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='mlogloss',
            verbosity=0, n_jobs=-1,
        )
    return None

def make_catboost():
    if HAS_CB:
        return cb.CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, auto_class_weights='Balanced',
            random_seed=42, verbose=False,
        )
    return None

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

# Dynamic weights based on available models
def build_model_list():
    models = []
    if HAS_LGB or True:      models.append(('lgb', make_lgb(), 0.35))
    if HAS_XGB:              models.append(('xgb', make_xgb(), 0.25))
    if HAS_CB:               models.append(('cb',  make_catboost(), 0.20))
    models.append(('rf', rf, 0.0))
    models.append(('et', et, 0.0))

    # Redistribute weights if XGB/CB missing
    base_w  = sum(w for _, _, w in models if w > 0)
    tree_w  = 1.0 - base_w
    n_tree  = sum(1 for n, _, _ in models if n in ['rf', 'et'])
    models  = [(n, m, (tree_w / n_tree) if n in ['rf', 'et'] else w)
               for n, m, w in models]
    total_w = sum(w for _, _, w in models)
    models  = [(n, m, w / total_w) for n, m, w in models]
    print("\nEnsemble weights:")
    for n, _, w in models:
        print(f"  {n}: {w:.2f}")
    return models

# ══════════════════════════════════════════════════════════
# 5. OOF TRAINING
# ══════════════════════════════════════════════════════════

# Country-aware stratification
country_enc = OrdinalEncoder().fit_transform(train[['country']])
strat_key   = y_enc * 10 + country_enc.ravel().astype(int)

skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_cls    = len(classes)
oof_prob = np.zeros((len(X_proc), n_cls))
tst_prob = np.zeros((len(X_test_proc), n_cls))
f1_folds = []

model_list = build_model_list()

print("\n" + "="*60)
print("5-FOLD CV  |  Metric: MACRO F1  |  Country-stratified")
print("="*60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_proc, strat_key), 1):
    X_tr, X_val = X_proc[tr_idx], X_proc[val_idx]
    y_tr, y_val = y_enc[tr_idx],  y_enc[val_idx]

    # SMOTE: boost High to 50% of Low, Medium to 70% of Low
    if HAS_SMOTE:
        n_low    = (y_tr == LOW_IDX).sum()
        strategy = {
            HIGH_IDX:   int(n_low * 0.50),
            MEDIUM_IDX: max(int(n_low * 0.70), (y_tr == MEDIUM_IDX).sum()),
        }
        strategy = {k: v for k, v in strategy.items() if v > (y_tr == k).sum()}
        if strategy:
            sm = SMOTE(sampling_strategy=strategy, k_neighbors=5, random_state=42)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    fold_val  = np.zeros((len(val_idx), n_cls))
    fold_test = np.zeros((len(X_test_proc), n_cls))

    for name, model, weight in model_list:
        if name == 'lgb' and HAS_LGB:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        elif name == 'xgb' and HAS_XGB:
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        elif name == 'cb' and HAS_CB:
            model.fit(X_tr, y_tr,
                      eval_set=(X_val, y_val),
                      early_stopping_rounds=50)
        else:
            model.fit(X_tr, y_tr)

        fold_val  += weight * model.predict_proba(X_val)
        fold_test += weight * model.predict_proba(X_test_proc)

    oof_prob[val_idx] = fold_val
    tst_prob         += fold_test / skf.n_splits

    preds_idx = fold_val.argmax(axis=1)
    preds_str = le.inverse_transform(preds_idx)
    y_val_str = le.inverse_transform(y_val)
    score     = f1_score(y_val_str, preds_str, average='macro')   # ← MACRO
    f1_folds.append(score)

    high_actual = (y_val == HIGH_IDX)
    high_recall = (preds_idx[high_actual] == HIGH_IDX).mean() if high_actual.sum() > 0 else 0
    print(f"  Fold {fold} | Macro F1: {score:.4f} | High recall: {high_recall:.2f}")

print(f"\nFold scores: {[round(s,4) for s in f1_folds]}")
print(f"Std Dev:     {np.std(f1_folds):.4f}")
print(f"\n{'='*60}")
print(f"✅ Mean MACRO F1 (before threshold): {np.mean(f1_folds):.4f}")
print(f"{'='*60}")

# ══════════════════════════════════════════════════════════
# 6. MULTI-THRESHOLD TUNING (High + Medium)  →  MACRO F1
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MULTI-THRESHOLD TUNING — Macro F1")
print("="*60)

best_f1         = 0
best_high_thresh = 0.50
best_med_thresh  = 0.50

for high_thresh in np.arange(0.10, 0.61, 0.02):
    for med_thresh in np.arange(0.15, 0.61, 0.02):
        preds = np.full(len(oof_prob), LOW_IDX)

        high_mask          = oof_prob[:, HIGH_IDX]   >= high_thresh
        preds[high_mask]   = HIGH_IDX

        med_mask           = (~high_mask) & (oof_prob[:, MEDIUM_IDX] >= med_thresh)
        preds[med_mask]    = MEDIUM_IDX

        score = f1_score(y_enc, preds, average='macro')   # ← MACRO
        if score > best_f1:
            best_f1          = score
            best_high_thresh = high_thresh
            best_med_thresh  = med_thresh

print(f"Best High threshold:  {best_high_thresh:.2f}")
print(f"Best Medium threshold:{best_med_thresh:.2f}")
print(f"Best OOF Macro F1:    {best_f1:.4f}")

# Apply to OOF for report
preds_oof                                          = np.full(len(oof_prob), LOW_IDX)
preds_oof[oof_prob[:, HIGH_IDX]   >= best_high_thresh] = HIGH_IDX
med_mask = (preds_oof != HIGH_IDX) & (oof_prob[:, MEDIUM_IDX] >= best_med_thresh)
preds_oof[med_mask]                                = MEDIUM_IDX
preds_oof_str = le.inverse_transform(preds_oof)

print("\nFull OOF Classification Report (macro, with threshold):")
print(classification_report(y, preds_oof_str, target_names=classes))

# ══════════════════════════════════════════════════════════
# 7. GENERATE FINAL SUBMISSION
# ══════════════════════════════════════════════════════════
final_preds                                              = np.full(len(tst_prob), LOW_IDX)
final_preds[tst_prob[:, HIGH_IDX]   >= best_high_thresh] = HIGH_IDX
med_mask = (final_preds != HIGH_IDX) & (tst_prob[:, MEDIUM_IDX] >= best_med_thresh)
final_preds[med_mask]                                    = MEDIUM_IDX
final_labels = le.inverse_transform(final_preds)

submission = pd.DataFrame({"ID": test["ID"], "Target": final_labels})
submission.to_csv(r"C:\Users\pc1\OneDrive - University of Venda\Desktop\SME_Project\submission_v4.csv", index=False)

print("\n" + "="*60)
print("✅ submission_v4.csv saved!")
print("Prediction distribution:")
print(submission['Target'].value_counts())
print("="*60)