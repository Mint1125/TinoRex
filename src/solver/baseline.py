"""Deterministic baseline script generator.

Produces a self-contained Python script that:
  - Auto-detects task type (binary/multi-class/regression)
  - Robust preprocessing (missing values, encoding)
  - Stacking ensemble (LightGBM + XGBoost + CatBoost)
  - 5-fold CV with proper OOF
  - Outputs CV_SCORE=<float> and submission.csv

No LLM involved — pure deterministic pipeline.
"""

from __future__ import annotations


BASELINE_SCRIPT = r'''
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error, f1_score, log_loss,
    mean_absolute_error, r2_score,
)
import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# =========================================================================
# 1. LOAD DATA
# =========================================================================
data_dir = Path("./home/data")
csv_files = sorted(data_dir.glob("*.csv"))
file_names = [f.name for f in csv_files]
print(f"Files: {file_names}")

# Find train, test, sample submission
train_path = None
test_path = None
sample_path = None

for f in csv_files:
    name_lower = f.name.lower()
    if "sample" in name_lower and "submission" in name_lower:
        sample_path = f
    elif "train" in name_lower:
        train_path = f
    elif "test" in name_lower:
        test_path = f

# Fallback: if no explicit train/test, use the two largest CSVs
if train_path is None:
    csvs_by_size = sorted(csv_files, key=lambda f: f.stat().st_size, reverse=True)
    csvs_no_sample = [f for f in csvs_by_size if "sample" not in f.name.lower()]
    if len(csvs_no_sample) >= 2:
        train_path = csvs_no_sample[0]
        test_path = csvs_no_sample[1]
    elif len(csvs_no_sample) == 1:
        train_path = csvs_no_sample[0]

assert train_path is not None, f"No training data found in {file_names}"
print(f"Train: {train_path.name}, Test: {test_path.name if test_path else 'None'}, Sample: {sample_path.name if sample_path else 'None'}")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path) if test_path else None
sample_df = pd.read_csv(sample_path) if sample_path else None

print(f"Train shape: {train_df.shape}")
if test_df is not None:
    print(f"Test shape: {test_df.shape}")

# =========================================================================
# 2. DETECT TASK TYPE & TARGET
# =========================================================================
if sample_df is not None:
    # Target columns = sample columns minus ID-like columns
    id_candidates = set()
    for col in sample_df.columns:
        if col.lower() in ("id", "index", "row_id", "uid"):
            id_candidates.add(col)
        elif test_df is not None and col in test_df.columns:
            if sample_df[col].nunique() == len(sample_df):
                id_candidates.add(col)

    target_cols = [c for c in sample_df.columns if c not in id_candidates]
    id_col = list(id_candidates)[0] if id_candidates else None
else:
    # Guess: last column of train is target
    target_cols = [train_df.columns[-1]]
    id_col = train_df.columns[0] if train_df.columns[0].lower() in ("id", "index") else None

print(f"ID column: {id_col}")
print(f"Target columns: {target_cols}")

# Determine target from train
train_target_cols = [c for c in target_cols if c in train_df.columns]
if not train_target_cols:
    train_only_cols = [c for c in train_df.columns if c not in (test_df.columns.tolist() if test_df is not None else [])]
    train_only_cols = [c for c in train_only_cols if c != id_col]
    if train_only_cols:
        train_target_cols = train_only_cols[:len(target_cols)]
    else:
        train_target_cols = target_cols

print(f"Train target columns: {train_target_cols}")

# Single vs multi-target
if len(train_target_cols) == 1:
    target_col = train_target_cols[0]
    y_train = train_df[target_col].copy()

    # Task type detection
    if y_train.dtype == "object" or y_train.dtype.name == "category":
        n_classes = y_train.nunique()
        task_type = "binary" if n_classes == 2 else "multiclass"
    elif y_train.dtype in (np.float64, np.float32):
        n_unique = y_train.nunique()
        if n_unique <= 20 and n_unique == int(n_unique):
            task_type = "binary" if n_unique == 2 else "multiclass"
        else:
            task_type = "regression"
    else:
        n_unique = y_train.nunique()
        task_type = "binary" if n_unique == 2 else ("multiclass" if n_unique <= 30 else "regression")
else:
    target_col = None
    task_type = "multi_target"

print(f"Task type: {task_type}")

# Encode string targets
label_map = None
if task_type in ("binary", "multiclass") and target_col is not None:
    if y_train.dtype == "object":
        classes = sorted(y_train.unique())
        label_map = {c: i for i, c in enumerate(classes)}
        y_train = y_train.map(label_map)
        print(f"Label encoding: {label_map}")

# =========================================================================
# 3. FEATURE ENGINEERING
# =========================================================================
drop_cols = set()
if id_col and id_col in train_df.columns:
    drop_cols.add(id_col)
if target_col and target_col in train_df.columns:
    drop_cols.add(target_col)
if task_type == "multi_target":
    drop_cols.update(train_target_cols)

feature_cols = [c for c in train_df.columns if c not in drop_cols]
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy() if test_df is not None else None

print(f"Features: {len(feature_cols)} columns")

# --- Preprocessing ---
num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Drop near-constant columns
for col in num_cols[:]:
    if X_train[col].nunique() <= 1:
        num_cols.remove(col)
        X_train = X_train.drop(columns=[col])
        if X_test is not None and col in X_test.columns:
            X_test = X_test.drop(columns=[col])

# Drop very high cardinality text columns (likely IDs)
for col in cat_cols[:]:
    if X_train[col].nunique() > 0.9 * len(X_train):
        cat_cols.remove(col)
        X_train = X_train.drop(columns=[col])
        if X_test is not None and col in X_test.columns:
            X_test = X_test.drop(columns=[col])

# Impute numeric
if num_cols:
    num_imputer = SimpleImputer(strategy="median")
    X_train.loc[:, num_cols] = num_imputer.fit_transform(X_train[num_cols])
    if X_test is not None:
        X_test.loc[:, num_cols] = num_imputer.transform(X_test[num_cols])

# Encode categoricals
if cat_cols:
    for col in cat_cols:
        X_train.loc[:, col] = X_train[col].fillna("__MISSING__").astype(str)
        if X_test is not None:
            X_test.loc[:, col] = X_test[col].fillna("__MISSING__").astype(str)

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train.loc[:, cat_cols] = oe.fit_transform(X_train[cat_cols])
    if X_test is not None:
        X_test.loc[:, cat_cols] = oe.transform(X_test[cat_cols])

# Convert all to float
X_train = X_train.astype(np.float32)
if X_test is not None:
    X_test = X_test.astype(np.float32)

print(f"After preprocessing: {X_train.shape}")

# =========================================================================
# 4. STACKING ENSEMBLE WITH 5-FOLD OOF
# =========================================================================
N_FOLDS = 5
SEED = 42

# Initialize prediction variables for all paths
final_pred = None
final_proba = None
final_preds = {}
cv_score = 0.0

if task_type in ("binary", "multiclass"):
    n_classes = int(y_train.nunique())
    is_binary = (n_classes == 2)
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Base models
    base_models = {
        "lgbm": lambda: lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=63,
            random_state=SEED, verbose=-1, n_jobs=-1,
        ),
        "xgb": lambda: xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0, n_jobs=-1,
            use_label_encoder=False, eval_metric="logloss",
        ),
    }
    if HAS_CATBOOST:
        base_models["catboost"] = lambda: CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=SEED, verbose=0, thread_count=-1,
        )

    # OOF predictions
    n_train = len(X_train)
    n_test = len(X_test) if X_test is not None else 0

    if is_binary:
        oof_preds = {name: np.zeros(n_train) for name in base_models}
        test_preds = {name: np.zeros(n_test) for name in base_models}
    else:
        oof_preds = {name: np.zeros((n_train, n_classes)) for name in base_models}
        test_preds = {name: np.zeros((n_test, n_classes)) for name in base_models}

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        for name, model_fn in base_models.items():
            model = model_fn()
            model.fit(X_tr, y_tr)

            if is_binary:
                val_pred = model.predict_proba(X_val)[:, 1]
                oof_preds[name][val_idx] = val_pred
                if X_test is not None:
                    test_preds[name] += model.predict_proba(X_test)[:, 1] / N_FOLDS
            else:
                val_pred = model.predict_proba(X_val)
                oof_preds[name][val_idx] = val_pred
                if X_test is not None:
                    test_preds[name] += model.predict_proba(X_test) / N_FOLDS

        print(f"  Fold {fold_idx+1}/{N_FOLDS} done")

    # Meta-learner: stack OOF predictions
    if is_binary:
        meta_X_train = np.column_stack([oof_preds[name] for name in base_models])
        meta_X_test = np.column_stack([test_preds[name] for name in base_models]) if X_test is not None else None
    else:
        meta_X_train = np.hstack([oof_preds[name] for name in base_models])
        meta_X_test = np.hstack([test_preds[name] for name in base_models]) if X_test is not None else None

    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta_model.fit(meta_X_train, y_train)

    # CV score
    if is_binary:
        oof_meta = meta_model.predict_proba(meta_X_train)[:, 1]
        try:
            cv_score = roc_auc_score(y_train, oof_meta)
            metric_name = "AUC"
        except Exception:
            cv_pred = meta_model.predict(meta_X_train)
            cv_score = accuracy_score(y_train, cv_pred)
            metric_name = "Accuracy"
    else:
        cv_pred = meta_model.predict(meta_X_train)
        cv_score = accuracy_score(y_train, cv_pred)
        metric_name = "Accuracy"

    print(f"Stacking CV {metric_name}: {cv_score:.6f}")
    print(f"CV_SCORE={cv_score}")

    # Final prediction
    if X_test is not None and meta_X_test is not None:
        final_pred = meta_model.predict(meta_X_test)
        if is_binary:
            final_proba = meta_model.predict_proba(meta_X_test)[:, 1]
        else:
            final_proba = meta_model.predict_proba(meta_X_test)

elif task_type == "regression":
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    base_models = {
        "lgbm": lambda: lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, num_leaves=63,
            random_state=SEED, verbose=-1, n_jobs=-1,
        ),
        "xgb": lambda: xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0, n_jobs=-1,
        ),
    }
    if HAS_CATBOOST:
        base_models["catboost"] = lambda: CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=SEED, verbose=0, thread_count=-1,
        )

    n_train = len(X_train)
    n_test = len(X_test) if X_test is not None else 0
    oof_preds = {name: np.zeros(n_train) for name in base_models}
    test_preds = {name: np.zeros(n_test) for name in base_models}

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        for name, model_fn in base_models.items():
            model = model_fn()
            model.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = model.predict(X_val)
            if X_test is not None:
                test_preds[name] += model.predict(X_test) / N_FOLDS

        print(f"  Fold {fold_idx+1}/{N_FOLDS} done")

    meta_X_train = np.column_stack([oof_preds[name] for name in base_models])
    meta_X_test = np.column_stack([test_preds[name] for name in base_models]) if X_test is not None else None

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X_train, y_train)

    oof_meta = meta_model.predict(meta_X_train)
    rmse = np.sqrt(mean_squared_error(y_train, oof_meta))
    cv_score = rmse
    print(f"Stacking CV RMSE: {rmse:.6f}")
    print(f"CV_SCORE={cv_score}")

    if X_test is not None and meta_X_test is not None:
        final_pred = meta_model.predict(meta_X_test)

elif task_type == "multi_target":
    y_trains = {col: train_df[col] for col in train_target_cols}
    cv_scores = []

    for tcol in train_target_cols:
        y_t = y_trains[tcol]
        is_reg = y_t.dtype in (np.float64, np.float32) or y_t.nunique() > 30

        if is_reg:
            model = lgb.LGBMRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                random_state=SEED, verbose=-1, n_jobs=-1,
            )
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        else:
            model = lgb.LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                random_state=SEED, verbose=-1, n_jobs=-1,
            )
            kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

        oof = np.zeros(len(X_train))
        test_pred_col = np.zeros(len(X_test)) if X_test is not None else None

        for tr_idx, val_idx in kf.split(X_train, y_t if not is_reg else None):
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_train.iloc[tr_idx], y_t.iloc[tr_idx])
            oof[val_idx] = model_fold.predict(X_train.iloc[val_idx])
            if X_test is not None and test_pred_col is not None:
                test_pred_col += model_fold.predict(X_test) / N_FOLDS

        if is_reg:
            score = np.sqrt(mean_squared_error(y_t, oof))
        else:
            score = accuracy_score(y_t, np.round(oof))
        cv_scores.append(score)
        if test_pred_col is not None:
            final_preds[tcol] = test_pred_col
        print(f"  Target '{tcol}': score={score:.6f}")

    cv_score = np.mean(cv_scores)
    print(f"CV_SCORE={cv_score}")

# =========================================================================
# 5. GENERATE SUBMISSION
# =========================================================================
submission = None

if sample_df is not None:
    submission = sample_df.copy()

    if task_type in ("binary", "multiclass") and final_pred is not None:
        for tcol in target_cols:
            if tcol in submission.columns:
                if task_type == "binary" and final_proba is not None:
                    # Check if sample expects probabilities or labels
                    sample_vals = sample_df[tcol].dropna().unique()
                    if set(sample_vals).issubset({0, 1}) or (sample_df[tcol].dtype == "object"):
                        pred_labels = final_pred.copy()
                        if label_map:
                            inv_map = {v: k for k, v in label_map.items()}
                            pred_labels = np.array([inv_map.get(int(p), p) for p in pred_labels])
                        submission[tcol] = pred_labels
                    else:
                        submission[tcol] = final_proba
                elif final_pred is not None:
                    pred_labels = final_pred.copy()
                    if label_map:
                        inv_map = {v: k for k, v in label_map.items()}
                        pred_labels = np.array([inv_map.get(int(p), p) for p in pred_labels])
                    submission[tcol] = pred_labels

    elif task_type == "regression" and final_pred is not None:
        for tcol in target_cols:
            if tcol in submission.columns:
                submission[tcol] = final_pred

    elif task_type == "multi_target" and final_preds:
        for tcol in target_cols:
            if tcol in submission.columns and tcol in final_preds:
                submission[tcol] = final_preds[tcol]

elif test_df is not None:
    # No sample: create submission from test predictions
    submission = pd.DataFrame()
    if id_col and id_col in test_df.columns:
        submission[id_col] = test_df[id_col]

    if task_type in ("binary", "multiclass") and final_pred is not None:
        submission[target_cols[0]] = final_pred
    elif task_type == "regression" and final_pred is not None:
        submission[target_cols[0]] = final_pred
    elif task_type == "multi_target" and final_preds:
        for tcol in target_cols:
            if tcol in final_preds:
                submission[tcol] = final_preds[tcol]

# Fallback: if no submission yet, use sample_submission as-is
if submission is None:
    if sample_df is not None:
        submission = sample_df.copy()
        print("WARNING: Using sample_submission as fallback (no test predictions)")
    else:
        # Last resort: create minimal submission
        submission = pd.DataFrame({"prediction": [0]})
        print("WARNING: Created minimal fallback submission")

# Fill any NaN safely
for col in submission.columns:
    if submission[col].isna().any():
        if submission[col].dtype == "object":
            col_mode = submission[col].mode()
            fill_val = col_mode.iloc[0] if len(col_mode) > 0 else "0"
            submission[col] = submission[col].fillna(fill_val)
        else:
            fill_val = submission[col].median()
            if pd.isna(fill_val):
                fill_val = 0
            submission[col] = submission[col].fillna(fill_val)

submission.to_csv("./submission.csv", index=False)
print(f"Submission saved: {submission.shape}")
print(f"Submission columns: {list(submission.columns)}")
print(f"Submission head:\n{submission.head()}")
'''


def get_baseline_script() -> str:
    """Return the deterministic baseline script."""
    return BASELINE_SCRIPT
