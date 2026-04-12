"""Strategy seeds for the tree search solver.

Each strategy provides a different initial approach hint that biases the first
solution node.  The tree search then iterates from that starting point.  Using
diverse seeds across parallel attempts is the structural equivalent of pass@k.

All strategies share a common foundation: understand the data FIRST before modeling.
"""

from __future__ import annotations

_EDA_COMMON = (
    "FIRST, always start by understanding the data:\n"
    "1. Read description.md to understand the competition goal and evaluation metric.\n"
    "2. Print column types, shapes, missing rates, and target distribution.\n"
    "3. Inspect sample_submission.csv to understand the expected output format.\n"
    "4. Identify key patterns: class imbalance, high-cardinality categoricals, "
    "splittable columns, boolean-like strings, numeric outliers.\n\n"
    "THEN, apply the following strategy:\n\n"
)

STRATEGIES: dict[str, str] = {
    # ── Robust fundamentals ─────────────────────────────────────────────
    "robust_fundamentals": _EDA_COMMON + (
        "Focus on ROBUST FUNDAMENTALS with careful preprocessing:\n"
        "- Handle missing values intelligently (median for numeric, mode for categorical)\n"
        "- Encode categoricals properly (OrdinalEncoder or TargetEncoder)\n"
        "- Engineer basic features: missing indicators, log transforms for skewed columns\n"
        "- Train an ensemble of gradient boosting models (XGBoost + LightGBM) with "
        "reasonable hyperparameters\n"
        "- Use 5-fold StratifiedKFold CV to evaluate\n"
        "- Blend predictions from XGBoost and LightGBM (simple average)\n"
        "- Prioritize a VALID, well-formatted submission above all else"
    ),

    # ── Aggressive feature engineering ──────────────────────────────────
    "feature_engineering": _EDA_COMMON + (
        "Focus on AGGRESSIVE FEATURE ENGINEERING:\n"
        "- Split composite columns (e.g., 'A/B/C' → 3 separate features)\n"
        "- Create interaction features between the most correlated column pairs\n"
        "- Create aggregation features (group-by statistics: mean, std, count)\n"
        "- Frequency encoding for categorical columns\n"
        "- Ratio features between related numeric columns\n"
        "- Boolean casting for True/False string columns\n"
        "- Log1p transforms for skewed numeric columns\n"
        "- After engineering, train a single strong model (LightGBM or CatBoost) "
        "and tune its key hyperparameters (learning_rate, max_depth, n_estimators, "
        "reg_alpha, reg_lambda) with a small grid search or manual tuning\n"
        "- Use 5-fold CV to evaluate and avoid overfitting"
    ),

    # ── Multi-model stacking ───────────────────────────────────────────
    "stacking_blend": _EDA_COMMON + (
        "Focus on MULTI-MODEL DIVERSITY and STACKING:\n"
        "- Apply solid preprocessing (imputation, encoding, scaling where needed)\n"
        "- Train multiple diverse models: LightGBM, XGBoost, CatBoost, "
        "RandomForest, HistGradientBoosting\n"
        "- For each model, generate out-of-fold (OOF) predictions using 5-fold CV\n"
        "- Build a simple stacking ensemble: use OOF predictions as features for "
        "a meta-learner (LogisticRegression for classification, Ridge for regression)\n"
        "- If stacking is too complex, fall back to weighted averaging of top 3 models\n"
        "- Optimize classification threshold on OOF predictions if applicable\n"
        "- Ensure submission format matches sample_submission.csv exactly"
    ),
}

DEFAULT_STRATEGY = "robust_fundamentals"

def get_strategy(name: str) -> str:
    return STRATEGIES.get(name, STRATEGIES[DEFAULT_STRATEGY])

def all_strategy_names() -> list[str]:
    return list(STRATEGIES.keys())
