"""Stage-wise debate prompt templates for AgentX v9.

Each stage has a system prompt and per-module user prompt builders.
All prompts are fully generic — zero dataset-specific knowledge.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared system prompt (used across all stages)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert ML engineer competing in a Kaggle-style competition.

You write COMPLETE, SELF-CONTAINED Python scripts. Every script must:
- Include all imports at the top.
- Be runnable in a fresh Python process with no prior state.
- Use only libraries available in the environment: pandas, numpy, sklearn, \
lightgbm, xgboost, catboost, optuna, scipy, pyarrow.

CRITICAL RULES:
- Data files are in ./home/data/ (train.csv, test.csv, sample_submission.csv, etc.)
- Save all output files to the CURRENT WORKING DIRECTORY (./) — NOT inside ./home/data/
- NEVER hardcode predictions, labels, or column names from memory.
- Always discover the data schema by reading files at runtime.
- Handle errors gracefully with try/except and print diagnostics.
- Keep stdout concise — only print what is needed.
- Always wrap main logic in try/except to ensure partial output on failure.
- Use print() for all output markers (CV_SCORE=, MODEL_RESULT=, etc.)
"""

# ---------------------------------------------------------------------------
# Output markers (parsed by solver agent)
# ---------------------------------------------------------------------------

MARKERS = {
    "eda": ("EDA_REPORT_START", "EDA_REPORT_END"),
    "features": ("CV_SCORE", None),
    "model_selection": ("MODEL_RESULT", None),
    "optuna": ("BEST_PARAMS", "BEST_SCORE"),
    "stacking": ("CV_SCORE", None),
    "threshold": ("CV_SCORE", "THRESHOLD"),
}

# ---------------------------------------------------------------------------
# Stage 1: EDA Profiling
# ---------------------------------------------------------------------------

EDA_SYSTEM = SYSTEM_PROMPT + """
Your task is exploratory data analysis. Write a script that reads the data \
and prints a JSON report wrapped between EDA_REPORT_START and EDA_REPORT_END markers.

Output format (print these exact markers):
print("EDA_REPORT_START")
print(json.dumps(report_dict))
print("EDA_REPORT_END")
"""

EDA_MODULES = {
    "A": {
        "focus": "statistical profiling",
        "instructions": """\
Focus on STATISTICAL PROFILING:
- Dataset shape (rows, columns) for train and test
- Target column identification (from sample_submission.csv columns[1])
- ID column identification (from sample_submission.csv columns[0])
- Task type detection (binary classification, multiclass, regression)
- For each column: dtype, missing rate, unique count
- Numeric columns: mean, std, min, max, median, skewness
- Target distribution (value_counts for classification)
- Pearson correlations between numeric features and target (top 10)

Print the report as a JSON dict with keys: task_type, target_col, id_col, n_train, \
n_test, n_features, columns (list of dicts), target_distribution, top_correlations.
""",
    },
    "B": {
        "focus": "data quality analysis",
        "instructions": """\
Focus on DATA QUALITY:
- Identify columns with >50% missing values
- Detect potential duplicate rows
- Find constant or near-constant columns (>99% same value)
- Detect potential data leakage (features too correlated with target, r>0.95)
- Identify high-cardinality categorical columns (>100 unique values)
- Check for mixed types in columns
- Check if train/test distributions differ significantly

Print the report as a JSON dict with keys: quality_issues (list of dicts with \
column, issue, details), leakage_suspects, high_cardinality_cols, \
train_test_drift (list of columns with distribution shift).
""",
    },
    "C": {
        "focus": "domain understanding and feature opportunities",
        "instructions": """\
Focus on DOMAIN UNDERSTANDING and FEATURE ENGINEERING OPPORTUNITIES:
- Read description.md to understand the competition goal and evaluation metric
- Infer semantic meaning of columns from names and values
- Identify columns that could be split (e.g., delimited strings, composite IDs)
- Identify columns that could be grouped
- Suggest interaction features (column pairs with non-linear relationships)
- Suggest aggregation features (group-by statistics)
- Identify boolean-like columns stored as strings
- Detect datetime-like columns

Print the report as a JSON dict with keys: competition_goal, eval_metric (if found), \
splittable_cols, groupable_cols, interaction_candidates (list of column pairs), \
boolean_cols, datetime_cols, encoding_suggestions.
""",
    },
}

EDA_SYNTHESIS_PROMPT = """\
You are given 3 EDA reports from independent analysts examining the same dataset.
Merge them into ONE comprehensive data understanding document.

Rules:
- Include ALL unique findings from all 3 reports.
- Resolve contradictions by preferring the more specific/detailed finding.
- Produce a single JSON object.
- Keep total output under 3000 tokens.

Required JSON keys:
- task_type: "binary_classification" | "multiclass_classification" | "regression"
- target_col: string
- id_col: string
- eval_metric: string or null
- n_train: int
- n_test: int
- columns: list of {{name, dtype, missing_rate, cardinality, role, notes}}
  where role is one of: "id", "target", "numeric", "categorical", "boolean", "datetime", "text", "drop"
- target_distribution: dict
- top_correlations: list of {{feature, correlation}}
- quality_issues: list of {{column, issue}}
- feature_opportunities: list of {{description, columns_involved, type}}
  where type is: "interaction", "aggregation", "split", "encoding", "datetime", "boolean_cast"

Report A:
{report_a}

Report B:
{report_b}

Report C:
{report_c}

Output ONLY the merged JSON (no markdown fences, no explanation):
"""

# ---------------------------------------------------------------------------
# Stage 2: Feature Engineering
# ---------------------------------------------------------------------------

FEATURES_SYSTEM = SYSTEM_PROMPT + """
Your task is feature engineering. Write a COMPLETE Python script that:
1. Reads train.csv and test.csv from ./home/data/
2. Reads sample_submission.csv to identify the ID column (columns[0]) and target column (columns[1])
3. Applies feature engineering informed by the EDA report
4. Runs a quick LightGBM 5-fold CV to validate feature quality
5. Prints exactly one line: CV_SCORE=<float> (e.g., CV_SCORE=0.8123)
6. Saves to CURRENT DIRECTORY (not home/data/):
   - train_features.parquet (features only, no target, no ID)
   - test_features.parquet (features only, no ID)
   - target.npy (numpy array of target values)
   - test_ids.npy (numpy array of test ID values)
   - feature_names.json (list of feature column names)

IMPORTANT:
- The target and ID columns must NOT be in the feature parquets.
- Use LabelEncoder or OrdinalEncoder for categorical columns before saving.
- Make sure train and test have the SAME columns in the SAME order.
- Wrap everything in try/except and print errors.

Example output line:
CV_SCORE=0.8123
"""

FEATURES_MODULES = {
    "A": {
        "focus": "interaction and polynomial features",
        "instructions": """\
Focus on INTERACTION and POLYNOMIAL features:
- Create pairwise interaction features for the top correlated numeric columns
- Create ratio features between related numeric columns
- Apply log1p transforms to skewed numeric columns
- Basic preprocessing: impute missing values, encode categoricals
- Use the EDA report to guide which interactions are most promising
""",
    },
    "B": {
        "focus": "aggregation and groupby features",
        "instructions": """\
Focus on AGGREGATION and GROUP-BY features:
- If there are groupable columns (from EDA), compute group statistics (mean, std, count)
- If there are splittable columns, split them and create derived features
- Create frequency encoding for categorical columns
- Basic preprocessing: impute missing values, encode categoricals
- Use the EDA report to identify the best grouping keys
""",
    },
    "C": {
        "focus": "encoding strategies and missing value handling",
        "instructions": """\
Focus on ENCODING and MISSING VALUE strategies:
- Apply ordinal encoding for low-cardinality categoricals
- Create missing value indicator columns for columns with >5% missing
- Apply intelligent imputation (median for numeric, mode for categorical)
- Cast boolean-like string columns to int
- Drop columns flagged as "drop" in the EDA report
- Basic preprocessing for all other columns
""",
    },
}

# ---------------------------------------------------------------------------
# Stage 3: Model Selection
# ---------------------------------------------------------------------------

MODEL_SYSTEM = SYSTEM_PROMPT + """
Your task is model selection. Write a COMPLETE Python script that:
1. Loads train_features.parquet and target.npy from the CURRENT DIRECTORY (./)
2. Trains multiple models with default/reasonable hyperparameters using 5-fold CV
3. For EACH model tested, prints exactly: MODEL_RESULT={"name":"ModelName", "cv_score":0.1234, "train_time_s":5.6}
4. Prints the best: CV_SCORE=<float>

Use StratifiedKFold for classification, KFold for regression. 5 folds.
Import and handle errors for each model independently — if one model fails, continue with others.

Example output:
MODEL_RESULT={"name":"LGBMClassifier", "cv_score":0.8123, "train_time_s":3.2}
MODEL_RESULT={"name":"XGBClassifier", "cv_score":0.8056, "train_time_s":5.1}
CV_SCORE=0.8123
"""

MODEL_MODULES = {
    "A": {
        "focus": "gradient boosting models",
        "instructions": """\
Test GRADIENT BOOSTING models:
- LightGBM (LGBMClassifier/LGBMRegressor) with verbose=-1
- XGBoost (XGBClassifier/XGBRegressor) with verbosity=0
- CatBoost (CatBoostClassifier/CatBoostRegressor) with verbose=0
- Print MODEL_RESULT JSON for each. Handle import/fit errors with try/except.
""",
    },
    "B": {
        "focus": "linear and SVM models",
        "instructions": """\
Test LINEAR models (with StandardScaler in a Pipeline):
- LogisticRegression / Ridge
- SVM (LinearSVC with dual='auto' or SGDClassifier)
- ElasticNet (regression) or SGDClassifier (classification)
- Print MODEL_RESULT JSON for each. Handle import/fit errors with try/except.
Note: Scale features first with StandardScaler.
""",
    },
    "C": {
        "focus": "ensemble and neural models",
        "instructions": """\
Test ENSEMBLE and TREE models:
- RandomForest (RandomForestClassifier/RandomForestRegressor)
- ExtraTrees (ExtraTreesClassifier/ExtraTreesRegressor)
- HistGradientBoosting (HistGradientBoostingClassifier/HistGradientBoostingRegressor)
- Print MODEL_RESULT JSON for each. Handle import/fit errors with try/except.
""",
    },
}

MODEL_SYNTHESIS_PROMPT = """\
You are selecting the top 3 models for a stacking ensemble.

Model results from 3 independent tests:
{all_model_results}

Task type: {task_type}

Select exactly 3 models for the ensemble. Priorities:
1. Highest CV scores
2. Model diversity — if the top 3 are all from the same family, \
replace the weakest with the best model from a different family.
3. Reasonable training time

Output a JSON list of 3 objects: [{{"name": "...", "cv_score": ..., "reason": "..."}}]
Output ONLY the JSON (no markdown fences, no explanation):
"""

# ---------------------------------------------------------------------------
# Stage 4: Optuna Hyperparameter Tuning
# ---------------------------------------------------------------------------

OPTUNA_SYSTEM = SYSTEM_PROMPT + """
Your task is hyperparameter tuning with Optuna. Write a COMPLETE Python script that:
1. Loads train_features.parquet and target.npy from the CURRENT DIRECTORY (./)
2. Defines an Optuna objective function for the specified model
3. You DECIDE the search space based on your expertise with this model family
4. Runs optuna.create_study().optimize(objective, n_trials=N)
5. Prints exactly: BEST_PARAMS={"param1": value1, ...}
6. Prints exactly: BEST_SCORE=<float>
7. Saves best_params_{model_name}.json to the CURRENT DIRECTORY

Use StratifiedKFold for classification, KFold for regression. 5 folds.
Set optuna.logging.set_verbosity(optuna.logging.WARNING) to suppress logs.

IMPORTANT: Use json.dumps() for BEST_PARAMS to ensure valid JSON output.
"""

OPTUNA_MODULE_TEMPLATE = """\
Tune the model: {model_name}
Current best CV score with defaults: {current_cv}
Dataset: {n_rows} rows, {n_features} features, task={task_type}

Design an appropriate search space for {model_name}. Consider:
- Dataset size when setting ranges (smaller dataset → lower complexity)
- Include regularization parameters
- n_trials = {n_trials}
"""

# ---------------------------------------------------------------------------
# Stage 5: Stacking Ensemble
# ---------------------------------------------------------------------------

STACKING_SYSTEM = SYSTEM_PROMPT + """
Your task is building a stacking ensemble. Write a COMPLETE Python script that:
1. Loads train_features.parquet, test_features.parquet, and target.npy from CURRENT DIRECTORY (./)
2. Loads test_ids.npy for the submission ID column
3. Reads sample_submission.csv from ./home/data/ to get column names and output format
4. Loads tuned parameters from best_params_*.json files in CURRENT DIRECTORY (use glob)
5. Level 1: For each base model, generate OOF predictions on train using 5-fold CV, and full predictions on test
6. Level 2: Train a meta-learner on the OOF predictions
7. Prints exactly: CV_SCORE=<float> (the full stacking pipeline CV score)
8. Saves: submission.csv (matching sample_submission.csv format exactly)
9. Also saves: oof_predictions.npy, test_predictions.npy

IMPORTANT:
- Use glob.glob("best_params_*.json") to find available param files
- If a param file is missing for a model, use that model with default params
- For binary classification, use predict_proba[:, 1] for OOF probabilities
- submission.csv must have the same columns as sample_submission.csv
- Use 5-fold StratifiedKFold for classification, KFold for regression
"""

STACKING_MODULES = {
    "A": {
        "focus": "LogisticRegression/Ridge meta-learner",
        "instructions": """\
Meta-learner strategy: LogisticRegression (classification) or Ridge (regression).
- OOF matrix: shape (n_train, n_base_models) with predicted probabilities or values.
- Meta-learner trained on OOF matrix → predict on test predictions matrix.
""",
    },
    "B": {
        "focus": "LightGBM meta-learner",
        "instructions": """\
Meta-learner strategy: LightGBM with low complexity (max_depth=3, n_estimators=100, verbose=-1).
- OOF matrix: shape (n_train, n_base_models) with predicted probabilities or values.
- This can capture non-linear interactions between base model predictions.
""",
    },
    "C": {
        "focus": "simple averaging",
        "instructions": """\
Meta-learner strategy: Simple averaging (no trained meta-learner).
- Average the OOF/test predictions from all base models.
- For classification: average probabilities, then threshold at 0.5.
- This is robust and avoids overfitting the meta-learner.
""",
    },
}

# ---------------------------------------------------------------------------
# Stage 6: Threshold Optimization
# ---------------------------------------------------------------------------

THRESHOLD_SYSTEM = SYSTEM_PROMPT + """
Your task is threshold optimization for classification. Write a COMPLETE Python script that:
1. Loads oof_predictions.npy and target.npy from CURRENT DIRECTORY
2. Loads test_predictions.npy and test_ids.npy from CURRENT DIRECTORY
3. Reads sample_submission.csv from ./home/data/ for output format
4. Searches for the optimal classification threshold
5. Prints exactly: CV_SCORE=<float> (score with optimized threshold on OOF)
6. Prints exactly: THRESHOLD=<float>
7. If the optimized score is better, saves a new submission.csv

IMPORTANT: Match the exact format of sample_submission.csv (same column names, same value format).
"""

THRESHOLD_MODULES = {
    "A": {
        "focus": "grid search",
        "instructions": """\
Strategy: GRID SEARCH over thresholds.
- Search range: numpy.arange(0.30, 0.70, 0.01)
- Evaluate each threshold on OOF predictions vs true labels using accuracy
- Print the best threshold and its score
""",
    },
    "B": {
        "focus": "scipy optimization",
        "instructions": """\
Strategy: SCIPY OPTIMIZATION.
- Use scipy.optimize.minimize_scalar to find the optimal threshold
- Bounds: (0.30, 0.70), method='bounded'
- Objective: negative accuracy on OOF
""",
    },
    "C": {
        "focus": "fine grid search",
        "instructions": """\
Strategy: TWO-PHASE GRID SEARCH.
- Phase 1: Coarse search numpy.arange(0.20, 0.80, 0.05)
- Phase 2: Fine search around best ±0.05 with step 0.005
- This finds a more precise optimum than a single grid
""",
    },
}

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

STAGE_CONFIGS = {
    "eda": {"system": EDA_SYSTEM, "modules": EDA_MODULES},
    "features": {"system": FEATURES_SYSTEM, "modules": FEATURES_MODULES},
    "model_selection": {"system": MODEL_SYSTEM, "modules": MODEL_MODULES},
    "optuna": {"system": OPTUNA_SYSTEM, "modules": None},  # dynamic
    "stacking": {"system": STACKING_SYSTEM, "modules": STACKING_MODULES},
    "threshold": {"system": THRESHOLD_SYSTEM, "modules": THRESHOLD_MODULES},
}


def build_user_prompt(
    *,
    stage: str,
    module: str,
    context: dict,
    description: str,
    file_listing: str,
    extra: dict | None = None,
) -> str:
    """Build the user prompt for a given stage + module."""
    parts: list[str] = []

    # Competition description
    parts.append(f"Competition description:\n{description}\n")
    parts.append(f"Files available:\n{file_listing}\n")

    # Accumulated context (summaries from prior stages)
    if context.get("eda_report"):
        parts.append(f"EDA Report:\n{context['eda_report']}\n")
    if context.get("feature_context"):
        parts.append(f"Feature Engineering Context:\n{context['feature_context']}\n")
    if context.get("model_selection_context"):
        parts.append(f"Model Selection Context:\n{context['model_selection_context']}\n")
    if context.get("tuning_context"):
        parts.append(f"Tuning Context:\n{context['tuning_context']}\n")
    if context.get("stacking_context"):
        parts.append(f"Stacking Context:\n{context['stacking_context']}\n")

    # Stage-specific module instructions
    if stage == "optuna" and extra:
        parts.append(OPTUNA_MODULE_TEMPLATE.format(**extra))
    else:
        cfg = STAGE_CONFIGS[stage]
        if cfg["modules"] and module in cfg["modules"]:
            mod = cfg["modules"][module]
            parts.append(f"Your focus: {mod['focus']}\n")
            parts.append(mod["instructions"])

    return "\n".join(parts)


def get_system_prompt(stage: str) -> str:
    """Return the system prompt for a stage."""
    return STAGE_CONFIGS[stage]["system"]
