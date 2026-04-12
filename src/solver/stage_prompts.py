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
lightgbm, xgboost, catboost, optuna, scipy, torch, torchvision.

Rules:
- NEVER hardcode predictions, labels, or column names from memory.
- Always discover the data schema by reading files at runtime.
- Handle errors gracefully and print diagnostics.
- Keep stdout concise — only print what is needed.
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

Output format:
```
EDA_REPORT_START
{...json...}
EDA_REPORT_END
```
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
- For each column: dtype, missing rate, unique count, top 5 values
- Numeric columns: mean, std, min, max, median, skewness
- Target distribution (value_counts for classification, histogram stats for regression)
- Pearson correlations between numeric features and target (top 10)
- Class balance ratio (for classification)

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
- Check for mixed types in columns (e.g., numbers stored as strings)
- Detect outliers (values beyond 3 std from mean)
- Check if train/test distributions differ significantly per column

Print the report as a JSON dict with keys: quality_issues (list of dicts with \
column, issue, details), leakage_suspects, high_cardinality_cols, \
mixed_type_cols, train_test_drift (list of columns with distribution shift).
""",
    },
    "C": {
        "focus": "domain understanding and feature opportunities",
        "instructions": """\
Focus on DOMAIN UNDERSTANDING and FEATURE ENGINEERING OPPORTUNITIES:
- Read description.md to understand the competition goal and evaluation metric
- Infer semantic meaning of columns from names and values
- Identify columns that could be split (e.g., delimited strings, composite IDs)
- Identify columns that could be grouped (e.g., same prefix, related semantics)
- Suggest interaction features (column pairs likely to have non-linear relationships)
- Suggest aggregation features (group-by statistics)
- Identify boolean-like columns stored as strings
- Detect datetime-like columns that could yield time features
- Suggest encoding strategies for different categorical types

Print the report as a JSON dict with keys: competition_goal, eval_metric (if found), \
splittable_cols (list with column and delimiter), groupable_cols, \
interaction_candidates (list of column pairs), aggregation_candidates, \
boolean_cols, datetime_cols, encoding_suggestions (dict of column: strategy).
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
Your task is feature engineering. Write a script that:
1. Reads train.csv and test.csv from ./home/data/
2. Applies feature engineering informed by the EDA report.
3. Runs a quick LightGBM 5-fold CV to validate feature quality.
4. Prints exactly: CV_SCORE=<float>
5. Saves engineered features as train_features.parquet and test_features.parquet \
in the current working directory.
6. Saves feature_names.json (list of feature column names).

IMPORTANT: The target column and ID column must NOT be in the feature parquets.
Save the target as target.npy and IDs as test_ids.npy separately.
"""

FEATURES_MODULES = {
    "A": {
        "focus": "interaction and polynomial features",
        "instructions": """\
Focus on INTERACTION and POLYNOMIAL features:
- Create pairwise interaction features for the top correlated numeric columns
- Create polynomial features (degree 2) for the most important numeric columns
- Create ratio features between related numeric columns
- Apply log1p transforms to skewed numeric columns
- Use the EDA report to guide which interactions are most promising
""",
    },
    "B": {
        "focus": "aggregation and groupby features",
        "instructions": """\
Focus on AGGREGATION and GROUP-BY features:
- If there are groupable columns (from EDA), compute group statistics (mean, std, count, min, max)
- If there are splittable columns, split them and create derived features
- Create frequency encoding for categorical columns
- Create target encoding using 5-fold CV (to avoid leakage) for classification tasks
- Use the EDA report to identify the best grouping keys
""",
    },
    "C": {
        "focus": "encoding strategies and missing value handling",
        "instructions": """\
Focus on ENCODING and MISSING VALUE strategies:
- Apply appropriate encoding per column type (from EDA suggestions):
  ordinal encoding for low-cardinality, target encoding for high-cardinality
- Create missing value indicator columns for columns with >5% missing
- Apply intelligent imputation (median for numeric, mode for categorical)
- Cast boolean-like string columns to int
- Parse datetime columns into year/month/day/dayofweek features
- Drop columns flagged as "drop" in the EDA report
""",
    },
}

# ---------------------------------------------------------------------------
# Stage 3: Model Selection
# ---------------------------------------------------------------------------

MODEL_SYSTEM = SYSTEM_PROMPT + """
Your task is model selection. Write a script that:
1. Loads train_features.parquet and target.npy from the current working directory.
2. Trains multiple models with default/reasonable hyperparameters using 5-fold CV.
3. For EACH model tested, prints: MODEL_RESULT={"name":"...", "cv_score":..., "train_time_s":...}
4. Prints the best CV_SCORE=<float>

Use StratifiedKFold for classification, KFold for regression.
"""

MODEL_MODULES = {
    "A": {
        "focus": "gradient boosting models",
        "instructions": """\
Test GRADIENT BOOSTING models:
- XGBoost (XGBClassifier/XGBRegressor) with reasonable defaults
- LightGBM (LGBMClassifier/LGBMRegressor) with reasonable defaults
- CatBoost (CatBoostClassifier/CatBoostRegressor) with reasonable defaults
- Print MODEL_RESULT for each.
""",
    },
    "B": {
        "focus": "linear and SVM models",
        "instructions": """\
Test LINEAR and SVM models:
- LogisticRegression / Ridge (with scaling via StandardScaler)
- SVM (SVC/SVR with scaling, use probability=True for classification)
- ElasticNet / SGDClassifier
- Print MODEL_RESULT for each.
Note: These models need feature scaling. Include StandardScaler in a Pipeline.
""",
    },
    "C": {
        "focus": "ensemble and neural models",
        "instructions": """\
Test ENSEMBLE and NEURAL models:
- RandomForest (RandomForestClassifier/RandomForestRegressor)
- ExtraTrees (ExtraTreesClassifier/ExtraTreesRegressor)
- MLP (MLPClassifier/MLPRegressor with scaling)
- HistGradientBoosting (HistGradientBoostingClassifier/HistGradientBoostingRegressor)
- Print MODEL_RESULT for each.
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
2. Model diversity — if the top 3 are all from the same family (e.g., all gradient boosting), \
replace the weakest with the best model from a different family.
3. Reasonable training time (skip models >300s unless significantly better)

Output a JSON list of 3 objects: [{{"name": "...", "cv_score": ..., "reason": "..."}}]
Output ONLY the JSON (no markdown fences, no explanation):
"""

# ---------------------------------------------------------------------------
# Stage 4: Optuna Hyperparameter Tuning
# ---------------------------------------------------------------------------

OPTUNA_SYSTEM = SYSTEM_PROMPT + """
Your task is hyperparameter tuning with Optuna. Write a script that:
1. Loads train_features.parquet and target.npy from the current working directory.
2. Defines an Optuna objective function for the specified model.
3. You DECIDE the search space based on your expertise with this model family \
and the dataset characteristics (n_rows, n_features from the EDA report).
4. Runs optuna.create_study().optimize(objective, n_trials=N).
5. Prints: BEST_PARAMS=<json dict>
6. Prints: BEST_SCORE=<float>
7. Saves best_params_{model_name}.json to the current working directory.

Use StratifiedKFold for classification, KFold for regression. Use 5 folds.
Suppress Optuna logging (optuna.logging.set_verbosity(optuna.logging.WARNING)).
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
Your task is building a stacking ensemble. Write a script that:
1. Loads train_features.parquet, test_features.parquet, and target.npy.
2. Loads tuned parameters from best_params_*.json files.
3. Level 1: For each of the specified base models, generate out-of-fold (OOF) \
predictions on train using K-fold CV, and full predictions on test.
4. Level 2: Train a meta-learner on the OOF predictions.
5. Prints: CV_SCORE=<float> (the full stacking pipeline CV score)
6. Saves: submission.csv (using test_ids.npy for the ID column), \
oof_predictions.npy, test_predictions.npy

Use 5-fold StratifiedKFold for classification, KFold for regression.
For binary classification, use predict_proba[:, 1] for OOF (probabilities).
"""

STACKING_MODULES = {
    "A": {
        "focus": "LogisticRegression/Ridge meta-learner",
        "instructions": """\
Meta-learner strategy: LogisticRegression (classification) or Ridge (regression).
- OOF matrix: shape (n_train, n_base_models) with predicted probabilities or values.
- Meta-learner trained on OOF matrix → predict on test predictions matrix.
- Apply proper CV on the meta-learner too (nested CV or simple fit on full OOF).
""",
    },
    "B": {
        "focus": "LightGBM meta-learner",
        "instructions": """\
Meta-learner strategy: LightGBM with low complexity (max_depth=3, n_estimators=100).
- OOF matrix: shape (n_train, n_base_models) with predicted probabilities or values.
- Use Optuna (10 trials) to quickly tune the meta-learner.
- This can capture non-linear interactions between base model predictions.
""",
    },
    "C": {
        "focus": "rank averaging (no meta-learner)",
        "instructions": """\
Meta-learner strategy: Rank averaging (no trained meta-learner).
- For each base model, rank its test predictions (scipy.stats.rankdata).
- Average the ranks across base models.
- For classification: convert averaged ranks to binary predictions \
using a threshold optimized on OOF rank averages.
- This is robust and avoids overfitting the meta-learner.
""",
    },
}

# ---------------------------------------------------------------------------
# Stage 6: Threshold Optimization
# ---------------------------------------------------------------------------

THRESHOLD_SYSTEM = SYSTEM_PROMPT + """
Your task is threshold optimization for classification. Write a script that:
1. Loads oof_predictions.npy and target.npy.
2. Searches for the optimal classification threshold to maximize the metric.
3. Prints: CV_SCORE=<float> (score with optimized threshold on OOF)
4. Prints: THRESHOLD=<float or json>
5. If the optimized score is better, loads test_predictions.npy, applies the \
threshold, and overwrites submission.csv (using test_ids.npy for IDs).

Read the target and sample_submission.csv to determine the output format \
(True/False vs 1/0 vs class labels).
"""

THRESHOLD_MODULES = {
    "A": {
        "focus": "grid search",
        "instructions": """\
Strategy: GRID SEARCH over thresholds.
- Search range: [0.01, 0.02, ..., 0.99] (99 thresholds)
- Evaluate each threshold on OOF predictions vs true labels
- Use accuracy (or the competition metric if known) as the objective
- Print the best threshold and its score
""",
    },
    "B": {
        "focus": "scipy optimization",
        "instructions": """\
Strategy: SCIPY OPTIMIZATION.
- Use scipy.optimize.minimize_scalar to find the optimal threshold
- Bounds: (0.01, 0.99), method='bounded'
- Objective: negative accuracy (or competition metric) on OOF
- This finds the precise optimum, not just grid points
""",
    },
    "C": {
        "focus": "Optuna threshold search",
        "instructions": """\
Strategy: OPTUNA THRESHOLD SEARCH.
- Use Optuna with 30 trials to optimize the threshold
- trial.suggest_float("threshold", 0.01, 0.99)
- Objective: accuracy (or competition metric) on OOF predictions
- For multiclass: optimize per-class thresholds
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
