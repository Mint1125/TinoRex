"""ML Toolkit documentation injected into LLM system prompt.

Provides the LLM with knowledge of proven patterns and available
libraries so it can make informed improvements rather than guessing.
"""

from __future__ import annotations

TOOLKIT_DOCS = """\
## Available ML Toolkit & Proven Patterns

You have access to these libraries and proven patterns. Use them to improve the solution.

### Feature Engineering Patterns (proven effective)
- **Target encoding** (with CV folds to prevent leakage):
  ```python
  from sklearn.model_selection import KFold
  kf = KFold(5, shuffle=True, random_state=42)
  for tr_idx, val_idx in kf.split(X):
      means = X.iloc[tr_idx].groupby(col)[target].mean()
      X.loc[val_idx, f'{col}_target_enc'] = X.loc[val_idx, col].map(means)
  ```
- **Frequency encoding**: `X[f'{col}_freq'] = X[col].map(X[col].value_counts(normalize=True))`
- **Interaction features**: `X['feat_a_x_b'] = X['a'] * X['b']`
- **Datetime decomposition**: year, month, day, hour, weekday, is_weekend
- **Text length / word count**: `X['text_len'] = X['text'].str.len()`
- **Missing value indicators**: `X[f'{col}_missing'] = X[col].isna().astype(int)`
- **Log transform** for skewed distributions: `np.log1p(X[col])`
- **Polynomial features** for key numeric pairs
- **Aggregation features**: group-level mean, std, count, min, max

### Model Configurations (battle-tested defaults)
- **LightGBM**: n_estimators=800, max_depth=7, lr=0.03, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
- **XGBoost**: n_estimators=800, max_depth=7, lr=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
- **CatBoost**: iterations=800, depth=7, lr=0.03, l2_leaf_reg=3.0
- **Neural nets** (tabular): Use simple MLP with BatchNorm, Dropout, lr=1e-3, epochs=50

### Ensemble Strategies
- **Weighted average**: `final = w1*pred1 + w2*pred2 + w3*pred3` (normalize weights)
- **Rank averaging**: `from scipy.stats import rankdata; avg_rank = mean of rankdata for each model`
- **Stacking with OOF**: Train base models with K-fold OOF, use OOF preds as features for meta-learner
- **Blending**: Hold out 20% for blending, train base on 80%, blend on holdout

### Common Pitfalls to Avoid
- Data leakage: NEVER use test data for feature engineering that depends on target
- Target encoding without CV folds = leakage
- Not matching sample_submission.csv format exactly
- Forgetting to handle NaN in predictions
- Using different preprocessing for train vs test
- Not stratifying folds for imbalanced classification

### Competition-Specific Heuristics
- If sample_submission has float values → likely expects probabilities (regression-style output)
- If sample_submission has int/string values → likely expects class labels
- Check if competition metric is specified in description.md
- For AUC-optimized competitions, output probabilities, not labels
- For accuracy-optimized, output labels with optional threshold tuning
"""


def get_toolkit_docs() -> str:
    """Return toolkit documentation string for LLM system prompt."""
    return TOOLKIT_DOCS
