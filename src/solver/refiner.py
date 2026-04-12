"""ReAct-style refinement phase using persistent session.

After tree search finds the best script, this module runs it in a persistent
Python session and iteratively improves it with short code snippets.
Variables, models, and data survive between calls — no cold starts.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from llm import LLMClient
from persistent_interp import PersistentInterpreter

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 4000

REFINE_SYSTEM = """\
You are refining an ML solution in a PERSISTENT Python session.
All variables, imports, and trained models from previous executions are still in memory.

Write SHORT code snippets (NOT complete scripts). You can:
- Access all existing variables directly (dataframes, models, features, etc.)
- Retrain with different hyperparameters: model.set_params(n_estimators=500); model.fit(X, y)
- Add new features: X_train['new'] = X_train['a'] * X_train['b']
- Try different models: from xgboost import XGBClassifier; model2 = XGBClassifier(); model2.fit(X, y)
- Ensemble: pred = 0.5 * pred1 + 0.5 * pred2
- Inspect data: print(df.describe()), print(model.feature_importances_)

Rules:
- Print CV_SCORE=<float> after every re-evaluation.
- ALWAYS save ./submission.csv at the end of each snippet after making predictions.
  Use: submission.to_csv('./submission.csv', index=False)
- Keep snippets focused — ONE improvement per turn.
- Do NOT reload data from files unless necessary — use existing variables.
- import warnings; warnings.filterwarnings('ignore') at the start if needed.
"""

REFINE_PROMPT = """\
Data profile:
{data_profile}

Session output from last execution:
```
{last_output}
```

Current best CV_SCORE={best_score}
Refinement step {step}/{max_steps}

Write a SHORT Python code snippet for ONE improvement. Do NOT rewrite everything from scratch.
IMPORTANT: After predicting, ALWAYS save submission.csv:
  submission.to_csv('./submission.csv', index=False)
  print('Saved submission.csv')

Focus on: {focus}
"""

REFINE_FOCUSES = [
    "inspect feature importances or correlations, then engineer the most impactful new feature",
    "tune the most sensitive hyperparameters (learning_rate, max_depth, n_estimators)",
    "try a different model family and compare scores",
    "blend or stack predictions from multiple models for better generalization",
    "fix any remaining issues with the submission format or NaN values",
]


def _parse_cv_score(text: str) -> float | None:
    matches = re.findall(r"CV_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head - 40
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def refine(
    *,
    workdir: Path,
    best_code: str,
    llm: LLMClient,
    data_profile: str,
    max_steps: int = 5,
    timeout_per_step: int = 300,
) -> tuple[float | None, bool]:
    """Run persistent-session refinement on the best tree search result.

    Returns (best_cv_score, improved: bool).
    Backs up submission.csv before each step and restores if score drops.
    """
    submission_path = workdir / "submission.csv"
    backup_path = workdir / "_submission_backup.csv"

    interp = PersistentInterpreter(workdir=workdir, timeout=timeout_per_step)
    try:
        interp.start()

        # Step 0: Execute the best code from tree search in persistent session
        logger.info("Refinement: executing best code in persistent session")
        result = interp.run(best_code)
        if not result.succeeded:
            logger.warning("Refinement: best code failed in persistent session: %s", result.error)
            return None, False

        initial_score = _parse_cv_score(result.output)
        best_score = initial_score
        last_output = _truncate(result.output, MAX_OUTPUT_CHARS)
        logger.info("Refinement: initial CV_SCORE=%s", initial_score)

        if initial_score is None:
            logger.warning("Refinement: no CV_SCORE from initial execution, skipping")
            return None, False

        # Backup the initial (known-good) submission
        if submission_path.exists():
            shutil.copy2(submission_path, backup_path)
            logger.info("Refinement: backed up initial submission.csv")

        profile_for_prompt = _truncate(data_profile, 3000)

        # Steps 1-N: iterative improvement
        for step in range(1, max_steps + 1):
            focus = REFINE_FOCUSES[(step - 1) % len(REFINE_FOCUSES)]

            user_prompt = REFINE_PROMPT.format(
                data_profile=profile_for_prompt,
                last_output=last_output,
                best_score=best_score if best_score is not None else "N/A",
                step=step,
                max_steps=max_steps,
                focus=focus,
            )

            snippet = llm.generate_code(system=REFINE_SYSTEM, user=user_prompt)
            if not snippet.strip():
                logger.info("Refinement step %d: LLM returned empty snippet, stopping", step)
                break

            result = interp.run(snippet)
            last_output = _truncate(result.output, MAX_OUTPUT_CHARS)

            if result.error:
                logger.warning(
                    "Refinement step %d: error=%s, restoring backup",
                    step, result.error,
                )
                # Restore backup on error
                if backup_path.exists():
                    shutil.copy2(backup_path, submission_path)
                continue

            score = _parse_cv_score(result.output)
            logger.info(
                "Refinement step %d/%d: cv_score=%s time=%.1fs",
                step, max_steps, score, result.exec_time,
            )

            if score is not None and score > best_score:
                best_score = score
                # Update backup with the improved submission
                if submission_path.exists():
                    shutil.copy2(submission_path, backup_path)
                    logger.info("Refinement step %d: improved! new best=%s", step, best_score)
            elif score is not None and score < best_score:
                # Score dropped — restore backup
                logger.info(
                    "Refinement step %d: score dropped %s -> %s, restoring backup",
                    step, best_score, score,
                )
                if backup_path.exists():
                    shutil.copy2(backup_path, submission_path)

        # Ensure the best submission is in place
        if backup_path.exists():
            if not submission_path.exists():
                shutil.copy2(backup_path, submission_path)
            backup_path.unlink(missing_ok=True)

        improved = best_score is not None and initial_score is not None and best_score > initial_score
        logger.info(
            "Refinement complete: initial=%s best=%s improved=%s",
            initial_score, best_score, improved,
        )
        return best_score, improved

    finally:
        interp.cleanup()
        # Safety: ensure backup doesn't leak
        if backup_path.exists():
            if not submission_path.exists():
                shutil.copy2(backup_path, submission_path)
            backup_path.unlink(missing_ok=True)
