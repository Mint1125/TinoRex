"""AIDE-style tree search over complete solution scripts.

The solver generates complete Python scripts at every node.  Each script must:
  1. Read data from ./home/data/
  2. Train a model and cross-validate, printing  CV_SCORE=<float>
  3. Predict on the test set and write ./submission.csv

The tree iterates: select best node -> ask LLM to improve -> execute -> score.
"""

from __future__ import annotations

import logging
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from interpreter import Interpreter
from llm import LLMClient
from ml_helpers import patch_submission_columns, validate_submission_report
from profiler import run_profiler
from refiner import refine

logger = logging.getLogger(__name__)

MAX_STDOUT_CHARS = 8000
STDOUT_FEEDBACK_CHARS = 6000

SYSTEM_PROMPT = """\
You are an expert ML engineer competing in a Kaggle-style competition.

You write COMPLETE, SELF-CONTAINED Python scripts.  Every script must:
1. Read data from ./home/data/ (train.csv, test.csv, sample_submission.csv, etc.)
2. Read ./home/data/description.md if present to understand the task.
3. Train a model using ANY library available (sklearn, lightgbm, xgboost, catboost, \
torch, torchvision, transformers, scipy, librosa, PIL/cv2, pandas, numpy, etc.).
4. Evaluate with cross-validation and print EXACTLY this line:  CV_SCORE=<float>
   Use the competition metric if known, otherwise use accuracy for classification \
or RMSE for regression.
5. Predict on the test set and save ./submission.csv matching the format of \
sample_submission.csv exactly (same columns, same row count, same ID column).

Rules:
- The script must be COMPLETE -- it will run in a fresh Python process.
- Include all imports at the top.
- Print CV_SCORE=<float> exactly once.  This is how your solution is scored.
- If you are unsure about the task type, inspect sample_submission.csv first.
- Handle errors gracefully: catch exceptions during training and print diagnostics.
- Keep stdout concise -- only print what is needed to debug.
- NEVER hardcode predictions or labels from memory -- always train a real model.
- ALWAYS read sample_submission.csv first to check expected format, columns, and row count.
- Before saving submission.csv, verify:
  (a) columns match sample_submission.csv exactly (same names, same order)
  (b) row count matches sample_submission.csv (or test.csv) exactly
  (c) no NaN values in any column
  (d) ID column values match sample_submission.csv
- If you are unsure about prediction values, use safe defaults (mean of train \
target for regression, mode for classification).
- You may save intermediate artifacts (features, models) to the working directory \
for potential reuse by later iterations. Use pickle or joblib for caching.
"""

INITIAL_PROMPT = """\
Competition description:
{description}

Files available:
{file_listing}

Strategy hint:
{strategy}

Write a COMPLETE Python script that solves this competition.  Remember to print \
CV_SCORE=<float> and save ./submission.csv.
"""

IMPROVE_PROMPT = """\
{description_section}

Your previous best solution (CV_SCORE={parent_score}):
```python
{parent_code}
```

Execution output (truncated):
```
{parent_stdout}
```

{error_context}

{validation_feedback}

{history_section}

Strategy bias: {strategy}

Make ONE specific improvement to increase the CV score.  Return the COMPLETE \
updated Python script.  Do not remove the CV_SCORE print.  Do not remove the \
submission.csv save.

CRITICAL submission requirements:
- submission.csv MUST have the exact same columns as sample_submission.csv
- submission.csv MUST have the exact same number of rows as sample_submission.csv
- submission.csv MUST NOT contain any NaN or missing values
- The ID column values must match sample_submission.csv exactly
- Always read sample_submission.csv first to understand the expected format

Focus on: {improvement_hint}
"""

# Ordered: early = safe/fast, late = complex/expensive
IMPROVEMENT_HINTS = [
    "fixing any errors or warnings from the previous run",
    "better preprocessing (missing value handling, encoding, scaling)",
    "better feature engineering (interactions, aggregations, domain-specific transforms)",
    "trying a different model family or algorithm",
    "hyperparameter tuning (learning rate, depth, regularization)",
    "data cleaning (outlier removal, deduplication, type casting)",
    "better cross-validation strategy (stratified, grouped, time-based)",
    "ensemble or blending multiple models",
]

RECOVERY_ITERATIONS = 3


@dataclass
class SolutionNode:
    node_id: int
    code: str
    cv_score: float | None = None
    stdout: str = ""
    exec_time: float = 0.0
    error: str | None = None
    parent_id: int | None = None
    iteration: int = 0
    validation_report: dict | None = None
    hint_used: str = ""


@dataclass
class TreeSearchResult:
    best_node: SolutionNode | None
    all_nodes: list[SolutionNode]
    total_time: float


class SolutionTree:
    def __init__(
        self,
        *,
        workdir: Path,
        llm: LLMClient,
        max_iterations: int = 12,
        code_timeout: int = 600,
        strategy_name: str = "",
        refine_steps: int = 5,
    ):
        self.workdir = workdir
        self.llm = llm
        self.max_iterations = max_iterations
        self.code_timeout = code_timeout
        self.strategy_name = strategy_name
        self.refine_steps = refine_steps
        self.nodes: list[SolutionNode] = []
        self._next_id = 0
        # Cached after first call
        self._file_listing: str | None = None
        self._description: str | None = None
        self._data_profile: str | None = None

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _list_files(self) -> str:
        if self._file_listing is not None:
            return self._file_listing
        data_dir = self.workdir / "home" / "data"
        if not data_dir.exists():
            data_dir = self.workdir
        entries = []
        for p in sorted(data_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.workdir)
                size_mb = p.stat().st_size / (1024 * 1024)
                entries.append(f"  ./{rel}  ({size_mb:.1f} MB)")
        self._file_listing = "\n".join(entries) if entries else "  <no files found>"
        return self._file_listing

    def _read_description(self) -> str:
        if self._description is not None:
            return self._description
        for name in ("description.md", "description.txt", "README.md"):
            path = self.workdir / "home" / "data" / name
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")
                if len(text) > 12000:
                    text = text[:12000] + "\n... (truncated)"
                self._description = text
                return self._description
        self._description = "<no description file found>"
        return self._description

    def _execute(self, code: str) -> tuple[float | None, str, float, str | None, dict]:
        """Run code, return (cv_score, stdout, exec_time, error, validation_report)."""
        interp = Interpreter(workdir=self.workdir, timeout=self.code_timeout)
        try:
            result = interp.run(code)
        finally:
            interp.cleanup()

        stdout = result.stdout
        if len(stdout) > MAX_STDOUT_CHARS:
            stdout = stdout[:MAX_STDOUT_CHARS] + "\n... (truncated)"

        error = None
        if not result.succeeded:
            error = result.exc_type or "UnknownError"

        cv_score = self._parse_cv_score(result.stdout)

        submission_path = self.workdir / "submission.csv"
        if submission_path.exists():
            validation = validate_submission_report(submission_path, self.workdir)
            if not validation["valid"]:
                patch_submission_columns(submission_path, self.workdir)
                validation = validate_submission_report(submission_path, self.workdir)
        else:
            validation = {
                "valid": False,
                "errors": ["submission.csv not found"],
                "warnings": [],
                "summary": "FAIL: submission.csv was not produced.",
            }
            if error is None:
                error = "NoSubmission"

        return cv_score, stdout, result.exec_time, error, validation

    @staticmethod
    def _parse_cv_score(stdout: str) -> float | None:
        matches = re.findall(r"CV_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stdout)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
        return None

    def _format_stdout_feedback(self, node: SolutionNode) -> str:
        """Format stdout for IMPROVE_PROMPT: show more context, head+tail for errors."""
        stdout = node.stdout
        if not stdout:
            return "<no output>"

        if node.error and len(stdout) > STDOUT_FEEDBACK_CHARS:
            # For errors: show head (setup/data info) + tail (traceback)
            head_n = STDOUT_FEEDBACK_CHARS // 3
            tail_n = STDOUT_FEEDBACK_CHARS - head_n - 60
            return (
                stdout[:head_n]
                + "\n...[middle truncated]...\n"
                + stdout[-tail_n:]
            )

        # For success: show tail (most recent output with scores)
        if len(stdout) > STDOUT_FEEDBACK_CHARS:
            return stdout[-STDOUT_FEEDBACK_CHARS:]

        return stdout

    def _build_history_section(self) -> str:
        """Build a summary of all previous attempts for the LLM."""
        if len(self.nodes) <= 1:
            return ""
        lines = ["Previous attempts (do NOT repeat failed approaches):"]
        for n in self.nodes:
            score_str = f"CV={n.cv_score:.4f}" if n.cv_score is not None else "no score"
            status = "OK" if n.error is None else f"ERROR({n.error})"
            valid = "valid" if (n.validation_report and n.validation_report.get("valid")) else "invalid"
            hint = n.hint_used or "initial"
            lines.append(
                f"  - Node {n.node_id} [{hint}]: {score_str}, {status}, submission={valid}"
            )
        return "\n".join(lines)

    def _select_parent(self, iteration: int) -> SolutionNode:
        """Select parent with diversity: epsilon-greedy + lineage rotation."""
        valid = [n for n in self.nodes if n.cv_score is not None and n.error is None]
        if not valid:
            scored = [n for n in self.nodes if n.cv_score is not None]
            if scored:
                valid = scored
            else:
                return self.nodes[-1]

        # Every 3rd iteration, pick a random node from top-50% to encourage exploration
        if iteration % 3 == 0 and len(valid) >= 2:
            sorted_nodes = sorted(valid, key=lambda n: n.cv_score, reverse=True)
            top_half = sorted_nodes[:max(1, len(sorted_nodes) // 2)]
            return random.choice(top_half)

        best = max(valid, key=lambda n: n.cv_score)

        # Avoid picking the same parent twice in a row
        if len(valid) >= 2 and len(self.nodes) >= 2:
            last_parent_id = self.nodes[-1].parent_id
            if best.node_id == last_parent_id:
                second_candidates = [n for n in valid if n.node_id != best.node_id]
                if second_candidates:
                    return max(second_candidates, key=lambda n: n.cv_score)

        return best

    def _best_node(self) -> SolutionNode | None:
        """Return the best node that has a valid submission and a score."""
        valid_sub = [
            n for n in self.nodes
            if n.cv_score is not None
            and n.validation_report is not None
            and n.validation_report.get("valid", False)
        ]
        if valid_sub:
            return max(valid_sub, key=lambda n: n.cv_score)

        scored = [n for n in self.nodes if n.cv_score is not None]
        if scored:
            return max(scored, key=lambda n: n.cv_score)

        no_error = [n for n in self.nodes if n.error is None]
        return no_error[-1] if no_error else (self.nodes[-1] if self.nodes else None)

    def _build_validation_feedback(self, node: SolutionNode) -> str:
        """Build validation feedback string for the IMPROVE_PROMPT."""
        if node.validation_report is None:
            return ""
        report = node.validation_report
        if not report.get("valid", True):
            return (
                "SUBMISSION VALIDATION ISSUES (fix these FIRST before improving the model):\n"
                + report["summary"]
            )
        if report.get("warnings"):
            return "Submission warnings:\n" + "\n".join(report["warnings"])
        return "Submission validation: OK (all checks passed)."

    def _build_description_section(self, iteration: int) -> str:
        """Full description for early nodes, abbreviated for later ones to save tokens."""
        desc = self._read_description()
        files = self._list_files()
        if iteration <= 2:
            return f"Competition description:\n{desc}\n\nFiles available:\n{files}"
        # After node 2, use abbreviated version
        if len(desc) > 3000:
            desc = desc[:3000] + "\n... (see full description in earlier iterations)"
        return f"Competition description (abbreviated):\n{desc}"

    def run(
        self,
        strategy: str,
        on_node_complete: Any = None,
    ) -> TreeSearchResult:
        """Run the full tree search loop with profiling and refinement phases."""
        start_time = time.time()
        description = self._read_description()
        file_listing = self._list_files()

        # -- Phase 1: Data profiling -----------------------------------------
        logger.info("Phase 1: Running data profiler")
        try:
            self._data_profile = run_profiler(self.workdir, timeout=60)
            logger.info("Data profile: %d chars", len(self._data_profile))
        except Exception as exc:
            logger.warning("Profiler failed: %s", exc)
            self._data_profile = ""

        # -- Phase 2: Tree search --------------------------------------------
        # -- Node 0: initial solution ----------------------------------------
        logger.info("Generating initial solution (strategy=%s)", strategy)
        profile_section = ""
        if self._data_profile:
            profile_text = self._data_profile[:4000] if len(self._data_profile) > 4000 else self._data_profile
            profile_section = f"\nData profile (auto-generated):\n{profile_text}\n"
        user_prompt = INITIAL_PROMPT.format(
            description=description,
            file_listing=file_listing,
            strategy=strategy,
        ) + profile_section
        code = self.llm.generate_code(system=SYSTEM_PROMPT, user=user_prompt)
        cv_score, stdout, exec_time, error, validation = self._execute(code)

        node0 = SolutionNode(
            node_id=self._new_id(),
            code=code,
            cv_score=cv_score,
            stdout=stdout,
            exec_time=exec_time,
            error=error,
            parent_id=None,
            iteration=0,
            validation_report=validation,
            hint_used=self.strategy_name or "initial",
        )
        self.nodes.append(node0)
        logger.info(
            "Node %d: cv_score=%s error=%s valid=%s exec_time=%.1fs",
            node0.node_id, cv_score, error,
            validation.get("valid"), exec_time,
        )
        if on_node_complete:
            on_node_complete(node0)

        # -- Iteration loop ---------------------------------------------------
        for iteration in range(1, self.max_iterations):
            parent = self._select_parent(iteration)
            hint = IMPROVEMENT_HINTS[iteration % len(IMPROVEMENT_HINTS)]

            error_context = ""
            if parent.error:
                error_context = (
                    f"The previous run had an error: {parent.error}\n"
                    "Fix this error first before making other improvements."
                )

            validation_feedback = self._build_validation_feedback(parent)
            history_section = self._build_history_section()
            description_section = self._build_description_section(iteration)
            stdout_feedback = self._format_stdout_feedback(parent)

            user_prompt = IMPROVE_PROMPT.format(
                description_section=description_section,
                parent_score=parent.cv_score if parent.cv_score is not None else "N/A (no score produced)",
                parent_code=parent.code,
                parent_stdout=stdout_feedback,
                error_context=error_context,
                validation_feedback=validation_feedback,
                history_section=history_section,
                strategy=strategy,
                improvement_hint=hint,
            )
            code = self.llm.generate_code(system=SYSTEM_PROMPT, user=user_prompt)
            cv_score, stdout, exec_time, error, validation = self._execute(code)

            node = SolutionNode(
                node_id=self._new_id(),
                code=code,
                cv_score=cv_score,
                stdout=stdout,
                exec_time=exec_time,
                error=error,
                parent_id=parent.node_id,
                iteration=iteration,
                validation_report=validation,
                hint_used=hint.split("(")[0].strip(),
            )
            self.nodes.append(node)
            logger.info(
                "Node %d (parent=%d): cv_score=%s error=%s valid=%s exec_time=%.1fs",
                node.node_id, parent.node_id, cv_score, error,
                validation.get("valid"), exec_time,
            )
            if on_node_complete:
                on_node_complete(node)

        # -- Recovery phase ---------------------------------------------------
        best = self._best_node()
        submission_path = self.workdir / "submission.csv"

        if best and (best.error or not submission_path.exists()):
            logger.info("Recovery phase: best node has issues, attempting recovery")

            if best.code:
                cv_score, stdout, exec_time, error, validation = self._execute(best.code)
                if submission_path.exists():
                    patch_submission_columns(submission_path, self.workdir)
                    best = self._best_node()

            if not submission_path.exists():
                logger.info("Recovery: running %d LLM recovery steps", RECOVERY_ITERATIONS)
                for ri in range(RECOVERY_ITERATIONS):
                    recovery_prompt = (
                        f"Competition description:\n{description}\n\n"
                        f"Files available:\n{file_listing}\n\n"
                        "CRITICAL: Your previous attempts failed to produce a valid submission.csv. "
                        "Write the SIMPLEST possible solution:\n"
                        "1. Read sample_submission.csv to understand the exact format needed\n"
                        "2. Read test.csv\n"
                        "3. Train a basic model (e.g. LogisticRegression or LGBMClassifier with defaults)\n"
                        "4. Predict and save ./submission.csv matching sample_submission.csv exactly\n"
                        "5. Print CV_SCORE=<float>\n"
                        "Prioritize producing a VALID submission over model quality."
                    )
                    code = self.llm.generate_code(system=SYSTEM_PROMPT, user=recovery_prompt)
                    cv_score, stdout, exec_time, error, validation = self._execute(code)
                    if submission_path.exists() and validation.get("valid", False):
                        node = SolutionNode(
                            node_id=self._new_id(), code=code,
                            cv_score=cv_score, stdout=stdout,
                            exec_time=exec_time, error=error,
                            parent_id=None, iteration=self.max_iterations + ri,
                            validation_report=validation,
                            hint_used="recovery",
                        )
                        self.nodes.append(node)
                        logger.info("Recovery step %d succeeded: cv_score=%s", ri, cv_score)
                        break

            if not submission_path.exists():
                sample_paths = list((self.workdir / "home" / "data").glob("sample_submission*.csv"))
                if sample_paths:
                    shutil.copy2(sample_paths[0], submission_path)
                    logger.warning("Recovery: copied sample_submission.csv as fallback")

        # Final patch on whatever submission we have
        if submission_path.exists():
            patch_submission_columns(submission_path, self.workdir)

        # -- Phase 3: Persistent session refinement ---------------------------
        best = self._best_node()
        if best and best.code and best.cv_score is not None and self.refine_steps > 0:
            logger.info(
                "Phase 3: Refining best node %d (cv_score=%s) with %d persistent session steps",
                best.node_id, best.cv_score, self.refine_steps,
            )
            try:
                refine_score, improved = refine(
                    workdir=self.workdir,
                    best_code=best.code,
                    llm=self.llm,
                    data_profile=self._data_profile or "",
                    max_steps=self.refine_steps,
                    timeout_per_step=self.code_timeout,
                )
                if improved and refine_score is not None:
                    logger.info(
                        "Refinement improved score: %s -> %s",
                        best.cv_score, refine_score,
                    )
                    # Re-patch submission after refinement
                    if submission_path.exists():
                        patch_submission_columns(submission_path, self.workdir)
            except Exception as exc:
                logger.warning("Refinement phase failed: %s", exc)

        best = self._best_node()
        total_time = time.time() - start_time
        logger.info(
            "Tree search complete: %d nodes, best=%s (cv_score=%s), total=%.0fs",
            len(self.nodes),
            best.node_id if best else None,
            best.cv_score if best else None,
            total_time,
        )
        return TreeSearchResult(best_node=best, all_nodes=self.nodes, total_time=total_time)
