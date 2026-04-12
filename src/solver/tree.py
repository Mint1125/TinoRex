"""Toolkit-augmented tree search over ML solutions.

v9 key difference from v8:
  - Node 0 = deterministic baseline (not LLM-generated)
  - LLM IMPROVES from baseline, not from scratch
  - System prompt includes toolkit docs (proven ML patterns)
  - UCB-inspired parent selection for better exploration
  - Shorter improvement prompts (baseline already works)
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from interpreter import Interpreter
from llm import LLMClient
from ml_helpers import patch_submission_columns, validate_submission_report
from toolkit import get_toolkit_docs

logger = logging.getLogger(__name__)

MAX_STDOUT_CHARS = 8000
STDOUT_FEEDBACK_CHARS = 6000

SYSTEM_PROMPT = """\
You are an expert ML engineer improving an existing solution for a Kaggle-style competition.

You write COMPLETE, SELF-CONTAINED Python scripts. Every script must:
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
- Print CV_SCORE=<float> exactly once.
- ALWAYS read sample_submission.csv first to check expected format.
- Before saving submission.csv, verify: columns match, row count matches, no NaN.
- You are IMPROVING an existing solution. Focus on what the current solution does \
poorly and make it better. Do NOT start over from scratch unless the approach is \
fundamentally wrong.

{toolkit_docs}
"""

IMPROVE_PROMPT = """\
{description_section}

The current best solution (CV_SCORE={parent_score}):
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

Make ONE specific improvement to increase the CV score. Return the COMPLETE \
updated Python script.

CRITICAL submission requirements:
- submission.csv MUST have the exact same columns as sample_submission.csv
- submission.csv MUST have the exact same number of rows as sample_submission.csv
- submission.csv MUST NOT contain any NaN or missing values
- The ID column values must match sample_submission.csv exactly

Focus on: {improvement_hint}
"""

# Improvement hints ordered by impact
IMPROVEMENT_HINTS = [
    "fixing any errors or warnings from the previous run — ensure valid submission first",
    "better feature engineering: interactions, aggregations, target encoding with CV folds",
    "hyperparameter tuning: try learning_rate=0.03, more estimators, different depth",
    "try a different model family or add a new model to the ensemble",
    "ensemble or stacking: combine multiple models with proper OOF predictions",
    "data cleaning: outlier removal, better missing value handling, type casting",
    "better cross-validation strategy: stratified, grouped, or time-based splits",
    "advanced features: rank features, polynomial interactions, frequency encoding",
]


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
    visit_count: int = 0  # for UCB selection


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
    ):
        self.workdir = workdir
        self.llm = llm
        self.max_iterations = max_iterations
        self.code_timeout = code_timeout
        self.nodes: list[SolutionNode] = []
        self._next_id = 0
        self._file_listing: str | None = None
        self._description: str | None = None
        self._data_profile: str | None = None
        self._total_visits = 0

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
        stdout = node.stdout
        if not stdout:
            return "<no output>"
        if node.error and len(stdout) > STDOUT_FEEDBACK_CHARS:
            head_n = STDOUT_FEEDBACK_CHARS // 3
            tail_n = STDOUT_FEEDBACK_CHARS - head_n - 60
            return stdout[:head_n] + "\n...[middle truncated]...\n" + stdout[-tail_n:]
        if len(stdout) > STDOUT_FEEDBACK_CHARS:
            return stdout[-STDOUT_FEEDBACK_CHARS:]
        return stdout

    def _build_history_section(self) -> str:
        if len(self.nodes) <= 1:
            return ""
        lines = ["Previous attempts (do NOT repeat failed approaches):"]
        for n in self.nodes:
            score_str = f"CV={n.cv_score:.6f}" if n.cv_score is not None else "no score"
            status = "OK" if n.error is None else f"ERROR({n.error})"
            valid = "valid" if (n.validation_report and n.validation_report.get("valid")) else "invalid"
            hint = n.hint_used or "baseline"
            lines.append(f"  - Node {n.node_id} [{hint}]: {score_str}, {status}, submission={valid}")
        return "\n".join(lines)

    def _select_parent(self) -> SolutionNode:
        """UCB1-inspired parent selection: balance exploitation (best score) with exploration."""
        valid = [n for n in self.nodes if n.cv_score is not None and n.error is None]
        if not valid:
            scored = [n for n in self.nodes if n.cv_score is not None]
            valid = scored if scored else [self.nodes[-1]]

        if len(valid) == 1:
            return valid[0]

        self._total_visits += 1
        C = 1.5  # exploration constant

        # Normalize scores to [0, 1] for UCB
        scores = [n.cv_score for n in valid]
        min_s, max_s = min(scores), max(scores)
        score_range = max_s - min_s if max_s > min_s else 1.0

        best_ucb = -float("inf")
        best_node = valid[0]
        for node in valid:
            normalized_score = (node.cv_score - min_s) / score_range
            exploration = C * math.sqrt(math.log(self._total_visits + 1) / (node.visit_count + 1))
            ucb_value = normalized_score + exploration

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_node = node

        best_node.visit_count += 1
        return best_node

    def _best_node(self) -> SolutionNode | None:
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
        if node.validation_report is None:
            return ""
        report = node.validation_report
        if not report.get("valid", True):
            return (
                "SUBMISSION VALIDATION ISSUES (fix these FIRST):\n"
                + report["summary"]
            )
        if report.get("warnings"):
            return "Submission warnings:\n" + "\n".join(report["warnings"])
        return "Submission validation: OK."

    def _build_description_section(self, iteration: int) -> str:
        desc = self._read_description()
        files = self._list_files()
        profile = ""
        if self._data_profile:
            p = self._data_profile[:3000] if len(self._data_profile) > 3000 else self._data_profile
            profile = f"\n\nData profile:\n{p}"
        if iteration <= 2:
            return f"Competition description:\n{desc}\n\nFiles:\n{files}{profile}"
        if len(desc) > 3000:
            desc = desc[:3000] + "\n... (truncated)"
        return f"Competition (abbreviated):\n{desc}{profile}"

    def run(
        self,
        baseline_code: str,
        data_profile: str = "",
        on_node_complete: Any = None,
    ) -> TreeSearchResult:
        """Run toolkit-augmented tree search starting from deterministic baseline."""
        start_time = time.time()
        self._data_profile = data_profile

        # -- Node 0: Execute deterministic baseline (no LLM) -------------------
        logger.info("Node 0: executing deterministic baseline")
        cv_score, stdout, exec_time, error, validation = self._execute(baseline_code)

        node0 = SolutionNode(
            node_id=self._new_id(),
            code=baseline_code,
            cv_score=cv_score,
            stdout=stdout,
            exec_time=exec_time,
            error=error,
            parent_id=None,
            iteration=0,
            validation_report=validation,
            hint_used="deterministic_baseline",
        )
        self.nodes.append(node0)
        logger.info(
            "Node 0 (baseline): cv_score=%s error=%s valid=%s time=%.1fs",
            cv_score, error, validation.get("valid"), exec_time,
        )
        if on_node_complete:
            on_node_complete(node0)

        # -- Iteration loop: LLM improves from baseline -------------------------
        system_prompt = SYSTEM_PROMPT.format(toolkit_docs=get_toolkit_docs())

        for iteration in range(1, self.max_iterations):
            parent = self._select_parent()
            hint = IMPROVEMENT_HINTS[iteration % len(IMPROVEMENT_HINTS)]

            error_context = ""
            if parent.error:
                error_context = (
                    f"The previous run had an error: {parent.error}\n"
                    "Fix this error FIRST before making other improvements."
                )

            user_prompt = IMPROVE_PROMPT.format(
                description_section=self._build_description_section(iteration),
                parent_score=(
                    f"{parent.cv_score:.6f}" if parent.cv_score is not None
                    else "N/A (no score)"
                ),
                parent_code=parent.code,
                parent_stdout=self._format_stdout_feedback(parent),
                error_context=error_context,
                validation_feedback=self._build_validation_feedback(parent),
                history_section=self._build_history_section(),
                improvement_hint=hint,
            )
            code = self.llm.generate_code(system=system_prompt, user=user_prompt)
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
                hint_used=hint.split("—")[0].strip() if "—" in hint else hint[:40],
            )
            self.nodes.append(node)
            logger.info(
                "Node %d (parent=%d): cv_score=%s error=%s valid=%s time=%.1fs",
                node.node_id, parent.node_id, cv_score, error,
                validation.get("valid"), exec_time,
            )
            if on_node_complete:
                on_node_complete(node)

        # -- Recovery if needed -------------------------------------------------
        best = self._best_node()
        submission_path = self.workdir / "submission.csv"

        if best and (best.error or not submission_path.exists()):
            logger.info("Recovery: re-executing best node %d", best.node_id)
            if best.code:
                self._execute(best.code)
                if submission_path.exists():
                    patch_submission_columns(submission_path, self.workdir)

        # -- Re-execute best to ensure submission is current --------------------
        best = self._best_node()
        if best and best.code:
            logger.info("Re-executing best node %d (cv=%s)", best.node_id, best.cv_score)
            self._execute(best.code)
            if submission_path.exists():
                patch_submission_columns(submission_path, self.workdir)

        total_time = time.time() - start_time
        best = self._best_node()
        logger.info(
            "Tree search complete: %d nodes, best=%s (cv=%s), total=%.0fs",
            len(self.nodes),
            best.node_id if best else None,
            best.cv_score if best else None,
            total_time,
        )
        return TreeSearchResult(best_node=best, all_nodes=self.nodes, total_time=total_time)
