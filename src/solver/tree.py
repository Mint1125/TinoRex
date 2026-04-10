"""AIDE-style tree search over complete solution scripts.

The solver generates complete Python scripts at every node.  Each script must:
  1. Read data from ./home/data/
  2. Train a model and cross-validate, printing  CV_SCORE=<float>
  3. Predict on the test set and write ./submission.csv

The tree iterates: select best node → ask LLM to improve → execute → score.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from interpreter import Interpreter
from llm import LLMClient

logger = logging.getLogger(__name__)

MAX_STDOUT_CHARS = 8000

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
- The script must be COMPLETE — it will run in a fresh Python process.
- Include all imports at the top.
- Print CV_SCORE=<float> exactly once.  This is how your solution is scored.
- If you are unsure about the task type, inspect sample_submission.csv first.
- Handle errors gracefully: catch exceptions during training and print diagnostics.
- Keep stdout concise — only print what is needed to debug.
- NEVER hardcode predictions or labels from memory — always train a real model.
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
Competition description:
{description}

Your previous best solution (CV_SCORE={parent_score}):
```python
{parent_code}
```

Execution output (truncated):
```
{parent_stdout}
```

{error_context}

Make ONE specific improvement to increase the CV score.  Return the COMPLETE \
updated Python script.  Do not remove the CV_SCORE print.  Do not remove the \
submission.csv save.

Focus on: {improvement_hint}
"""

IMPROVEMENT_HINTS = [
    "better feature engineering (interactions, aggregations, domain-specific transforms)",
    "trying a different model family or algorithm",
    "hyperparameter tuning (learning rate, depth, regularization)",
    "better preprocessing (missing value handling, encoding, scaling)",
    "ensemble or blending multiple models",
    "fixing any errors or warnings from the previous run",
    "better cross-validation strategy (stratified, grouped, time-based)",
    "data cleaning (outlier removal, deduplication, type casting)",
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

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _list_files(self) -> str:
        data_dir = self.workdir / "home" / "data"
        if not data_dir.exists():
            data_dir = self.workdir
        entries = []
        for p in sorted(data_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.workdir)
                size_mb = p.stat().st_size / (1024 * 1024)
                entries.append(f"  ./{rel}  ({size_mb:.1f} MB)")
        return "\n".join(entries) if entries else "  <no files found>"

    def _read_description(self) -> str:
        for name in ("description.md", "description.txt", "README.md"):
            path = self.workdir / "home" / "data" / name
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="replace")
                if len(text) > 12000:
                    text = text[:12000] + "\n... (truncated)"
                return text
        return "<no description file found>"

    def _execute(self, code: str) -> tuple[float | None, str, float, str | None]:
        """Run code, return (cv_score, stdout, exec_time, error)."""
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
        if not submission_path.exists():
            if error is None:
                error = "NoSubmission"

        return cv_score, stdout, result.exec_time, error

    @staticmethod
    def _parse_cv_score(stdout: str) -> float | None:
        matches = re.findall(r"CV_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stdout)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
        return None

    def _select_parent(self) -> SolutionNode:
        """Select the best-scoring node to branch from."""
        valid = [n for n in self.nodes if n.cv_score is not None and n.error is None]
        if valid:
            return max(valid, key=lambda n: n.cv_score)
        scored = [n for n in self.nodes if n.cv_score is not None]
        if scored:
            return max(scored, key=lambda n: n.cv_score)
        return self.nodes[-1]

    def _best_node(self) -> SolutionNode | None:
        """Return the best node that has a submission.csv and a score."""
        valid = [n for n in self.nodes if n.cv_score is not None]
        if not valid:
            # Fallback: any node without an error
            no_error = [n for n in self.nodes if n.error is None]
            return no_error[-1] if no_error else (self.nodes[-1] if self.nodes else None)
        return max(valid, key=lambda n: n.cv_score)

    def run(
        self,
        strategy: str,
        on_node_complete: Any = None,
    ) -> TreeSearchResult:
        """Run the full tree search loop."""
        start_time = time.time()
        description = self._read_description()
        file_listing = self._list_files()

        # ── Node 0: initial solution ─────────────────────────────────────
        logger.info("Generating initial solution (strategy=%s)", strategy)
        user_prompt = INITIAL_PROMPT.format(
            description=description,
            file_listing=file_listing,
            strategy=strategy,
        )
        code = self.llm.generate_code(system=SYSTEM_PROMPT, user=user_prompt)
        cv_score, stdout, exec_time, error = self._execute(code)

        node0 = SolutionNode(
            node_id=self._new_id(),
            code=code,
            cv_score=cv_score,
            stdout=stdout,
            exec_time=exec_time,
            error=error,
            parent_id=None,
            iteration=0,
        )
        self.nodes.append(node0)
        logger.info(
            "Node %d: cv_score=%s error=%s exec_time=%.1fs",
            node0.node_id, cv_score, error, exec_time,
        )
        if on_node_complete:
            on_node_complete(node0)

        # ── Iteration loop ───────────────────────────────────────────────
        for iteration in range(1, self.max_iterations):
            parent = self._select_parent()
            hint = IMPROVEMENT_HINTS[iteration % len(IMPROVEMENT_HINTS)]

            error_context = ""
            if parent.error:
                error_context = (
                    f"The previous run had an error: {parent.error}\n"
                    "Fix this error first before making other improvements."
                )

            user_prompt = IMPROVE_PROMPT.format(
                description=description,
                parent_score=parent.cv_score if parent.cv_score is not None else "N/A (no score produced)",
                parent_code=parent.code,
                parent_stdout=parent.stdout[-3000:] if parent.stdout else "<no output>",
                error_context=error_context,
                improvement_hint=hint,
            )
            code = self.llm.generate_code(system=SYSTEM_PROMPT, user=user_prompt)
            cv_score, stdout, exec_time, error = self._execute(code)

            node = SolutionNode(
                node_id=self._new_id(),
                code=code,
                cv_score=cv_score,
                stdout=stdout,
                exec_time=exec_time,
                error=error,
                parent_id=parent.node_id,
                iteration=iteration,
            )
            self.nodes.append(node)
            logger.info(
                "Node %d (parent=%d): cv_score=%s error=%s exec_time=%.1fs",
                node.node_id, parent.node_id, cv_score, error, exec_time,
            )
            if on_node_complete:
                on_node_complete(node)

        total_time = time.time() - start_time
        best = self._best_node()
        logger.info(
            "Tree search complete: %d nodes, best=%s (cv_score=%s), total=%.0fs",
            len(self.nodes),
            best.node_id if best else None,
            best.cv_score if best else None,
            total_time,
        )
        return TreeSearchResult(best_node=best, all_nodes=self.nodes, total_time=total_time)
