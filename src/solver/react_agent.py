"""ReAct ML agent with persistent Python session and tool use.

Instead of generating complete scripts (tree search), the LLM interacts
with data through a persistent Python session using tool calls:
  - run_python: execute code snippets (variables survive between calls)
  - validate_submission: check submission.csv format before finishing

Inspired by BN-Purple-Agent's LangGraph ReAct approach, implemented
with native OpenAI tool calling (no LangGraph dependency).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from llm import LLMClient
from persistent_interp import PersistentInterpreter

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 4000
ERROR_HEAD_CHARS = 2200

SYSTEM_PROMPT = """\
You are an expert ML engineer solving a Kaggle competition from the MLE-bench benchmark.

TOOLS:
- `run_python`: executes Python in a PERSISTENT session (variables, imports, models survive).
- `validate_submission`: checks ./submission.csv vs sample submission and test row counts.
  Call it BEFORE you finish; if it reports FAIL, fix with run_python and validate again.

ENVIRONMENT:
- Working directory contains all competition files
- Competition description: ./home/data/description.md
- Competition data: ./home/data/ (train.csv, test.csv, etc.)
- List data files first: use os.listdir('./home/data/') to find exact filenames
- Your output target: ./submission.csv (save here when done)

WORKFLOW:
1. List ./home/data/ to find exact filenames, then read description.md
2. Read the sample submission file and print its columns — your submission MUST match exactly
3. Explore train/test shapes, columns, target distribution (keep prints brief)
4. Build a model with cross-validation aligned with the metric in description.md
   Try to achieve THE BEST possible score!
5. After evaluation, print exactly one line: CV_SCORE=<float>
   (higher is better; negate loss if lower is better)
6. Predict on the test set and save ./submission.csv with ALL columns from sample submission
7. Call `validate_submission` and fix any FAIL until it passes
8. Reply with plain text (no tool call) when finished and validate_submission passes

GUIDELINES:
- All file paths must start with ./ (e.g. ./home/data/train.csv)
- Standard ML libraries are available: pandas, numpy, sklearn, xgboost, lightgbm, catboost, torch
- Keep it simple and robust; a working baseline beats a crashed complex model
- Print only key metrics — avoid verbose output
- Add `import warnings; warnings.filterwarnings('ignore')` to suppress warnings
- If code raises an error, read the traceback and fix it in the next call
- CRITICAL: submission.csv must match sample submission schema and row count
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code in a persistent session. Variables, imports, "
                "and trained models survive between calls. Returns stdout/stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_submission",
            "description": (
                "Check ./submission.csv against sample submission schema, "
                "row count vs test.csv, and NA values. Call before finishing."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

_WARNING_RE = re.compile(
    r"^/.+?:\d+:.*?Warning:.*?$\n(?:^\s+.*?$\n)*",
    re.MULTILINE,
)
_CV_SCORE_RE = re.compile(
    r"CV_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def _strip_warnings(text: str) -> str:
    return _WARNING_RE.sub("", text).strip()


def _truncate_output(text: str, max_chars: int, *, is_error: bool = False) -> str:
    if len(text) <= max_chars:
        return text
    if not is_error:
        keep = max_chars - 80
        truncated = len(text) - keep
        return f"[...{truncated} chars truncated...]\n{text[-keep:]}"
    sep = "\n...[middle truncated]...\n"
    usable = max(400, max_chars - len(sep) - 60)
    head_n = min(ERROR_HEAD_CHARS, usable // 2)
    tail_n = usable - head_n
    if head_n + tail_n >= len(text):
        return text[:max_chars] + "\n[truncated]\n"
    return f"{text[:head_n]}{sep}{text[-tail_n:]}"


def _validate_submission_report(workdir: Path) -> str:
    """Check submission.csv vs sample and test."""
    import glob
    import pandas as pd

    sub_path = workdir / "submission.csv"
    data_dir = workdir / "home" / "data"
    lines: list[str] = []

    if not sub_path.is_file():
        return "validate_submission: FAIL — ./submission.csv does not exist."

    try:
        sub = pd.read_csv(sub_path)
    except Exception as exc:
        return f"validate_submission: FAIL — cannot read submission.csv: {exc}"

    lines.append(f"submission.csv: rows={len(sub)}, columns={list(sub.columns)}")

    sample_paths = sorted(glob.glob(str(data_dir / "sample_submission*.csv")))
    if sample_paths:
        try:
            sample = pd.read_csv(sample_paths[0])
            exp_cols = list(sample.columns)
            act_cols = list(sub.columns)
            if exp_cols == act_cols:
                lines.append("OK — column names and order match sample_submission.")
            else:
                lines.append(f"FAIL — columns differ. Expected {exp_cols}, got {act_cols}.")
            missing = [c for c in exp_cols if c not in sub.columns]
            extra = [c for c in act_cols if c not in exp_cols]
            if missing:
                lines.append(f"FAIL — missing columns: {missing}")
            if extra:
                lines.append(f"WARN — extra columns: {extra}")
        except Exception as exc:
            lines.append(f"WARN — could not read sample submission: {exc}")
    else:
        lines.append("WARN — no sample_submission*.csv found.")

    test_paths = sorted(glob.glob(str(data_dir / "test.csv")))
    if test_paths:
        try:
            test_df = pd.read_csv(test_paths[0])
            if len(sub) != len(test_df):
                lines.append(f"FAIL — row count {len(sub)} != test.csv rows {len(test_df)}.")
            else:
                lines.append(f"OK — row count matches test.csv ({len(test_df)}).")
        except Exception as exc:
            lines.append(f"WARN — could not read test.csv: {exc}")

    na_cols = sub.columns[sub.isna().any()].tolist()
    if na_cols:
        lines.append(f"FAIL — NA values in columns: {na_cols}")
    else:
        lines.append("OK — no NA values.")

    return "\n".join(lines)


class ReactMLAgent:
    """ReAct agent that solves ML competitions via persistent Python session."""

    def __init__(
        self,
        workdir: Path,
        llm: LLMClient,
        max_iterations: int = 30,
        code_timeout: int = 600,
        exploration_hint: str | None = None,
    ):
        self.workdir = workdir
        self.llm = llm
        self.max_iterations = max_iterations
        self.code_timeout = code_timeout
        self.exploration_hint = exploration_hint
        self.best_cv_score: float | None = None
        self._interpreter: PersistentInterpreter | None = None

    def _get_interpreter(self) -> PersistentInterpreter:
        if self._interpreter is None:
            self._interpreter = PersistentInterpreter(
                workdir=self.workdir, timeout=self.code_timeout
            )
            self._interpreter.start()
        return self._interpreter

    def _execute_python(self, code: str) -> str:
        """Run code in persistent session, track CV score, return output."""
        interp = self._get_interpreter()
        result = interp.run(code)

        clean = _strip_warnings(result.output)
        # Track CV score
        match = _CV_SCORE_RE.findall(clean)
        if match:
            try:
                score = float(match[-1])
                if self.best_cv_score is None or score > self.best_cv_score:
                    self.best_cv_score = score
            except ValueError:
                pass

        is_err = result.error is not None
        return _truncate_output(clean, MAX_OUTPUT_CHARS, is_error=is_err)

    def _validate_submission(self) -> str:
        return _validate_submission_report(self.workdir)

    def _handle_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tool calls and return tool result messages."""
        results = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            tc_id = tc["id"]

            if name == "run_python":
                code = args.get("code", "")
                output = self._execute_python(code)
                logger.info(
                    "  run_python: %d chars code → %d chars output, cv=%s",
                    len(code), len(output), self.best_cv_score,
                )
                results.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": output,
                })
            elif name == "validate_submission":
                report = self._validate_submission()
                logger.info("  validate_submission: %s", report[:200])
                results.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": report,
                })
            else:
                results.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": f"Unknown tool: {name}",
                })

        return results

    def run(
        self,
        instructions: str,
        on_step: callable = None,
    ) -> Path | None:
        """Run the ReAct agent loop. Returns path to submission.csv or None."""
        try:
            # Build initial messages
            user_content = instructions
            if self.exploration_hint:
                user_content = (
                    f"[Exploration — follow this bias for your first approach]\n"
                    f"{self.exploration_hint}\n\n"
                    f"[Task instructions]\n{instructions}"
                )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # ReAct loop
            for step in range(self.max_iterations):
                logger.info("Step %d/%d", step + 1, self.max_iterations)
                if on_step:
                    on_step(step + 1, self.max_iterations, self.best_cv_score)

                # Call LLM with tools
                response = self.llm.chat(
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                )

                # Add assistant message to history
                messages.append(response["message"])

                # Check for tool calls
                tool_calls = response.get("tool_calls", [])
                if not tool_calls:
                    # LLM is done (no tool call = finished)
                    logger.info("Agent finished (no tool call) at step %d", step + 1)
                    break

                # Execute tools and add results
                tool_results = self._handle_tool_calls(tool_calls)
                messages.extend(tool_results)

            # Recovery if no submission
            submission = self.workdir / "submission.csv"
            if not submission.exists():
                logger.warning("No submission.csv after main phase, attempting recovery")
                recovery_msg = {
                    "role": "user",
                    "content": (
                        "CRITICAL: ./submission.csv is still missing. "
                        "Use run_python to create it matching sample_submission columns. "
                        "Then call validate_submission."
                    ),
                }
                messages.append(recovery_msg)

                recovery_steps = max(5, self.max_iterations // 4)
                for step in range(recovery_steps):
                    if on_step:
                        on_step(step + 1, recovery_steps, self.best_cv_score)

                    response = self.llm.chat(messages=messages, tools=TOOL_DEFINITIONS)
                    messages.append(response["message"])

                    tool_calls = response.get("tool_calls", [])
                    if not tool_calls:
                        break

                    tool_results = self._handle_tool_calls(tool_calls)
                    messages.extend(tool_results)

                    if submission.exists():
                        break

            return submission if submission.exists() else None

        finally:
            if self._interpreter:
                self._interpreter.cleanup()
                self._interpreter = None
