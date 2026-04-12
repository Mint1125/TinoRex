"""v9 Hybrid Solver: Deterministic Baseline + Toolkit-Augmented Tree Search + Refinement.

5-phase pipeline:
  Phase 1: Data profiling (automated, no LLM)
  Phase 2: Deterministic baseline (stacking ensemble, no LLM)
  Phase 3: LLM tree search (improves FROM baseline, not from scratch)
  Phase 4: Persistent session refinement (incremental improvements)
  Phase 5: Final ensemble (blend baseline + tree search best)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from baseline import get_baseline_script
from interpreter import Interpreter
from llm import LLMClient
from ml_helpers import patch_submission_columns, validate_submission_report
from profiler import run_profiler
from refiner import refine
from strategies import DEFAULT_STRATEGY, get_strategy
from tree import SolutionTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_API_KEY_FILE = Path(r"C:/Users/PC4/OneDrive/바탕 화면/개인/개인정보/api_key.txt")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "o4-mini")
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "600"))


def _load_api_keys() -> dict[str, str]:
    keys: dict[str, str] = {}
    for env_var, provider in [
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("GOOGLE_API_KEY", "google"),
    ]:
        val = os.environ.get(env_var, "")
        if val:
            keys[provider] = val

    if _API_KEY_FILE.exists():
        lines = _API_KEY_FILE.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("sk-proj-") and "openai" not in keys:
                keys["openai"] = line
            elif line.startswith("sk-ant-") and "anthropic" not in keys:
                keys["anthropic"] = line
            elif line.startswith("AIza") and "google" not in keys:
                keys["google"] = line
    return keys


def _load_api_key() -> str:
    from llm import _detect_provider
    provider = _detect_provider(OPENAI_MODEL)
    keys = _load_api_keys()
    return keys.get(provider, keys.get("openai", ""))


def _extract_tar_b64(b64_text: str, dest: Path) -> None:
    raw = base64.b64decode(b64_text)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


def _first_tar_from_message(message: Message) -> str | None:
    for part in message.parts:
        root = part.root
        if isinstance(root, FilePart):
            fd = root.file
            if isinstance(fd, FileWithBytes) and fd.bytes is not None:
                raw = fd.bytes
                if isinstance(raw, str):
                    return raw
                if isinstance(raw, (bytes, bytearray)):
                    return base64.b64encode(raw).decode("ascii")
    return None


def _parse_strategy(message: Message) -> str:
    text = get_message_text(message)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload.get("strategy", DEFAULT_STRATEGY)
    except (json.JSONDecodeError, TypeError):
        pass
    return DEFAULT_STRATEGY


def _blend_submissions(
    baseline_path: Path,
    treesearch_path: Path,
    output_path: Path,
    workdir: Path,
    baseline_weight: float = 0.4,
) -> None:
    """Blend two submission CSVs by weighted average (numeric) or majority vote (labels)."""
    try:
        if not baseline_path.exists() or not treesearch_path.exists():
            # If one doesn't exist, use the other
            src = baseline_path if baseline_path.exists() else treesearch_path
            if src.exists():
                shutil.copy2(src, output_path)
            return

        base_df = pd.read_csv(baseline_path)
        tree_df = pd.read_csv(treesearch_path)

        if list(base_df.columns) != list(tree_df.columns):
            logger.warning("Blend: column mismatch, using tree search result")
            shutil.copy2(treesearch_path, output_path)
            return

        if len(base_df) != len(tree_df):
            logger.warning("Blend: row count mismatch, using tree search result")
            shutil.copy2(treesearch_path, output_path)
            return

        # Find sample submission to detect ID column
        data_dir = workdir / "home" / "data"
        sample_paths = sorted(data_dir.glob("sample_submission*.csv"))
        id_cols = set()
        if sample_paths:
            sample = pd.read_csv(sample_paths[0])
            # ID column: in both sample and test, unique values
            for col in sample.columns:
                if col.lower() in ("id", "index", "row_id") or sample[col].nunique() == len(sample):
                    id_cols.add(col)

        result = base_df.copy()
        tree_weight = 1.0 - baseline_weight

        for col in base_df.columns:
            if col in id_cols:
                continue

            if pd.api.types.is_numeric_dtype(base_df[col]) and pd.api.types.is_numeric_dtype(tree_df[col]):
                result[col] = baseline_weight * base_df[col] + tree_weight * tree_df[col]
            else:
                # For non-numeric: use tree search result (it's usually better)
                result[col] = tree_df[col]

        # Fill NaN
        for col in result.columns:
            if result[col].isna().any():
                if col in base_df.columns:
                    result[col] = result[col].fillna(base_df[col])
                result[col] = result[col].fillna(0)

        result.to_csv(output_path, index=False)
        logger.info("Blended submission: baseline_w=%.2f, tree_w=%.2f", baseline_weight, tree_weight)

    except Exception as exc:
        logger.warning("Blend failed: %s, using tree search result", exc)
        if treesearch_path.exists():
            shutil.copy2(treesearch_path, output_path)
        elif baseline_path.exists():
            shutil.copy2(baseline_path, output_path)


class Agent:
    def __init__(self):
        self._done_contexts: set[str] = set()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        ctx = message.context_id or "default"
        if ctx in self._done_contexts:
            return

        tar_b64 = _first_tar_from_message(message)
        if not tar_b64:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: no competition tar.gz"))],
                name="Error",
            )
            return

        strategy_name = _parse_strategy(message)
        strategy = get_strategy(strategy_name)

        api_key = _load_api_key()
        if not api_key:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: API key required"))],
                name="Error",
            )
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"v9 Hybrid Solver starting (strategy={strategy.name}, model={OPENAI_MODEL})"
            ),
        )

        with tempfile.TemporaryDirectory(prefix=f"solver-{ctx}-") as tmpdir:
            workdir = Path(tmpdir)
            submission_path = workdir / "submission.csv"

            try:
                _extract_tar_b64(tar_b64, workdir)
            except Exception as exc:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=f"Error extracting tar: {exc}"))],
                    name="Error",
                )
                return

            loop = __import__("asyncio").get_running_loop()

            # =================================================================
            # PHASE 1: Data Profiling
            # =================================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Phase 1: Data profiling..."),
            )
            try:
                data_profile = await loop.run_in_executor(
                    None, lambda: run_profiler(workdir, timeout=90)
                )
                logger.info("Phase 1 complete: %d chars profile", len(data_profile))
            except Exception as exc:
                logger.warning("Phase 1 failed: %s", exc)
                data_profile = ""

            # =================================================================
            # PHASE 2: Deterministic Baseline
            # =================================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Phase 2: Running deterministic baseline..."),
            )

            baseline_code = get_baseline_script()
            baseline_score = None
            baseline_submission = workdir / "_baseline_submission.csv"

            try:
                interp = Interpreter(workdir=workdir, timeout=CODE_TIMEOUT)
                try:
                    result = interp.run(baseline_code)
                finally:
                    interp.cleanup()

                if result.succeeded:
                    import re
                    matches = re.findall(
                        r"CV_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                        result.stdout,
                    )
                    if matches:
                        baseline_score = float(matches[-1])

                    if submission_path.exists():
                        patch_submission_columns(submission_path, workdir)
                        shutil.copy2(submission_path, baseline_submission)
                        logger.info("Phase 2 complete: baseline_score=%s", baseline_score)
                    else:
                        logger.warning("Phase 2: no submission.csv produced")
                else:
                    logger.warning("Phase 2 failed: %s", result.exc_type)
                    logger.warning("Baseline stderr: %s", result.stdout[-2000:] if result.stdout else "")

            except Exception as exc:
                logger.warning("Phase 2 error: %s", exc)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Phase 2 complete: baseline_score={baseline_score}"
                ),
            )

            # =================================================================
            # PHASE 3: LLM Tree Search (improving FROM baseline)
            # =================================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Phase 3: LLM tree search ({strategy.max_iterations} iterations)..."
                ),
            )

            llm = LLMClient(api_key=api_key, model=OPENAI_MODEL)
            tree = SolutionTree(
                workdir=workdir,
                llm=llm,
                max_iterations=strategy.max_iterations,
                code_timeout=CODE_TIMEOUT,
            )

            def on_node_complete(node):
                try:
                    __import__("asyncio").run_coroutine_threadsafe(
                        updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                f"Phase 3 Node {node.node_id}: "
                                f"cv={node.cv_score} err={node.error} "
                                f"time={node.exec_time:.0f}s"
                            ),
                        ),
                        loop,
                    )
                except Exception:
                    pass

            try:
                tree_result = await loop.run_in_executor(
                    None,
                    lambda: tree.run(
                        baseline_code=baseline_code,
                        data_profile=data_profile,
                        on_node_complete=on_node_complete,
                    ),
                )
                tree_score = tree_result.best_node.cv_score if tree_result.best_node else None
                logger.info("Phase 3 complete: tree_score=%s", tree_score)
            except Exception as exc:
                logger.exception("Phase 3 failed")
                tree_result = None
                tree_score = None

            # Save tree search submission
            tree_submission = workdir / "_tree_submission.csv"
            if submission_path.exists():
                shutil.copy2(submission_path, tree_submission)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Phase 3 complete: tree_score={tree_score} "
                    f"({len(tree_result.all_nodes) if tree_result else 0} nodes)"
                ),
            )

            # =================================================================
            # PHASE 4: Persistent Session Refinement
            # =================================================================
            best_code = None
            if tree_result and tree_result.best_node and tree_result.best_node.code:
                best_code = tree_result.best_node.code
            elif baseline_code:
                best_code = baseline_code

            refine_score = None
            if best_code and strategy.refine_steps > 0:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Phase 4: Refinement ({strategy.refine_steps} steps)..."
                    ),
                )
                try:
                    refine_score, improved = await loop.run_in_executor(
                        None,
                        lambda: refine(
                            workdir=workdir,
                            best_code=best_code,
                            llm=llm,
                            data_profile=data_profile,
                            max_steps=strategy.refine_steps,
                            timeout_per_step=CODE_TIMEOUT,
                        ),
                    )
                    logger.info("Phase 4 complete: refine_score=%s improved=%s", refine_score, improved)
                except Exception as exc:
                    logger.warning("Phase 4 failed: %s", exc)

            # =================================================================
            # PHASE 5: Final Ensemble
            # =================================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Phase 5: Final ensemble..."),
            )

            # Determine which submission to use or blend
            final_submission = submission_path

            if baseline_submission.exists() and tree_submission.exists():
                # Both exist: blend them
                _blend_submissions(
                    baseline_path=baseline_submission,
                    treesearch_path=tree_submission if not submission_path.exists() else submission_path,
                    output_path=final_submission,
                    workdir=workdir,
                    baseline_weight=0.3,  # Tree search usually better
                )
            elif not submission_path.exists():
                # Use whatever exists
                if tree_submission.exists():
                    shutil.copy2(tree_submission, final_submission)
                elif baseline_submission.exists():
                    shutil.copy2(baseline_submission, final_submission)

            # Final validation and patching
            if final_submission.exists():
                patch_submission_columns(final_submission, workdir)
                validation = validate_submission_report(final_submission, workdir)
                if not validation["valid"]:
                    logger.warning("Final validation issues: %s", validation["summary"])
                    # Fallback to best individual submission
                    if tree_submission.exists():
                        shutil.copy2(tree_submission, final_submission)
                        patch_submission_columns(final_submission, workdir)
                    elif baseline_submission.exists():
                        shutil.copy2(baseline_submission, final_submission)
                        patch_submission_columns(final_submission, workdir)

            if not final_submission.exists():
                # Last resort: copy sample submission
                data_dir = workdir / "home" / "data"
                sample_paths = list(data_dir.glob("sample_submission*.csv"))
                if sample_paths:
                    shutil.copy2(sample_paths[0], final_submission)
                    logger.warning("Using sample_submission as fallback")

            if not final_submission.exists():
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: no submission.csv produced"))],
                    name="Error",
                )
                return

            # Build summary
            best_score = max(
                filter(None, [baseline_score, tree_score, refine_score]),
                default=None,
            )

            csv_bytes = final_submission.read_bytes()
            b64_out = base64.b64encode(csv_bytes).decode("ascii")

            summary = (
                f"v9 Hybrid complete: "
                f"baseline={baseline_score}, "
                f"tree={tree_score} ({len(tree_result.all_nodes) if tree_result else 0} nodes), "
                f"refine={refine_score}, "
                f"best_score={best_score}, "
                f"strategy={strategy.name}"
            )

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(
                        root=FilePart(
                            file=FileWithBytes(
                                bytes=b64_out,
                                name="submission.csv",
                                mime_type="text/csv",
                            )
                        )
                    ),
                ],
                name="submission",
            )
            self._done_contexts.add(ctx)
            logger.info("Submitted: %d bytes, %s", len(csv_bytes), summary)
