"""v9 Hybrid Solver: Baseline safety net + LLM tree search + refinement.

Pipeline:
  Phase 1: Data profiling (automated)
  Phase 2: Baseline (deterministic, parallel safety net)
  Phase 3: LLM tree search (writes from scratch, NOT from baseline)
           + persistent session refinement (integrated into tree search)
  Final:   Pick best by CV score between baseline and tree search
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from baseline import get_baseline_script
from interpreter import Interpreter
from llm import LLMClient
from ml_helpers import patch_submission_columns, validate_submission_report
from profiler import run_profiler
from strategies import DEFAULT_STRATEGY, get_strategy
from tree import SolutionTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_API_KEY_FILE = Path(r"C:/Users/PC4/OneDrive/바탕 화면/개인/개인정보/api_key.txt")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "o4-mini")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "15"))
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "600"))
REFINE_STEPS = int(os.environ.get("REFINE_STEPS", "5"))


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


def _parse_cv_score(stdout: str) -> float | None:
    matches = re.findall(r"CV_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stdout)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return None


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
        strategy_text = get_strategy(strategy_name)

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
                f"v9 Solver starting (strategy={strategy_name}, model={OPENAI_MODEL}, "
                f"iterations={MAX_ITERATIONS})"
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
            # PHASE 2: Deterministic Baseline (safety net)
            # =================================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Phase 2: Running deterministic baseline (safety net)..."),
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
                    baseline_score = _parse_cv_score(result.stdout)
                    if submission_path.exists():
                        patch_submission_columns(submission_path, workdir)
                        shutil.copy2(submission_path, baseline_submission)
                        logger.info("Phase 2 complete: baseline_score=%s", baseline_score)
                    else:
                        logger.warning("Phase 2: no submission.csv produced")
                else:
                    logger.warning("Phase 2 failed: %s", result.exc_type)
            except Exception as exc:
                logger.warning("Phase 2 error: %s", exc)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Phase 2 done: baseline_score={baseline_score}"),
            )

            # =================================================================
            # PHASE 3: LLM Tree Search (from scratch, NOT from baseline)
            #          + integrated persistent refinement
            # =================================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Phase 3: LLM tree search ({MAX_ITERATIONS} iterations + "
                    f"{REFINE_STEPS} refine steps)..."
                ),
            )

            llm = LLMClient(api_key=api_key, model=OPENAI_MODEL)
            tree = SolutionTree(
                workdir=workdir,
                llm=llm,
                max_iterations=MAX_ITERATIONS,
                code_timeout=CODE_TIMEOUT,
                strategy_name=strategy_name,
                refine_steps=REFINE_STEPS,
            )

            def on_node_complete(node):
                try:
                    __import__("asyncio").run_coroutine_threadsafe(
                        updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                f"Node {node.node_id}: "
                                f"cv_score={node.cv_score} "
                                f"error={node.error} "
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
                        strategy=strategy_text,
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

            # =================================================================
            # FINAL: Pick best submission by CV score
            # =================================================================
            # Tree search result is current submission.csv
            tree_submission = workdir / "_tree_submission.csv"
            if submission_path.exists():
                shutil.copy2(submission_path, tree_submission)

            # Decide: use tree search or baseline?
            use_baseline = False
            if tree_score is not None and baseline_score is not None:
                use_baseline = baseline_score > tree_score
            elif tree_score is None and baseline_score is not None:
                use_baseline = True

            if use_baseline and baseline_submission.exists():
                logger.info("Using baseline (score=%s > tree=%s)", baseline_score, tree_score)
                shutil.copy2(baseline_submission, submission_path)
            elif tree_submission.exists():
                shutil.copy2(tree_submission, submission_path)
            elif baseline_submission.exists():
                shutil.copy2(baseline_submission, submission_path)

            # Final validation
            if submission_path.exists():
                patch_submission_columns(submission_path, workdir)

            if not submission_path.exists():
                # Last resort: sample submission
                data_dir = workdir / "home" / "data"
                sample_paths = list(data_dir.glob("sample_submission*.csv"))
                if sample_paths:
                    shutil.copy2(sample_paths[0], submission_path)
                    logger.warning("Using sample_submission as fallback")

            if not submission_path.exists():
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: no submission.csv produced"))],
                    name="Error",
                )
                return

            best_score = max(filter(None, [baseline_score, tree_score]), default=None)
            csv_bytes = submission_path.read_bytes()
            b64_out = base64.b64encode(csv_bytes).decode("ascii")

            summary = (
                f"Tree search complete: "
                f"{len(tree_result.all_nodes) if tree_result else 0} nodes, "
                f"best_score={best_score}, "
                f"tree={tree_score}, baseline={baseline_score}, "
                f"used={'baseline' if use_baseline else 'tree'}, "
                f"total_time={tree_result.total_time:.0f}s, "
                f"strategy={strategy_name}"
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
