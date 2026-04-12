"""v10 ReAct Solver: Persistent session agent with tool use.

Replaces tree search (complete script generation) with ReAct pattern:
  - LLM interacts via run_python (persistent session) + validate_submission
  - Variables, imports, and models survive between calls
  - LLM decides when to explore, model, validate, and finish
  - Baseline safety net if ReAct agent fails to produce submission
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
from react_agent import ReactMLAgent
from strategies import DEFAULT_STRATEGY, get_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_API_KEY_FILE = Path(r"C:/Users/PC4/OneDrive/바탕 화면/개인/개인정보/api_key.txt")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "o4-mini")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "30"))
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


def _parse_config(message: Message) -> dict:
    """Parse arena payload: {strategy, max_iterations, extra_instructions}."""
    text = get_message_text(message)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


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

        config = _parse_config(message)
        strategy_name = config.get("strategy", DEFAULT_STRATEGY)
        strategy_text = get_strategy(strategy_name)
        try:
            iter_override = int(config["max_iterations"])
        except (KeyError, TypeError, ValueError):
            iter_override = None
        extra_instructions = config.get("extra_instructions", "")

        api_key = _load_api_key()
        if not api_key:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: API key required"))],
                name="Error",
            )
            return

        max_iter = iter_override if iter_override else MAX_ITERATIONS

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"ReAct Solver starting (strategy={strategy_name}, model={OPENAI_MODEL}, "
                f"max_steps={max_iter})"
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
            # PHASE 1: Data Profiling (quick, for status reporting)
            # =================================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Phase 1: Data profiling..."),
            )
            try:
                data_profile = await loop.run_in_executor(
                    None, lambda: run_profiler(workdir, timeout=60)
                )
                logger.info("Data profile: %d chars", len(data_profile))
            except Exception:
                data_profile = ""

            # =================================================================
            # PHASE 2: ReAct Agent (main solving engine)
            # =================================================================
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Phase 2: ReAct agent ({max_iter} steps, strategy={strategy_name})..."
                ),
            )

            llm = LLMClient(api_key=api_key, model=OPENAI_MODEL)

            # Build instructions
            instructions = get_message_text(message) or ""
            if extra_instructions:
                instructions = f"{instructions}\n\n{extra_instructions}"

            def on_step(step, total, cv_score):
                try:
                    __import__("asyncio").run_coroutine_threadsafe(
                        updater.update_status(
                            TaskState.working,
                            new_agent_text_message(
                                f"Step {step}/{total}: cv={cv_score}"
                            ),
                        ),
                        loop,
                    )
                except Exception:
                    pass

            react_score = None
            try:
                agent = ReactMLAgent(
                    workdir=workdir,
                    llm=llm,
                    max_iterations=max_iter,
                    code_timeout=CODE_TIMEOUT,
                    exploration_hint=strategy_text,
                )
                result_path = await loop.run_in_executor(
                    None,
                    lambda: agent.run(instructions, on_step=on_step),
                )
                react_score = agent.best_cv_score
                logger.info("ReAct agent done: path=%s cv=%s", result_path, react_score)
            except Exception as exc:
                logger.exception("ReAct agent failed: %s", exc)
                result_path = None

            # Save ReAct submission
            react_submission = workdir / "_react_submission.csv"
            if submission_path.exists():
                patch_submission_columns(submission_path, workdir)
                shutil.copy2(submission_path, react_submission)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"ReAct done: cv={react_score}"),
            )

            # =================================================================
            # PHASE 3: Baseline Safety Net (if ReAct failed)
            # =================================================================
            baseline_score = None
            baseline_submission = workdir / "_baseline_submission.csv"

            if not react_submission.exists():
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message("ReAct failed — running baseline safety net..."),
                )
                try:
                    baseline_code = get_baseline_script()
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
                            logger.info("Baseline safety net: score=%s", baseline_score)
                except Exception as exc:
                    logger.warning("Baseline also failed: %s", exc)

            # =================================================================
            # FINAL: Pick best submission
            # =================================================================
            if react_submission.exists():
                shutil.copy2(react_submission, submission_path)
            elif baseline_submission.exists():
                shutil.copy2(baseline_submission, submission_path)

            if submission_path.exists():
                patch_submission_columns(submission_path, workdir)

            if not submission_path.exists():
                data_dir = workdir / "home" / "data"
                sample_paths = list(data_dir.glob("sample_submission*.csv"))
                if sample_paths:
                    shutil.copy2(sample_paths[0], submission_path)
                    logger.warning("Using sample_submission as last resort")

            if not submission_path.exists():
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: no submission.csv produced"))],
                    name="Error",
                )
                return

            best_score = react_score or baseline_score
            csv_bytes = submission_path.read_bytes()
            b64_out = base64.b64encode(csv_bytes).decode("ascii")

            # Format scores as numbers (Arena regex requires numeric values)
            def _fmt(v):
                return f"{v:.6f}" if v is not None else "-1"

            summary = (
                f"ReAct solver complete: "
                f"best_score={_fmt(best_score)}, "
                f"react_cv={_fmt(react_score)}, "
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
