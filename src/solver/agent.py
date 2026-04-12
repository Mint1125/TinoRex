"""Solver agent: receives competition tar.gz + strategy, runs tree search, returns submission."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from llm import LLMClient
from ml_helpers import patch_submission_columns, validate_submission_report
from strategies import DEFAULT_STRATEGY, get_strategy
from tree import SolutionTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_API_KEY_FILE = Path(r"C:/Users/PC4/OneDrive/바탕 화면/개인/개인정보/api_key.txt")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "o4-mini")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "20"))
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "600"))
REFINE_STEPS = int(os.environ.get("REFINE_STEPS", "8"))


def _load_api_keys() -> dict[str, str]:
    """Load all API keys from file. Returns {provider: key}."""
    keys: dict[str, str] = {}
    # Check env vars first
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
    """Legacy: return the key matching the current model's provider."""
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
    """Extract strategy name from message text JSON, or use default."""
    text = get_message_text(message)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload.get("strategy", DEFAULT_STRATEGY)
    except (json.JSONDecodeError, TypeError):
        pass
    return DEFAULT_STRATEGY


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
                parts=[Part(root=TextPart(text="Error: no competition tar.gz in message"))],
                name="Error",
            )
            return

        strategy_name = _parse_strategy(message)
        strategy_text = get_strategy(strategy_name)

        api_key = _load_api_key()
        if not api_key:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: OPENAI_API_KEY required"))],
                name="Error",
            )
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting tree search (strategy={strategy_name}, model={OPENAI_MODEL}, "
                f"iterations={MAX_ITERATIONS})"
            ),
        )

        with tempfile.TemporaryDirectory(prefix=f"solver-{ctx}-") as tmpdir:
            workdir = Path(tmpdir)

            try:
                _extract_tar_b64(tar_b64, workdir)
            except Exception as exc:
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=f"Error extracting tar: {exc}"))],
                    name="Error",
                )
                return

            llm = LLMClient(api_key=api_key, model=OPENAI_MODEL)
            tree = SolutionTree(
                workdir=workdir,
                llm=llm,
                max_iterations=MAX_ITERATIONS,
                code_timeout=CODE_TIMEOUT,
                strategy_name=strategy_name,
                refine_steps=REFINE_STEPS,
            )

            loop = __import__("asyncio").get_running_loop()

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
                result = await loop.run_in_executor(
                    None,
                    lambda: tree.run(strategy_text, on_node_complete=on_node_complete),
                )
            except Exception as exc:
                logger.exception("Tree search failed")
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=f"Tree search error: {exc}"))],
                    name="Error",
                )
                return

            # Find submission.csv from the best node's execution
            submission_path = workdir / "submission.csv"
            if not submission_path.exists():
                # If best node didn't leave a file, re-run its code
                if result.best_node and result.best_node.code:
                    from interpreter import Interpreter
                    interp = Interpreter(workdir=workdir, timeout=CODE_TIMEOUT)
                    try:
                        interp.run(result.best_node.code)
                    finally:
                        interp.cleanup()

            if not submission_path.exists():
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text="Error: no submission.csv produced"))],
                    name="Error",
                )
                return

            # Final validation and patching
            validation = validate_submission_report(submission_path, workdir)
            if not validation["valid"]:
                logger.warning("Final submission has issues: %s", validation["summary"])
                patch_submission_columns(submission_path, workdir)
                validation = validate_submission_report(submission_path, workdir)
                if not validation["valid"]:
                    logger.warning("Post-patch still has issues: %s", validation["summary"])

            csv_bytes = submission_path.read_bytes()
            b64_out = base64.b64encode(csv_bytes).decode("ascii")

            summary = (
                f"Tree search complete: {len(result.all_nodes)} nodes, "
                f"best_score={result.best_node.cv_score if result.best_node else 'N/A'}, "
                f"total_time={result.total_time:.0f}s, "
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
            logger.info("Submitted: %s bytes, %s", len(csv_bytes), summary)
