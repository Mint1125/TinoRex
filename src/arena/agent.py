"""Arena agent v9: dispatches competition to solver with multiple strategies.

Fans out to solver with different strategy configs, collects submissions,
returns the best one. Each strategy controls the balance between deterministic
baseline, tree search depth, and refinement intensity.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    FilePart,
    FileWithBytes,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils import get_message_text, new_agent_text_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOLVER_URL = os.environ.get("SOLVER_URL", "http://127.0.0.1:8001/")
STRATEGIES = os.environ.get("STRATEGIES", "quick_baseline,data_first,big_model").split(",")
MAX_CONCURRENT_SOLVERS = int(os.environ.get("MAX_CONCURRENT_SOLVERS", "2"))


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


async def _run_solver(
    solver_url: str,
    strategy: str,
    instructions_text: str,
    tar_b64: str,
) -> dict | None:
    """Send a competition to the solver with a specific strategy."""
    try:
        async with httpx.AsyncClient(timeout=7200) as hc:
            resolver = A2ACardResolver(httpx_client=hc, base_url=solver_url)
            agent_card = await resolver.get_agent_card()
            config = ClientConfig(httpx_client=hc, streaming=True)
            factory = ClientFactory(config)
            client = factory.create(agent_card)

            payload = json.dumps({"strategy": strategy})
            msg = Message(
                kind="message",
                role=Role.user,
                parts=[
                    Part(root=TextPart(text=payload)),
                    Part(root=FilePart(
                        file=FileWithBytes(
                            bytes=tar_b64,
                            name="competition.tar.gz",
                            mime_type="application/gzip",
                        )
                    )),
                ],
                message_id=uuid4().hex,
            )

            submission_csv_b64: str | None = None
            summary_text: str = ""

            async for event in client.send_message(msg):
                match event:
                    case (task, TaskArtifactUpdateEvent()):
                        for artifact in task.artifacts or []:
                            for part in artifact.parts:
                                if isinstance(part.root, FilePart):
                                    fd = part.root.file
                                    if isinstance(fd, FileWithBytes) and fd.bytes:
                                        raw = fd.bytes
                                        if isinstance(raw, (bytes, bytearray)):
                                            submission_csv_b64 = base64.b64encode(raw).decode("ascii")
                                        else:
                                            submission_csv_b64 = raw
                                elif isinstance(part.root, TextPart):
                                    summary_text = part.root.text
                    case _:
                        pass

            if submission_csv_b64:
                return {
                    "strategy": strategy,
                    "csv_b64": submission_csv_b64,
                    "summary": summary_text,
                }
            return None

    except Exception as exc:
        logger.error("Solver attempt (strategy=%s) failed: %s", strategy, exc)
        return None


def _extract_best_score(summary: str) -> float:
    """Parse best_score from the solver summary text."""
    match = re.search(r"best_score=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", summary)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return -1e9


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

        instructions_text = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Arena v9: launching {len(STRATEGIES)} solver attempts "
                f"(strategies: {STRATEGIES})"
            ),
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SOLVERS)

        async def _limited_run(strategy: str) -> dict | None:
            async with semaphore:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Arena: starting solver (strategy={strategy})"),
                )
                return await _run_solver(SOLVER_URL, strategy, instructions_text, tar_b64)

        tasks = [_limited_run(strategy) for strategy in STRATEGIES]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r is not None]
        logger.info("Arena: %d/%d attempts succeeded", len(successful), len(STRATEGIES))

        if not successful:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: all solver attempts failed"))],
                name="Error",
            )
            return

        best = max(successful, key=lambda r: _extract_best_score(r["summary"]))

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Arena: best strategy={best['strategy']} "
                f"(score={_extract_best_score(best['summary'])}), "
                f"{len(successful)}/{len(STRATEGIES)} succeeded"
            ),
        )

        await updater.add_artifact(
            parts=[
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=best["csv_b64"],
                            name="submission.csv",
                            mime_type="text/csv",
                        )
                    )
                )
            ],
            name="submission",
        )
        self._done_contexts.add(ctx)
        logger.info("Arena: submitted best result (strategy=%s)", best["strategy"])
