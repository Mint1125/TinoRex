"""Arena v10: Explore → Select → Refine pipeline.

The arena is the brain. It orchestrates:
  Phase 1 (Exploration): Fan out N short solver runs with diverse hints
  Phase 2 (Selection):   Pick the best by CV score
  Phase 3 (Refinement):  Deep optimization on the winner's strategy

The solver is the muscle — it just runs a ReAct agent with the given config.
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

# Pipeline configuration
EXPLORATION_BRANCHES = int(os.environ.get("EXPLORATION_BRANCHES", "3"))
EXPLORATION_STEPS = int(os.environ.get("EXPLORATION_STEPS", "10"))
REFINEMENT_STEPS = int(os.environ.get("REFINEMENT_STEPS", "20"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_SOLVERS", "2"))

# Diverse exploration hints (structural pass@k)
EXPLORATION_HINTS = [
    (
        "quick_baseline",
        "Start with the SIMPLEST robust baseline: minimal preprocessing, one strong "
        "default model (e.g. LogisticRegression or GradientBoosting), valid submission first.",
    ),
    (
        "data_first",
        "Prioritize EDA: inspect missing values, target balance, feature types; then "
        "model in a way that directly addresses what you found.",
    ),
    (
        "big_model",
        "Prioritize model capacity: stronger boosting (XGBoost/LightGBM with tuned "
        "hyperparameters) or careful feature interactions after a quick baseline check.",
    ),
]


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


async def _call_solver(
    solver_url: str,
    tar_b64: str,
    strategy: str,
    max_iterations: int,
    extra_instructions: str = "",
) -> dict | None:
    """Call solver with specific config. Returns {strategy, csv_b64, summary, cv_score} or None."""
    try:
        async with httpx.AsyncClient(timeout=7200) as hc:
            resolver = A2ACardResolver(httpx_client=hc, base_url=solver_url)
            agent_card = await resolver.get_agent_card()
            config = ClientConfig(httpx_client=hc, streaming=True)
            factory = ClientFactory(config)
            client = factory.create(agent_card)

            payload = json.dumps({
                "strategy": strategy,
                "max_iterations": max_iterations,
                "extra_instructions": extra_instructions,
            })
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
                cv_score = _extract_cv_score(summary_text)
                return {
                    "strategy": strategy,
                    "csv_b64": submission_csv_b64,
                    "summary": summary_text,
                    "cv_score": cv_score,
                }
            return None

    except Exception as exc:
        logger.error("Solver call failed (strategy=%s): %s", strategy, exc)
        return None


def _extract_cv_score(summary: str) -> float:
    """Parse best_score or react_cv from solver summary."""
    for pattern in [
        r"best_score=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        r"react_cv=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    ]:
        match = re.search(pattern, summary)
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

        n_branches = min(EXPLORATION_BRANCHES, len(EXPLORATION_HINTS))

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Arena pipeline: {n_branches} exploration branches × {EXPLORATION_STEPS} steps, "
                f"then {REFINEMENT_STEPS} refinement steps"
            ),
        )

        # =================================================================
        # PHASE 1: EXPLORATION — diverse short runs
        # =================================================================
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Phase 1: Exploration ({n_branches} branches, {EXPLORATION_STEPS} steps each)..."
            ),
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def _explore(idx: int) -> dict | None:
            name, hint = EXPLORATION_HINTS[idx % len(EXPLORATION_HINTS)]
            async with semaphore:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Exploration {idx+1}/{n_branches}: {name}"),
                )
                return await _call_solver(
                    SOLVER_URL, tar_b64,
                    strategy=name,
                    max_iterations=EXPLORATION_STEPS,
                )

        explore_tasks = [_explore(i) for i in range(n_branches)]
        explore_results = await asyncio.gather(*explore_tasks)

        # =================================================================
        # PHASE 2: SELECTION — pick best exploration
        # =================================================================
        successful = [r for r in explore_results if r is not None]
        logger.info(
            "Exploration: %d/%d succeeded", len(successful), n_branches
        )

        if not successful:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: all exploration branches failed"))],
                name="Error",
            )
            return

        winner = max(successful, key=lambda r: r["cv_score"])
        winner_strategy = winner["strategy"]
        winner_cv = winner["cv_score"]
        cv_display = f"{winner_cv:.4f}" if winner_cv > -1e8 else "unknown"

        all_scores = ", ".join(
            f"{r['strategy']}={r['cv_score']:.4f}" if r['cv_score'] > -1e8
            else f"{r['strategy']}=N/A"
            for r in successful
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Phase 2: Selected '{winner_strategy}' (CV≈{cv_display}). "
                f"All scores: [{all_scores}]"
            ),
        )

        # =================================================================
        # PHASE 3: REFINEMENT — deep optimization on winner's strategy
        # =================================================================
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Phase 3: Refinement ({REFINEMENT_STEPS} steps) on '{winner_strategy}'..."
            ),
        )

        refine_extra = (
            f"[Refinement phase] The best exploration branch used strategy '{winner_strategy}' "
            f"and achieved approx CV_SCORE={cv_display}. "
            f"Build on this approach and improve: try ensembling, hyperparameter tuning, "
            f"better feature engineering, or calibration. "
            f"Try to beat {cv_display}!"
        )

        refine_result = await _call_solver(
            SOLVER_URL, tar_b64,
            strategy=winner_strategy,
            max_iterations=REFINEMENT_STEPS,
            extra_instructions=refine_extra,
        )

        # =================================================================
        # FINAL: Pick best between exploration winner and refinement
        # =================================================================
        best = winner  # default to exploration winner

        if refine_result is not None:
            refine_cv = refine_result["cv_score"]
            logger.info(
                "Refinement: cv=%s (exploration winner was %s)",
                refine_cv, winner_cv,
            )
            if refine_cv > winner_cv:
                best = refine_result
                logger.info("Refinement improved: %s → %s", winner_cv, refine_cv)
            else:
                logger.info("Refinement did not improve, keeping exploration winner")
        else:
            logger.warning("Refinement failed, keeping exploration winner")

        final_cv = best["cv_score"]
        final_strategy = best["strategy"]
        is_refined = best is refine_result

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Final: strategy={final_strategy}, cv={final_cv:.4f}, "
                f"refined={'yes' if is_refined else 'no'}"
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
        logger.info(
            "Arena submitted: strategy=%s cv=%s refined=%s scores=[%s]",
            final_strategy, final_cv, is_refined, all_scores,
        )
