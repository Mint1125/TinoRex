"""Arena agent v9: 6-stage pipeline with 3-module debate at each stage.

Flow:
  1. EDA Profiling       — 3 modules → LLM synthesis → eda_report
  2. Feature Engineering — 3 modules → CV comparison → feature_context
  3. Model Selection     — 3 modules → CV ranking + LLM diversity → top 3 models
  4. Optuna Tuning       — 3 modules (one per model) → collect params
  5. Stacking Ensemble   — 3 modules → CV comparison → submission.csv
  6. Threshold Opt       — 3 modules → CV comparison → final submission.csv
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
    TextPart,
)
from a2a.utils import get_message_text, new_agent_text_message

from llm import ArenaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOLVER_URL = os.environ.get("SOLVER_URL", "http://127.0.0.1:8001/")

# Stage-specific synthesis prompts (imported inline to avoid circular deps)
EDA_SYNTHESIS_PROMPT = None  # loaded lazily


def _get_eda_synthesis_prompt() -> str:
    """Load EDA synthesis prompt template from solver's stage_prompts."""
    global EDA_SYNTHESIS_PROMPT
    if EDA_SYNTHESIS_PROMPT is None:
        # We store it here directly to avoid cross-service imports
        EDA_SYNTHESIS_PROMPT = """\
You are given 3 EDA reports from independent analysts examining the same dataset.
Merge them into ONE comprehensive data understanding document.

Rules:
- Include ALL unique findings from all 3 reports.
- Resolve contradictions by preferring the more specific/detailed finding.
- Produce a single JSON object.
- Keep total output under 3000 tokens.

Required JSON keys:
- task_type: "binary_classification" | "multiclass_classification" | "regression"
- target_col: string
- id_col: string
- eval_metric: string or null
- n_train: int
- n_test: int
- columns: list of {name, dtype, missing_rate, cardinality, role, notes}
- target_distribution: dict
- top_correlations: list of {feature, correlation}
- quality_issues: list of {column, issue}
- feature_opportunities: list of {description, columns_involved, type}

Report A:
{report_a}

Report B:
{report_b}

Report C:
{report_c}

Output ONLY the merged JSON (no markdown fences, no explanation):
"""
    return EDA_SYNTHESIS_PROMPT


MODEL_SYNTHESIS_PROMPT = """\
You are selecting the top 3 models for a stacking ensemble.

Model results from 3 independent tests:
{all_model_results}

Task type: {task_type}

Select exactly 3 models for the ensemble. Priorities:
1. Highest CV scores
2. Model diversity — if the top 3 are all from the same family, \
replace the weakest with the best from a different family.
3. Reasonable training time

Output a JSON list of 3 objects: [{{"name": "...", "cv_score": ..., "reason": "..."}}]
Output ONLY the JSON (no markdown fences, no explanation):
"""


# ── Solver communication ──────────────────────────────────────────────────

async def _call_solver(
    solver_url: str,
    payload: dict,
    tar_b64: str | None = None,
) -> dict | None:
    """Send a stage/module request to the solver and collect the result."""
    try:
        async with httpx.AsyncClient(timeout=7200) as hc:
            resolver = A2ACardResolver(httpx_client=hc, base_url=solver_url)
            agent_card = await resolver.get_agent_card()
            config = ClientConfig(httpx_client=hc, streaming=True)
            factory = ClientFactory(config)
            client = factory.create(agent_card)

            parts = [Part(root=TextPart(text=json.dumps(payload)))]
            if tar_b64:
                parts.append(
                    Part(root=FilePart(
                        file=FileWithBytes(
                            bytes=tar_b64,
                            name="competition.tar.gz",
                            mime_type="application/gzip",
                        )
                    ))
                )

            msg = Message(
                kind="message",
                role=Role.user,
                parts=parts,
                message_id=uuid4().hex,
            )

            result_text = ""
            async for event in client.send_message(msg):
                match event:
                    case (task, TaskArtifactUpdateEvent()):
                        for artifact in task.artifacts or []:
                            for part in artifact.parts:
                                if isinstance(part.root, TextPart):
                                    result_text = part.root.text
                    case _:
                        pass

            if result_text:
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    logger.warning("Solver returned non-JSON: %s", result_text[:200])
                    return None
            return None

    except Exception as exc:
        logger.error("Solver call failed (stage=%s, module=%s): %s",
                      payload.get("stage"), payload.get("module"), exc)
        return None


async def _run_stage_modules(
    solver_url: str,
    stage: str,
    session_id: str,
    context: dict,
    tar_b64: str | None,
    modules: list[str] = ["A", "B", "C"],
    extra_per_module: dict[str, dict] | None = None,
) -> list[dict | None]:
    """Fan out module calls to solver in parallel."""
    tasks = []
    for mod in modules:
        payload = {
            "stage": stage,
            "module": mod,
            "context": context,
            "session_id": session_id,
        }
        if extra_per_module and mod in extra_per_module:
            payload["extra"] = extra_per_module[mod]
        # Only send tar on first stage call (solver caches it by session)
        tasks.append(_call_solver(solver_url, payload, tar_b64))

    return await asyncio.gather(*tasks)


def _pick_best_by_cv(results: list[dict | None]) -> tuple[dict | None, str]:
    """Pick the result with the highest cv_score. Returns (result, module_letter)."""
    best = None
    best_module = "A"
    best_cv = -1e9
    for r in results:
        if r and r.get("succeeded") and r.get("cv_score") is not None:
            if r["cv_score"] > best_cv:
                best_cv = r["cv_score"]
                best = r
                best_module = r.get("module", "?")
    return best, best_module


# ── Pipeline ──────────────────────────────────────────────────────────────

class Agent:
    def __init__(self):
        self._done_contexts: set[str] = set()
        self._llm = ArenaLLM()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        ctx = message.context_id or "default"
        if ctx in self._done_contexts:
            return

        # Extract tar from message
        tar_b64 = _first_tar_from_message(message)
        if not tar_b64:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: no competition tar.gz"))],
                name="Error",
            )
            return

        session_id = uuid4().hex[:12]
        accumulated: dict = {}

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[Arena v9] Starting 6-stage debate pipeline (session={session_id})"
            ),
        )

        # ══════════════════════════════════════════════════════════════════
        # Stage 1: EDA Profiling
        # ══════════════════════════════════════════════════════════════════
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Stage 1/6] EDA Profiling — 3 modules running..."),
        )

        eda_results = await _run_stage_modules(
            SOLVER_URL, "eda", session_id, accumulated, tar_b64
        )

        # Collect EDA reports
        reports = {}
        for r in eda_results:
            if r and r.get("succeeded") and r.get("eda_report"):
                reports[r["module"]] = r["eda_report"]

        if not reports:
            # Fallback: use any stdout as raw report
            for r in eda_results:
                if r and r.get("stdout"):
                    reports[r["module"]] = r["stdout"][:2000]

        if len(reports) >= 2:
            # Synthesize via LLM
            rpt_a = reports.get("A", "(unavailable)")
            rpt_b = reports.get("B", "(unavailable)")
            rpt_c = reports.get("C", "(unavailable)")
            synthesis_prompt = _get_eda_synthesis_prompt().format(
                report_a=rpt_a, report_b=rpt_b, report_c=rpt_c
            )
            eda_report = self._llm.synthesize(
                system="You are a data analyst merging EDA reports into one JSON.",
                user=synthesis_prompt,
            )
        elif reports:
            eda_report = list(reports.values())[0]
        else:
            eda_report = '{"error": "All EDA modules failed"}'

        accumulated["eda_report"] = eda_report
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[Stage 1/6] EDA complete — {len(reports)}/3 modules succeeded"
            ),
        )

        # ══════════════════════════════════════════════════════════════════
        # Stage 2: Feature Engineering
        # ══════════════════════════════════════════════════════════════════
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Stage 2/6] Feature Engineering — 3 modules running..."),
        )

        fe_results = await _run_stage_modules(
            SOLVER_URL, "features", session_id, accumulated, tar_b64
        )

        best_fe, best_fe_module = _pick_best_by_cv(fe_results)
        if best_fe:
            # Tell solver to promote winning module's artifacts
            await _call_solver(SOLVER_URL, {
                "stage": "promote",
                "module": best_fe_module,
                "context": {},
                "session_id": session_id,
                "extra": {"promote_stage": "features"},
            })

            accumulated["feature_context"] = json.dumps({
                "best_cv_score": best_fe.get("cv_score"),
                "winning_module": best_fe_module,
            })
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[Stage 2/6] Features complete — best CV={best_fe.get('cv_score')} "
                    f"(Module {best_fe_module})"
                ),
            )
        else:
            accumulated["feature_context"] = '{"error": "All FE modules failed"}'
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("[Stage 2/6] Features — all modules failed, continuing..."),
            )

        # ══════════════════════════════════════════════════════════════════
        # Stage 3: Model Selection
        # ══════════════════════════════════════════════════════════════════
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Stage 3/6] Model Selection — 3 modules running..."),
        )

        model_results = await _run_stage_modules(
            SOLVER_URL, "model_selection", session_id, accumulated, tar_b64
        )

        # Collect all MODEL_RESULT entries
        all_model_entries = []
        for r in model_results:
            if r and r.get("model_results"):
                all_model_entries.extend(r["model_results"])

        # Determine task type from EDA
        task_type = "binary_classification"
        try:
            eda_json = json.loads(eda_report)
            task_type = eda_json.get("task_type", task_type)
        except (json.JSONDecodeError, TypeError):
            pass

        if len(all_model_entries) >= 3:
            # LLM synthesis to pick top 3 diverse models
            synthesis_prompt = MODEL_SYNTHESIS_PROMPT.format(
                all_model_results=json.dumps(all_model_entries, indent=2),
                task_type=task_type,
            )
            selected_json = self._llm.synthesize(
                system="You are an ML model selection expert.",
                user=synthesis_prompt,
            )
            accumulated["model_selection_context"] = selected_json
        elif all_model_entries:
            # Just take what we have
            sorted_models = sorted(
                all_model_entries, key=lambda x: x.get("cv_score", 0), reverse=True
            )[:3]
            accumulated["model_selection_context"] = json.dumps(sorted_models)
        else:
            # Fallback: use default models
            accumulated["model_selection_context"] = json.dumps([
                {"name": "LGBMClassifier", "cv_score": 0, "reason": "default fallback"},
                {"name": "XGBClassifier", "cv_score": 0, "reason": "default fallback"},
                {"name": "CatBoostClassifier", "cv_score": 0, "reason": "default fallback"},
            ])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[Stage 3/6] Model Selection complete — "
                f"{len(all_model_entries)} models tested, top 3 selected"
            ),
        )

        # ══════════════════════════════════════════════════════════════════
        # Stage 4: Optuna Hyperparameter Tuning
        # ══════════════════════════════════════════════════════════════════
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Stage 4/6] Optuna Tuning — 3 models in parallel..."),
        )

        # Parse selected models
        try:
            selected_models = json.loads(accumulated["model_selection_context"])
        except (json.JSONDecodeError, TypeError):
            selected_models = [
                {"name": "LGBMClassifier"},
                {"name": "XGBClassifier"},
                {"name": "CatBoostClassifier"},
            ]

        # Parse EDA stats for Optuna prompts
        n_rows, n_features = 10000, 20
        try:
            eda_j = json.loads(eda_report)
            n_rows = eda_j.get("n_train", n_rows)
            n_features = len(eda_j.get("columns", [])) or n_features
        except (json.JSONDecodeError, TypeError):
            pass

        n_trials = 50
        extra_per_module = {}
        module_letters = ["A", "B", "C"]
        for i, m in enumerate(selected_models[:3]):
            letter = module_letters[i]
            extra_per_module[letter] = {
                "model_name": m.get("name", "LGBMClassifier"),
                "current_cv": m.get("cv_score", 0),
                "n_rows": n_rows,
                "n_features": n_features,
                "task_type": task_type,
                "n_trials": n_trials,
            }

        optuna_results = await _run_stage_modules(
            SOLVER_URL, "optuna", session_id, accumulated, tar_b64,
            extra_per_module=extra_per_module,
        )

        # Collect tuning results
        tuning_entries = []
        for r in optuna_results:
            if r and r.get("succeeded"):
                entry = {
                    "module": r.get("module"),
                    "best_params": r.get("best_params", "{}"),
                    "best_score": r.get("best_score", r.get("cv_score")),
                }
                # Map module to model name
                idx = module_letters.index(r["module"]) if r.get("module") in module_letters else 0
                if idx < len(selected_models):
                    entry["model_name"] = selected_models[idx].get("name", "Unknown")
                tuning_entries.append(entry)

        # Promote all Optuna outputs (each module saved best_params_*.json)
        for r in optuna_results:
            if r and r.get("succeeded"):
                await _call_solver(SOLVER_URL, {
                    "stage": "promote",
                    "module": r["module"],
                    "context": {},
                    "session_id": session_id,
                    "extra": {"promote_stage": "optuna"},
                })

        accumulated["tuning_context"] = json.dumps(tuning_entries)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[Stage 4/6] Optuna complete — "
                f"{len(tuning_entries)}/3 models tuned"
            ),
        )

        # ══════════════════════════════════════════════════════════════════
        # Stage 5: Stacking Ensemble
        # ══════════════════════════════════════════════════════════════════
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("[Stage 5/6] Stacking Ensemble — 3 meta-learner strategies..."),
        )

        stacking_results = await _run_stage_modules(
            SOLVER_URL, "stacking", session_id, accumulated, tar_b64
        )

        best_stack, best_stack_module = _pick_best_by_cv(stacking_results)
        stacking_cv = None
        best_submission_b64 = None

        if best_stack:
            stacking_cv = best_stack.get("cv_score")
            best_submission_b64 = best_stack.get("submission_csv_b64")

            # Promote stacking winner
            await _call_solver(SOLVER_URL, {
                "stage": "promote",
                "module": best_stack_module,
                "context": {},
                "session_id": session_id,
                "extra": {"promote_stage": "stacking"},
            })

            accumulated["stacking_context"] = json.dumps({
                "stacking_cv": stacking_cv,
                "meta_learner": best_stack_module,
            })
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[Stage 5/6] Stacking complete — CV={stacking_cv} "
                    f"(Module {best_stack_module})"
                ),
            )
        else:
            accumulated["stacking_context"] = '{"error": "All stacking modules failed"}'
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("[Stage 5/6] Stacking — all modules failed"),
            )

        # ══════════════════════════════════════════════════════════════════
        # Stage 6: Threshold Optimization (classification only)
        # ══════════════════════════════════════════════════════════════════
        if task_type in ("binary_classification", "multiclass_classification") and stacking_cv:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("[Stage 6/6] Threshold Optimization — 3 strategies..."),
            )

            thresh_results = await _run_stage_modules(
                SOLVER_URL, "threshold", session_id, accumulated, tar_b64
            )

            best_thresh, best_thresh_module = _pick_best_by_cv(thresh_results)

            if best_thresh and best_thresh.get("cv_score", 0) > (stacking_cv or 0):
                best_submission_b64 = best_thresh.get("submission_csv_b64", best_submission_b64)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"[Stage 6/6] Threshold improved: "
                        f"{stacking_cv} → {best_thresh['cv_score']} "
                        f"(threshold={best_thresh.get('threshold')})"
                    ),
                )
            else:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        "[Stage 6/6] Threshold did not improve — using stacking result"
                    ),
                )
        else:
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("[Stage 6/6] Skipped (not classification or no stacking result)"),
            )

        # ══════════════════════════════════════════════════════════════════
        # Return final submission
        # ══════════════════════════════════════════════════════════════════
        if not best_submission_b64:
            # Last resort: try to find any submission from any stage
            for stage_results in [stacking_results, fe_results, model_results]:
                for r in (stage_results or []):
                    if r and r.get("submission_csv_b64"):
                        best_submission_b64 = r["submission_csv_b64"]
                        break
                if best_submission_b64:
                    break

        if not best_submission_b64:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: no submission.csv produced"))],
                name="Error",
            )
            return

        await updater.add_artifact(
            parts=[
                Part(root=FilePart(
                    file=FileWithBytes(
                        bytes=best_submission_b64,
                        name="submission.csv",
                        mime_type="text/csv",
                    )
                ))
            ],
            name="submission",
        )
        self._done_contexts.add(ctx)
        logger.info("[Arena v9] Pipeline complete (session=%s)", session_id)


# ── Helpers ───────────────────────────────────────────────────────────────

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
