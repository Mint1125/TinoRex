"""Solver agent v9: stage-aware single-shot code generation and execution.

Receives {stage, module, context, session_id} from Arena.
Generates one Python script via LLM, executes it, returns structured result.
Workdir is shared across stages via session_id.
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

from interpreter import Interpreter
from llm import LLMClient
from stage_prompts import build_user_prompt, get_system_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_API_KEY_FILE = Path(r"C:/Users/PC4/OneDrive/바탕 화면/개인/개인정보/api_key.txt")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "o4-mini")
CODE_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "300"))
MAX_STDOUT_CHARS = 8000


def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    if _API_KEY_FILE.exists():
        lines = _API_KEY_FILE.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                return line.strip()
    return ""


# ── Session workdir management ────────────────────────────────────────────

_session_workdirs: dict[str, Path] = {}


def _get_session_workdir(session_id: str) -> Path:
    """Get or create a persistent workdir for this session."""
    if session_id not in _session_workdirs:
        workdir = Path(tempfile.mkdtemp(prefix=f"v9-{session_id[:8]}-"))
        _session_workdirs[session_id] = workdir
        logger.info("Created session workdir: %s", workdir)
    return _session_workdirs[session_id]


def _get_module_workdir(session_id: str, stage: str, module: str) -> Path:
    """Get a module-specific subdirectory within the session workdir."""
    base = _get_session_workdir(session_id)
    mod_dir = base / f"{stage}_{module}"
    mod_dir.mkdir(parents=True, exist_ok=True)

    # Symlink or copy the data dir so scripts can find ./home/data/
    data_src = base / "home"
    data_dst = mod_dir / "home"
    if data_src.exists() and not data_dst.exists():
        # On Windows, use junction or copy
        try:
            data_dst.symlink_to(data_src, target_is_directory=True)
        except OSError:
            shutil.copytree(str(data_src), str(data_dst))

    # Copy shared artifacts (parquet, json, npy) from session root
    for ext in ("*.parquet", "*.json", "*.npy", "*.csv"):
        for f in base.glob(ext):
            dst = mod_dir / f.name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))

    return mod_dir


def _promote_artifacts(module_dir: Path, session_dir: Path) -> None:
    """Copy winning module's output artifacts to session root."""
    for ext in ("*.parquet", "*.json", "*.npy", "*.csv"):
        for f in module_dir.glob(ext):
            dst = session_dir / f.name
            shutil.copy2(str(f), str(dst))
    logger.info("Promoted artifacts from %s to session root", module_dir.name)


# ── Tar extraction ────────────────────────────────────────────────────────

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


# ── Result parsing ────────────────────────────────────────────────────────

def _parse_cv_score(stdout: str) -> float | None:
    matches = re.findall(r"CV_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stdout)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def _parse_marker(stdout: str, start_marker: str, end_marker: str) -> str | None:
    if start_marker in stdout and end_marker in stdout:
        s = stdout.index(start_marker) + len(start_marker)
        e = stdout.index(end_marker)
        return stdout[s:e].strip()
    return None


def _parse_json_marker(stdout: str, marker: str) -> str | None:
    """Parse a line like MARKER={...json...}"""
    pattern = rf"{marker}\s*=\s*(\{{.*?\}}|\[.*?\])"
    match = re.search(pattern, stdout, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _parse_model_results(stdout: str) -> list[dict]:
    results = []
    for match in re.finditer(r"MODEL_RESULT\s*=\s*(\{.*?\})", stdout):
        try:
            results.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    return results


# ── File listing ──────────────────────────────────────────────────────────

def _list_files(workdir: Path) -> str:
    data_dir = workdir / "home" / "data"
    if not data_dir.exists():
        data_dir = workdir
    entries = []
    for p in sorted(data_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(workdir)
            size_mb = p.stat().st_size / (1024 * 1024)
            entries.append(f"  ./{rel}  ({size_mb:.1f} MB)")
    return "\n".join(entries) if entries else "  <no files found>"


def _read_description(workdir: Path) -> str:
    for name in ("description.md", "description.txt", "README.md"):
        path = workdir / "home" / "data" / name
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > 12000:
                text = text[:12000] + "\n... (truncated)"
            return text
    return "<no description file found>"


# ── Agent ─────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self):
        self._done_contexts: set[str] = set()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        ctx = message.context_id or "default"

        # Parse payload
        text = get_message_text(message)
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: invalid JSON payload"))],
                name="Error",
            )
            return

        stage = payload.get("stage", "")
        module = payload.get("module", "A")
        context = payload.get("context", {})
        session_id = payload.get("session_id", ctx)
        extra = payload.get("extra", {})

        # Handle promote requests (copy winning module's artifacts to session root)
        if stage == "promote":
            promote_stage = extra.get("promote_stage", "")
            session_dir = _get_session_workdir(session_id)
            mod_dir = session_dir / f"{promote_stage}_{module}"
            if mod_dir.exists():
                _promote_artifacts(mod_dir, session_dir)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=json.dumps({"promoted": True})))],
                name="promote_result",
            )
            return

        # Extract tar if present and not yet extracted
        session_dir = _get_session_workdir(session_id)
        data_dir = session_dir / "home"
        if not data_dir.exists():
            tar_b64 = _first_tar_from_message(message)
            if tar_b64:
                _extract_tar_b64(tar_b64, session_dir)
                logger.info("Extracted tar to %s", session_dir)

        # Setup
        api_key = _load_api_key()
        if not api_key:
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: OPENAI_API_KEY required"))],
                name="Error",
            )
            return

        description = _read_description(session_dir)
        file_listing = _list_files(session_dir)

        # Build prompt
        system_prompt = get_system_prompt(stage)
        user_prompt = build_user_prompt(
            stage=stage,
            module=module,
            context=context,
            description=description,
            file_listing=file_listing,
            extra=extra,
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[Solver] Stage={stage} Module={module}: generating code (model={OPENAI_MODEL})"
            ),
        )

        # Generate code
        llm = LLMClient(api_key=api_key, model=OPENAI_MODEL)
        code = llm.generate_code(system=system_prompt, user=user_prompt, temperature=1.0)

        # Execute in module-specific subdirectory
        mod_dir = _get_module_workdir(session_id, stage, module)
        interp = Interpreter(workdir=mod_dir, timeout=CODE_TIMEOUT)
        result = interp.run(code)

        stdout = result.stdout
        if len(stdout) > MAX_STDOUT_CHARS:
            stdout = stdout[:MAX_STDOUT_CHARS] + "\n... (truncated)"

        # If failed, try one repair
        if not result.succeeded:
            error_tail = stdout[-2000:] if stdout else "(no output)"
            logger.warning(
                "[%s/%s] First attempt failed: %s\n--- error tail ---\n%s",
                stage, module, result.exc_type, error_tail[-500:]
            )
            repair_prompt = (
                f"The previous code FAILED. Here is the error output:\n"
                f"```\n{error_tail}\n```\n\n"
                f"IMPORTANT REMINDERS:\n"
                f"- Data files are in ./home/data/ (train.csv, test.csv, etc.)\n"
                f"- Save output files to ./ (current directory), NOT ./home/data/\n"
                f"- Include all imports at the top\n"
                f"- Wrap main logic in try/except\n\n"
                f"Fix the error and return the COMPLETE corrected script.\n\n"
                f"Original task:\n{user_prompt}"
            )
            code = llm.generate_code(system=system_prompt, user=repair_prompt, temperature=1.0)
            result = interp.run(code)
            stdout = result.stdout
            if len(stdout) > MAX_STDOUT_CHARS:
                stdout = stdout[:MAX_STDOUT_CHARS] + "\n... (truncated)"

        # Build structured response
        response: dict = {
            "stage": stage,
            "module": module,
            "succeeded": result.succeeded,
            "exec_time": result.exec_time,
            "stdout": stdout[-2000:],  # keep last 2000 chars for arena
        }

        # Parse stage-specific outputs
        cv_score = _parse_cv_score(stdout)
        if cv_score is not None:
            response["cv_score"] = cv_score

        if stage == "eda":
            eda_json = _parse_marker(stdout, "EDA_REPORT_START", "EDA_REPORT_END")
            if eda_json:
                response["eda_report"] = eda_json

        elif stage == "model_selection":
            response["model_results"] = _parse_model_results(stdout)

        elif stage == "optuna":
            best_params = _parse_json_marker(stdout, "BEST_PARAMS")
            best_score_match = re.search(
                r"BEST_SCORE\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stdout
            )
            if best_params:
                response["best_params"] = best_params
            if best_score_match:
                response["best_score"] = float(best_score_match.group(1))

        elif stage == "threshold":
            thresh_match = re.search(
                r"THRESHOLD\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", stdout
            )
            if thresh_match:
                response["threshold"] = float(thresh_match.group(1))

        # Check for submission.csv
        submission_path = mod_dir / "submission.csv"
        if submission_path.exists():
            csv_bytes = submission_path.read_bytes()
            response["submission_csv_b64"] = base64.b64encode(csv_bytes).decode("ascii")

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"[Solver] Stage={stage} Module={module}: "
                f"{'OK' if result.succeeded else 'FAILED'} "
                f"cv={cv_score} time={result.exec_time:.0f}s"
            ),
        )

        # Return result as artifact
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps(response)))],
            name=f"{stage}_{module}_result",
        )

        logger.info(
            "[%s/%s] Done: succeeded=%s cv=%s time=%.0fs",
            stage, module, result.succeeded, cv_score, result.exec_time,
        )


def promote_winner(session_id: str, stage: str, winning_module: str) -> None:
    """Called by arena to promote winning module's artifacts to session root.

    This is also exposed as a utility so the arena can call it via a special
    'promote' stage request.
    """
    session_dir = _get_session_workdir(session_id)
    mod_dir = session_dir / f"{stage}_{winning_module}"
    if mod_dir.exists():
        _promote_artifacts(mod_dir, session_dir)
