"""Execute Python scripts in isolated subprocess -- Windows-safe."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    stdout: str
    exec_time: float
    exc_type: str | None = None

    @property
    def timed_out(self) -> bool:
        return self.exc_type == "TimeoutError"

    @property
    def succeeded(self) -> bool:
        return self.exc_type is None


def _classify_error(returncode: int, stderr: str) -> str:
    """Classify error type from return code and stderr for better LLM feedback."""
    stderr_lower = stderr.lower()
    if "modulenotfounderror" in stderr_lower or "no module named" in stderr_lower:
        return "ImportError"
    if "memoryerror" in stderr_lower or "killed" in stderr_lower:
        return "MemoryError"
    if "syntaxerror" in stderr_lower:
        return "SyntaxError"
    if "keyerror" in stderr_lower:
        return "KeyError"
    if "filenotfounderror" in stderr_lower:
        return "FileNotFoundError"
    if "valueerror" in stderr_lower:
        return "ValueError"
    if "typeerror" in stderr_lower:
        return "TypeError"
    if "indexerror" in stderr_lower:
        return "IndexError"
    return "RuntimeError"


class Interpreter:
    def __init__(self, workdir: str | Path, timeout: int = 600):
        self.working_dir = str(Path(workdir).resolve())
        self.timeout = timeout

    def run(self, code: str) -> ExecutionResult:
        """Execute code in a fresh subprocess. Uses unique filename to avoid race conditions."""
        script = Path(self.working_dir) / f"_solver_run_{uuid4().hex[:8]}.py"
        try:
            script.write_text(code, encoding="utf-8")
        except Exception as exc:
            return ExecutionResult(
                stdout=f"Failed to write script: {exc}",
                exec_time=0.0,
                exc_type="WriteError",
            )

        start = time.time()
        try:
            proc = subprocess.run(
                ["python", str(script)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
                encoding="utf-8",
                errors="replace",
            )
            exec_time = time.time() - start
            stdout = proc.stdout
            if proc.stderr:
                stdout += "\n--- stderr ---\n" + proc.stderr

            if proc.returncode == 0:
                return ExecutionResult(stdout=stdout, exec_time=exec_time, exc_type=None)

            exc_type = _classify_error(proc.returncode, proc.stderr)
            return ExecutionResult(stdout=stdout, exec_time=exec_time, exc_type=exc_type)

        except subprocess.TimeoutExpired as e:
            exec_time = time.time() - start
            partial = ""
            if e.stdout:
                partial = e.stdout if isinstance(e.stdout, str) else e.stdout.decode("utf-8", errors="replace")
            return ExecutionResult(
                stdout=f"{partial}\nTimeoutError: execution exceeded {self.timeout}s time limit",
                exec_time=exec_time,
                exc_type="TimeoutError",
            )
        except Exception as exc:
            exec_time = time.time() - start
            return ExecutionResult(
                stdout=f"Subprocess error: {exc}",
                exec_time=exec_time,
                exc_type="SubprocessError",
            )
        finally:
            try:
                script.unlink(missing_ok=True)
            except Exception:
                pass

    def cleanup(self) -> None:
        pass
