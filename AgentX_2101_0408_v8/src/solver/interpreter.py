"""Execute Python scripts in isolated subprocess — Windows-safe."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

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


class Interpreter:
    def __init__(self, workdir: str | Path, timeout: int = 600):
        self.working_dir = str(Path(workdir).resolve())
        self.timeout = timeout

    def run(self, code: str) -> ExecutionResult:
        """Execute code in a fresh subprocess.  Windows-safe."""
        script = Path(self.working_dir) / "_solver_run.py"
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
            exc_type = None if proc.returncode == 0 else "RuntimeError"
            return ExecutionResult(stdout=stdout, exec_time=exec_time, exc_type=exc_type)

        except subprocess.TimeoutExpired:
            exec_time = time.time() - start
            return ExecutionResult(
                stdout=f"TimeoutError: exceeded {self.timeout}s limit",
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
