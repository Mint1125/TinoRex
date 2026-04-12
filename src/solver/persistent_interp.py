"""Persistent Python session interpreter — variables survive across calls.

Uses multiprocessing Queues for cross-platform (Windows/Linux) compatibility.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SessionResult:
    output: str
    exec_time: float
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class _StdoutRedirect:
    """Redirect stdout/stderr to a multiprocessing Queue."""
    def __init__(self, q: Queue):
        self._q = q

    def write(self, text: str) -> None:
        try:
            self._q.put_nowait(text)
        except Exception:
            pass

    def flush(self) -> None:
        pass


def _session_worker(workdir: str, code_q: Queue, output_q: Queue, event_q: Queue) -> None:
    """Child process: persistent Python session with shared global scope."""
    os.chdir(workdir)
    sys.path.insert(0, workdir)
    sys.stdout = sys.stderr = _StdoutRedirect(output_q)

    scope: dict = {}
    while True:
        code = code_q.get()
        if code is None:  # Shutdown signal
            break
        os.chdir(workdir)
        event_q.put(("ready",))
        try:
            exec(compile(code, "<refine>", "exec"), scope)
        except BaseException as exc:
            tb = traceback.format_exception(exc)
            clean_tb = "".join(
                line for line in tb
                if "persistent_interp.py" not in line and "importlib" not in line
            )
            output_q.put(clean_tb)
            event_q.put(("finished", type(exc).__name__))
        else:
            event_q.put(("finished", None))
        output_q.put("<|EOF|>")


class PersistentInterpreter:
    """Persistent Python session — variables, imports, and models survive across run() calls."""

    def __init__(self, workdir: str | Path, timeout: int = 300):
        self.workdir = str(Path(workdir).resolve())
        self.timeout = timeout
        self._process: Process | None = None
        self._code_q: Queue | None = None
        self._output_q: Queue | None = None
        self._event_q: Queue | None = None

    def start(self) -> None:
        """Start (or restart) the persistent session."""
        self.cleanup()
        self._code_q = Queue()
        self._output_q = Queue()
        self._event_q = Queue()
        self._process = Process(
            target=_session_worker,
            args=(self.workdir, self._code_q, self._output_q, self._event_q),
            daemon=True,
        )
        self._process.start()

    def run(self, code: str) -> SessionResult:
        """Execute code in the persistent session. Variables survive between calls."""
        if self._process is None or not self._process.is_alive():
            self.start()

        self._code_q.put(code)

        # Wait for ready signal
        try:
            event = self._event_q.get(timeout=10)
        except queue.Empty:
            self.cleanup()
            return SessionResult(
                output="Session failed to start execution",
                exec_time=0.0,
                error="SessionError",
            )

        # Wait for completion
        start = time.time()
        error_type = None
        while True:
            try:
                event = self._event_q.get(timeout=1.0)
                error_type = event[1]
                break
            except queue.Empty:
                elapsed = time.time() - start
                if elapsed > self.timeout:
                    logger.warning("Persistent session timed out after %ds", self.timeout)
                    self.cleanup()
                    return SessionResult(
                        output=f"TimeoutError: exceeded {self.timeout}s",
                        exec_time=self.timeout,
                        error="TimeoutError",
                    )
                if self._process is not None and not self._process.is_alive():
                    return SessionResult(
                        output="Session process died unexpectedly",
                        exec_time=time.time() - start,
                        error="SessionCrash",
                    )

        exec_time = time.time() - start

        # Collect output
        chunks: list[str] = []
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                chunk = self._output_q.get(timeout=0.5)
                if chunk == "<|EOF|>":
                    break
                chunks.append(chunk)
            except queue.Empty:
                continue

        output = "".join(chunks)
        return SessionResult(output=output, exec_time=exec_time, error=error_type)

    def cleanup(self) -> None:
        if self._process is None:
            return
        try:
            self._process.terminate()
            self._process.join(timeout=3.0)
            if self._process.exitcode is None:
                self._process.kill()
                self._process.join(timeout=2.0)
        except Exception as exc:
            logger.error("Error cleaning up persistent session: %s", exc)
        finally:
            self._process = None

    def __del__(self) -> None:
        self.cleanup()
