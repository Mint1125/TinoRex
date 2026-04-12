"""OpenAI client for Arena synthesis calls."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

_API_KEY_FILE = Path(r"C:/Users/PC4/OneDrive/바탕 화면/개인/개인정보/api_key.txt")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "o4-mini")


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


class ArenaLLM:
    """Lightweight LLM client for synthesis/merge calls in the Arena."""

    def __init__(self):
        api_key = _load_api_key()
        self.model = OPENAI_MODEL
        self._client = OpenAI(api_key=api_key) if api_key else None

    def synthesize(self, *, system: str, user: str, temperature: float = 1.0) -> str:
        """Generate a text response (for merging reports, selecting models, etc.)."""
        if not self._client:
            logger.error("No API key for Arena LLM")
            return ""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        choice = resp.choices[0]
        return choice.message.content or ""
