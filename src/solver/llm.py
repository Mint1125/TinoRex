"""OpenAI client for the stage-wise solver."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int]


class LLMClient:
    def __init__(self, api_key: str, model: str = "o4-mini"):
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def generate(self, *, system: str, user: str, temperature: float = 1.0) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        return LLMResponse(text=text, usage=usage)

    def generate_code(self, *, system: str, user: str, temperature: float = 1.0) -> str:
        """Generate and extract a Python code block from the LLM response."""
        resp = self.generate(system=system, user=user, temperature=temperature)
        return self._extract_code(resp.text)

    @staticmethod
    def _extract_code(text: str) -> str:
        if "```python" in text:
            parts = text.split("```python", 1)[1]
            code = parts.split("```", 1)[0]
            return code.strip()
        if "```" in text:
            parts = text.split("```", 1)[1]
            code = parts.split("```", 1)[0]
            return code.strip()
        return text.strip()
