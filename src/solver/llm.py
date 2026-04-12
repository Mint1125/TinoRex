"""Anthropic client for the tree search solver."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from anthropic import Anthropic

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int]


class LLMClient:
    def __init__(self, api_key: str, model: str = "claude-opus-4-6"):
        self.model = model
        self._client = Anthropic(api_key=api_key)

    def generate(self, *, system: str, user: str, temperature: float = 1.0) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=16384,
            system=system,
            messages=[
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        text = resp.content[0].text if resp.content else ""
        usage = {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
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
