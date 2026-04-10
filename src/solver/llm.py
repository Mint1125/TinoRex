"""LLM client for the tree search solver — supports OpenAI, Anthropic, and Google."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# Provider configs: (base_url, needs special handling)
_PROVIDER_CONFIG = {
    "openai": {"base_url": None},
    "anthropic": {"base_url": "https://api.anthropic.com/v1/"},
    "google": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
}


def _detect_provider(model: str) -> str:
    """Detect provider from model name."""
    model_lower = model.lower()
    if any(k in model_lower for k in ("claude",)):
        return "anthropic"
    if any(k in model_lower for k in ("gemini",)):
        return "google"
    return "openai"


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int]


class LLMClient:
    def __init__(self, api_key: str, model: str = "o4-mini",
                 base_url: str | None = None, provider: str | None = None):
        self.model = model
        self.provider = provider or _detect_provider(model)
        cfg = _PROVIDER_CONFIG.get(self.provider, _PROVIDER_CONFIG["openai"])
        effective_base_url = base_url or cfg["base_url"]
        kwargs: dict[str, Any] = {"api_key": api_key}
        if effective_base_url:
            kwargs["base_url"] = effective_base_url
        self._client = OpenAI(**kwargs)
        logger.info("LLMClient: provider=%s, model=%s, base_url=%s",
                     self.provider, self.model, effective_base_url)

    def generate(self, *, system: str, user: str, temperature: float = 1.0) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Reasoning models (o-series) don't accept temperature
        model_lower = self.model.lower()
        is_reasoning = model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4")
        if not is_reasoning:
            kwargs["temperature"] = temperature
        # Anthropic requires max_tokens
        if self.provider == "anthropic":
            kwargs["max_tokens"] = 16384

        resp = self._client.chat.completions.create(**kwargs)
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
