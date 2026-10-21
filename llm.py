"""LLM backends. The agent only needs: llm.chat(system, messages) -> str."""
from __future__ import annotations

import os


class AnthropicLLM:
    def __init__(self, model: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def chat(self, system: str, messages: list[dict]) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=800, temperature=0, system=system, messages=messages)
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAICompatLLM:
    """Works with OpenAI, Groq, Together, etc. via OPENAI_BASE_URL."""
    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def chat(self, system: str, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0, max_tokens=800,
            messages=[{"role": "system", "content": system}, *messages])
        return resp.choices[0].message.content or ""


class ScriptedLLM:
    """Replays canned replies. Used for tests and the offline demo."""
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, system: str, messages: list[dict]) -> str:
        if self.calls >= len(self.replies):
            raise RuntimeError("ScriptedLLM ran out of replies")
        self.calls += 1
        return self.replies[self.calls - 1]


def get_llm(provider: str):
    if provider == "anthropic":
        return AnthropicLLM()
    if provider == "openai":
        return OpenAICompatLLM()
    raise ValueError(f"Unknown provider '{provider}' (use anthropic | openai)")
