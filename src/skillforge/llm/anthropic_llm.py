"""Real backend. Needs ANTHROPIC_API_KEY (or an ``ant auth login`` profile)."""
from __future__ import annotations

import os

from .base import LLMResponse, ToolCall


class AnthropicLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 4096, effort: str = "low"):
        import anthropic  # imported lazily so the offline path has no API dependency

        self.client = anthropic.Anthropic()
        self.name = model or os.environ.get("SKILLFORGE_MODEL", "claude-opus-5-5")
        self.max_tokens = max_tokens
        self.effort = effort

    @staticmethod
    def _to_api(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        pending: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                pending.append({"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]})
                continue
            if pending:
                out.append({"role": "user", "content": pending})
                pending = []
            if m["role"] == "assistant" and m.get("raw") is not None:
                out.append({"role": "assistant", "content": m["raw"]})   # keeps thinking blocks intact
            else:
                out.append({"role": m["role"], "content": m["content"]})
        if pending:
            out.append({"role": "user", "content": pending})
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        api_tools = [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                     for t in tools]
        resp = self.client.messages.create(
            model=self.name, max_tokens=self.max_tokens, system=system,
            messages=self._to_api(messages), tools=api_tools or None,
            output_config={"effort": self.effort},
        ) if api_tools else self.client.messages.create(
            model=self.name, max_tokens=self.max_tokens, system=system,
            messages=self._to_api(messages), output_config={"effort": self.effort},
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"]
        u = resp.usage
        tin = u.input_tokens + (getattr(u, "cache_read_input_tokens", 0) or 0) + (
            getattr(u, "cache_creation_input_tokens", 0) or 0)
        return LLMResponse(text=text, tool_calls=calls, input_tokens=tin, output_tokens=u.output_tokens,
                           raw=resp.content)
