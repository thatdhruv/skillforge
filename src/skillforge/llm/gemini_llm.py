"""Gemini backend (Google AI Studio REST API). Needs GEMINI_API_KEY; model via SKILLFORGE_GEMINI_MODEL."""
from __future__ import annotations

import os
import re
import time

import httpx

from .base import LLMResponse, ToolCall

BASE = "https://generativelanguage.googleapis.com/v1beta"
_DROP = {"title", "default", "additionalProperties", "$schema"}


def _clean(schema):
    if isinstance(schema, dict):
        return {k: _clean(v) for k, v in schema.items() if k not in _DROP}
    if isinstance(schema, list):
        return [_clean(x) for x in schema]
    return schema


class GeminiLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 4096, timeout: float = 120.0):
        self.name = model or os.environ.get("SKILLFORGE_GEMINI_MODEL", "gemini-3.1-flash-lite")
        self.max_tokens = max_tokens
        self.thinking = os.environ.get("SKILLFORGE_GEMINI_THINKING", "minimal")   # minimal | low | medium | high
        self.min_gap = 60.0 / float(os.environ.get("SKILLFORGE_RPM", "7"))   # free tier: 15 requests/min across processes
        self._last = 0.0
        self.http = httpx.Client(timeout=timeout, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})

    @staticmethod
    def _contents(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        pending: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                pending.append({"functionResponse": {"name": m["name"], "response": {"result": m["content"]}}})
                continue
            if pending:
                out.append({"role": "user", "parts": pending})
                pending = []
            if m["role"] == "assistant":
                if m.get("raw"):                       # replay the model's own parts (keeps thought signatures)
                    out.append({"role": "model", "parts": m["raw"]})
                else:
                    parts = ([{"text": m["content"]}] if m["content"] else []) + [
                        {"functionCall": {"name": c.name, "args": c.args}} for c in m.get("tool_calls", [])]
                    out.append({"role": "model", "parts": parts or [{"text": ""}]})
            else:
                out.append({"role": "user", "parts": [{"text": m["content"]}]})
        if pending:
            out.append({"role": "user", "parts": pending})
        return out

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body: dict = {"systemInstruction": {"parts": [{"text": system}]}, "contents": self._contents(messages),
                      "generationConfig": {"maxOutputTokens": self.max_tokens,
                                           "thinkingConfig": {"thinkingLevel": self.thinking}}}
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t["name"], "description": t["description"], "parameters": _clean(t["input_schema"])}
                for t in tools]}]
        for attempt in range(8):
            wait = self._last + self.min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = self.http.post(f"{BASE}/models/{self.name}:generateContent", json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                m = re.search(r"retry in ([\d.]+)s", r.text)
                time.sleep(float(m.group(1)) + 1 if m else min(20, 2 ** attempt) + 0.5 * attempt)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Gemini API error: HTTP {r.status_code} {r.text[:300]}")
            break
        else:
            raise RuntimeError(f"Gemini API unavailable after retries: HTTP {r.status_code} {r.text[:200]}")
        d = r.json()
        cand = (d.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        calls = [ToolCall(f"call-{i}", p["functionCall"]["name"], dict(p["functionCall"].get("args", {})))
                 for i, p in enumerate(x for x in parts if "functionCall" in x)]
        u = d.get("usageMetadata", {})
        return LLMResponse(text=text, tool_calls=calls, input_tokens=u.get("promptTokenCount", 0),
                           output_tokens=u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0), raw=parts)
