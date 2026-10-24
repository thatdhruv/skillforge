"""LLM interface shared by the real (Anthropic) and simulated backends."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

# $ per 1M tokens (input, output). First-party list prices at time of writing; override with
# SKILLFORGE_PRICE_IN / SKILLFORGE_PRICE_OUT. The simulated model is billed like Haiku 5.5.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "sim": (0.10, 0.50),
}


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""
    return max(1, math.ceil(len(text) / 4))


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES.get(model, PRICES["sim"])
    pin = float(os.environ.get("SKILLFORGE_PRICE_IN", pin))
    pout = float(os.environ.get("SKILLFORGE_PRICE_OUT", pout))
    return (tokens_in * pin + tokens_out * pout) / 1_000_000


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None          # provider-native content, replayed verbatim on the next turn


class LLM(Protocol):
    name: str

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, r: LLMResponse) -> None:
        self.input_tokens += r.input_tokens
        self.output_tokens += r.output_tokens
        self.calls += 1

    def merge(self, o: "Usage") -> None:
        self.input_tokens += o.input_tokens
        self.output_tokens += o.output_tokens
        self.calls += o.calls

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens
