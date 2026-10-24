"""SimLLM: a deterministic stand-in for a language model.

It is **not** a claim about real model behaviour. It exists so the whole pipeline (agent, MCP tools, event bus,
distiller, gate, memory service, evaluation) can be exercised and reproduced offline. The mechanisms it uses are
real and deliberately limited:

* it chooses a skill by lexical similarity between the task and skill docstrings (so look-alike skills can be confused);
* when it writes code from scratch it picks the correct or a flawed implementation with probability ``p_flaw``
  (the flaw only manifests on some inputs, so a flawed solution can pass by luck);
* it answers fact questions only from what ``memory_search`` returned;
* tokens are counted from the real text it reads and writes.

Draws are keyed on (seed, prompt, key) so conditions share random numbers on identical tasks (paired comparison).
"""
from __future__ import annotations

import hashlib
import json
import re
from random import Random

from ..envs.families import FAMILIES
from .base import LLMResponse, ToolCall, estimate_tokens
from .textutil import cosine_overlap, stem_of

SKILL_LINE = re.compile(r"^- (\w+)\(([^)]*)\): (.*)$")


class SimLLM:
    name = "sim"

    def __init__(self, seed: int = 0, p_flaw: float = 0.55, p_flaw_retry: float = 0.15, p_slip: float = 0.06,
                 skill_threshold: float = 0.30):
        self.seed, self.p_flaw, self.p_flaw_retry, self.p_slip = seed, p_flaw, p_flaw_retry, p_slip
        self.skill_threshold = skill_threshold

    # ------------------------------------------------------------------ helpers
    def _rng(self, prompt: str, key: str) -> Random:
        h = hashlib.sha256(f"{self.seed}|{prompt}|{key}".encode()).hexdigest()
        return Random(int(h[:12], 16))

    @staticmethod
    def _catalog(system: str) -> list[tuple[str, list[str], str]]:
        out = []
        for ln in system.splitlines():
            m = SKILL_LINE.match(ln.strip())
            if m:
                params = [p.strip() for p in m.group(2).split(",") if p.strip()]
                out.append((m.group(1), params, m.group(3)))
        return out

    def _best_skill(self, need: str, nparams: int, catalog):
        best, score = None, 0.0
        for name, params, doc in catalog:
            if len(params) != nparams:
                continue
            s = cosine_overlap(need, doc)
            if s > score:
                best, score = (name, params), s
        return best if score >= self.skill_threshold else None

    def _identify(self, prompt: str):
        m = re.search(r"INPUT = (\{.*\})\s*$", prompt, re.S)
        if not m:
            return None, None
        inputs = json.loads(m.group(1))
        cands = [f for f in FAMILIES.values() if set(f.keys) == set(inputs)]
        if not cands:
            return None, inputs
        stem = stem_of(prompt)
        return max(cands, key=lambda f: max(cosine_overlap(stem, s) for s in f.stems)), inputs

    def _pick_src(self, fam, prompt: str, tag: str, p_flaw: float) -> str:
        r = self._rng(prompt, f"flaw:{fam.name}:{tag}")
        if fam.bug_srcs and r.random() < p_flaw:
            return r.choice(fam.bug_srcs)
        return fam.ref_src

    def _write_code(self, prompt: str, attempt: int, catalog) -> tuple[str, str]:
        fam, _ = self._identify(prompt)
        if fam is None:
            return "print('unknown')", "I could not identify the task; trying a placeholder."
        pf = self.p_flaw if attempt == 0 else self.p_flaw_retry
        stem = stem_of(prompt)
        notes = ""
        if fam.components:
            parts = []
            for comp in fam.components:
                cf = FAMILIES[comp]
                need = " ".join(cf.stems)
                hit = self._best_skill(need, len(cf.keys), catalog) if attempt == 0 else None
                if hit:
                    parts.append(f"from skills import {hit[0]} as {cf.fn_name}")
                    notes += f"reuse {hit[0]}; "
                else:
                    parts.append(self._pick_src(cf, prompt, f"c{attempt}", pf).replace("FN", cf.fn_name))
            code = "\n".join(parts) + "\n" + fam.glue_src + "\nprint(solve(**INPUT))\n"
        else:
            hit = self._best_skill(stem, len(fam.keys), catalog) if attempt == 0 else None
            if hit:
                name, params = hit
                call = f"{name}(**INPUT)" if set(params) == set(fam.keys) else f"{name}(*INPUT.values())"
                code = f"from skills import {name}\nprint({call})\n"
                notes = f"reuse skill {name}"
            else:
                code = (self._pick_src(fam, prompt, f"s{attempt}", pf).replace("FN", "solve")
                        + "\nprint(solve(**INPUT))\n")
        if attempt == 0 and not notes and self._rng(prompt, "slip").random() < self.p_slip:
            code = code.replace("print(solve(", "print(solv(")
        return code, ("Plan: " + (notes or "write a solution from scratch") + ". Running it.")

    # ------------------------------------------------------------------ facts
    @staticmethod
    def _facts(content: str) -> list[dict]:
        try:
            return json.loads(content)
        except Exception:
            return []

    def _fact_step(self, prompt: str, messages: list[dict], tools: set[str]):
        m = re.search(r"for the ([\w-]+) service", prompt) or re.search(r"([\w-]+) service", prompt)
        svc = m.group(1)
        endpoint = "endpoint" in prompt
        searches = [x for x in messages if x["role"] == "assistant"
                    and any(t.name == "memory_search" for t in x.get("tool_calls", []))]
        facts: list[dict] = []
        for x in messages:
            if x["role"] == "tool" and x.get("name") == "memory_search":
                facts += self._facts(x["content"])

        def find(pred: str):
            for f in facts:
                if f.get("subject") == svc and f.get("predicate") == pred:
                    return f["object"]
            return None

        need = ["region", "port"] if endpoint else ["oncall"]
        missing = [p for p in need if find(p) is None]
        if "memory_search" in tools and missing and len(searches) < 2:
            q = f"{svc} {missing[0]}" if searches else f"{svc} " + " ".join(need)
            return LLMResponse(text="Checking my notes.",
                               tool_calls=[ToolCall(f"c{len(searches)}", "memory_search", {"query": q, "k": 6})])
        if endpoint:
            region, port = find("region"), find("port")
            ans = f"https://{svc}.{region}.example.com:{port}" if region and port else "unknown"
        else:
            ans = find("oncall") or "unknown"
        return LLMResponse(text=f"FINAL: {ans}")

    # ------------------------------------------------------------------ chat
    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        tool_names = {t["name"] for t in tools}
        resp = self._decide(system, messages[0]["content"], messages, tool_names)
        resp.input_tokens = estimate_tokens(
            system + json.dumps(tools) + "".join(str(m.get("content", "")) for m in messages))
        resp.output_tokens = estimate_tokens(resp.text + "".join(json.dumps(c.args) for c in resp.tool_calls))
        return resp

    def _decide(self, system, prompt, messages, tool_names) -> LLMResponse:
        if "earlier" in prompt and "INPUT =" not in prompt:
            return self._fact_step(prompt, messages, tool_names)
        runs = [m for m in messages if m["role"] == "assistant"
                and any(c.name == "run_python" for c in m.get("tool_calls", []))]
        if runs:
            last = [m for m in messages if m["role"] == "tool"][-1]["content"]
            if not last.startswith("ERROR"):
                out = json.loads(last)["stdout"].strip().splitlines()
                return LLMResponse(text=f"FINAL: {out[-1].strip() if out else 'unknown'}")
            if len(runs) >= 2:
                return LLMResponse(text="FINAL: unknown")
        if "run_python" not in tool_names:
            return LLMResponse(text="FINAL: unknown")
        code, note = self._write_code(prompt, len(runs), self._catalog(system))
        return LLMResponse(text=note, tool_calls=[ToolCall(f"r{len(runs)}", "run_python", {"code": code})])
