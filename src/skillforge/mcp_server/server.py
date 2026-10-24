"""MCP tool server. Which tools exist depends on the experimental condition, not on the agent code."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from mcp.server.mcpserver import MCPServer

from ..envs.sandbox import run_python as sandbox_run
from ..memory.client import Memory

INTERNAL_TOOLS = {"skill_search"}     # used by the agent graph for retrieval, never offered to the model


@dataclass
class ToolContext:
    memory: Memory | None = None
    use_memory: bool = False          # semantic memory tool enabled
    use_skills: bool = False          # procedural memory (skills importable + searchable)
    now: Callable[[], float] = lambda: 0.0
    inputs: dict = field(default_factory=dict)   # INPUT bound for the current task
    timeout: float = 5.0


def build_server(ctx: ToolContext, name: str = "skillforge-tools") -> MCPServer:
    server = MCPServer(name)

    @server.tool()
    def run_python(code: str) -> str:
        """Run Python 3 code in a sandbox and return its stdout. The dict INPUT holds the task input; print the answer.
        Imports are restricted to the standard library subset (re, math, json, itertools, collections, datetime, ...)."""
        skills = ctx.memory.active_sources() if (ctx.use_skills and ctx.memory) else {}
        r = sandbox_run(code, ctx.inputs, skills, ctx.timeout)
        if not r.ok:
            return "ERROR: " + (r.stderr or "failed")[-600:]
        return json.dumps({"stdout": r.stdout, "stderr": r.stderr, "skills_used": r.skills_used})

    if ctx.use_memory and ctx.memory:
        @server.tool()
        def memory_search(query: str, k: int = 5) -> str:
            """Search long-term memory of facts told to you earlier (hybrid vector + keyword). Returns JSON facts."""
            hits = ctx.memory.search_facts(query, k, ctx.now())
            keep = ("subject", "predicate", "object", "text", "valid_from")
            return json.dumps([{x: h.get(x) for x in keep} for h in hits])

    if ctx.use_skills and ctx.memory:
        @server.tool()
        def skill_search(query: str, k: int = 5) -> str:
            """Find library skills (Python functions) relevant to a task description."""
            hits = ctx.memory.search_skills(query, k, ctx.now())
            return json.dumps([{"name": h["name"], "version": h["version"], "params": h["params"],
                                "docstring": h["docstring"]} for h in hits])

    return server
