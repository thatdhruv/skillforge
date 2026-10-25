"""The agent: a LangGraph state machine ``retrieve -> think -> act -> think ... -> finish``.

All tools are reached through an MCP client; the graph itself knows nothing about sandboxes or memory.
"""
from __future__ import annotations

import json
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import Client

from ..llm.base import LLM, Usage, cost_usd
from ..llm.textutil import stem_of
from ..mcp_server.server import INTERNAL_TOOLS
from .trajectory import Trajectory

MAX_STEPS = 6

SYSTEM = """You are a careful programming and operations assistant. Solve the task, then reply with a final line
`FINAL: <answer>` containing only the answer. Use the tools to run code and to look things up; do not guess when a tool can verify.
Once a tool result shows the answer, stop calling tools and reply with the FINAL line immediately; do not re-run to double-check.
{skills}"""


class State(TypedDict):
    task_id: str
    prompt: str
    system: str
    messages: Annotated[list, operator.add]
    steps: int
    answer: str
    done: bool


def _skill_block(skills: list[dict]) -> str:
    if not skills:
        return ""
    lines = ["## Skill library (verified functions; import with `from skills import <name>`)"]
    for s in skills:
        doc = s["docstring"].strip().splitlines()[0] if s["docstring"].strip() else ""
        lines.append(f"- {s['name']}({', '.join(s['params'])}): {doc}")
    return "\n".join(lines)


class Agent:
    def __init__(self, llm: LLM, mcp: Client, tools: list[dict], retrieve_k: int = 5):
        self.llm, self.mcp, self.tools, self.k = llm, mcp, tools, retrieve_k
        self.has_skills = any(t["name"] in INTERNAL_TOOLS for t in tools)
        self.visible = [t for t in tools if t["name"] not in INTERNAL_TOOLS]
        self._traj: Trajectory | None = None
        self._usage = Usage()
        self.graph = self._build()

    # ----------------------------------------------------------------- nodes
    async def _retrieve(self, st: State) -> dict:
        offered: list[dict] = []
        if self.has_skills:
            r = await self.mcp.call_tool("skill_search", {"query": stem_of(st["prompt"]), "k": self.k})
            offered = json.loads(r.content[0].text)
        self._traj.offered_skills = [{"name": s["name"], "version": s["version"]} for s in offered]
        return {"system": SYSTEM.format(skills=_skill_block(offered))}

    async def _think(self, st: State) -> dict:
        resp = self.llm.chat(st["system"], st["messages"], self.visible)
        self._usage.add(resp)
        self._traj.steps.append({"text": resp.text,
                                 "tool_calls": [{"name": c.name, "args": c.args} for c in resp.tool_calls],
                                 "results": []})
        msg = {"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls, "raw": resp.raw}
        if resp.tool_calls and st["steps"] + 1 < MAX_STEPS:
            return {"messages": [msg], "steps": st["steps"] + 1}
        ans = resp.text.split("FINAL:", 1)[1].strip() if "FINAL:" in resp.text else resp.text.strip()
        if not ans and self._traj.last_stdout:        # step budget exhausted: fall back to the last successful output
            ans = self._traj.last_stdout
        return {"messages": [msg], "steps": st["steps"] + 1, "answer": ans, "done": True}

    async def _act(self, st: State) -> dict:
        last = st["messages"][-1]
        out = []
        for call in last["tool_calls"]:
            r = await self.mcp.call_tool(call.name, call.args)
            text = r.content[0].text if r.content else ""
            if call.name == "run_python":
                if text.startswith("ERROR"):
                    self._traj.run_errors += 1
                else:
                    self._traj.final_code = call.args.get("code", "")
                    lines = [ln for ln in json.loads(text).get("stdout", "").strip().splitlines() if ln.strip()]
                    self._traj.last_stdout = lines[-1].strip() if lines else ""
                    self._traj.skills_used += json.loads(text).get("skills_used", [])
            self._traj.steps[-1]["results"].append(text[:500])
            out.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": text})
        return {"messages": out}

    def _build(self):
        g = StateGraph(State)
        g.add_node("retrieve", self._retrieve)
        g.add_node("think", self._think)
        g.add_node("act", self._act)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "think")
        g.add_conditional_edges("think", lambda s: END if s["done"] else "act")
        g.add_edge("act", "think")
        return g.compile()

    # ----------------------------------------------------------------- API
    async def solve(self, task_id: str, prompt: str, kind: str = "code") -> Trajectory:
        self._traj = Trajectory(task_id=task_id, prompt=prompt, kind=kind)
        self._usage = Usage()
        init: State = {"task_id": task_id, "prompt": prompt, "system": "",
                       "messages": [{"role": "user", "content": prompt}], "steps": 0, "answer": "", "done": False}
        final = await self.graph.ainvoke(init)
        t = self._traj
        t.answer = final["answer"]
        t.tokens_in, t.tokens_out, t.llm_calls = self._usage.input_tokens, self._usage.output_tokens, self._usage.calls
        t.cost = cost_usd(self.llm.name, t.tokens_in, t.tokens_out)
        return t
