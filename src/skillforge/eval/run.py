"""Experiment runner: the same task stream under four conditions (+ ablations), several seeds.

Every run is independent and writes ``<out>/<condition>_s<seed>.json``; runs execute in parallel processes and are
resumable. Two kinds of rows are produced:

* ``stream``  - the online task stream (the agent learns as it goes);
* ``probe``   - read-only evaluations at checkpoints (no memory writes, no usage updates, no distillation):
  ``heldin`` (fresh instances of trained families), ``transfer`` (unseen composite families),
  ``retention`` (fixed instances of the earliest-taught families) and ``fact`` (queries on the current world state).
"""
from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from mcp import Client

from ..agent import Agent
from ..bus import CANDIDATES
from ..envs.stream import Task, build_stream
from ..llm.base import estimate_tokens
from ..mcp_server.server import ToolContext, build_server
from ..memory.client import LocalMemory
from ..skills.gate import GateConfig
from ..skills.pipeline import LearnConfig, Learner


@dataclass(frozen=True)
class Condition:
    use_memory: bool
    use_skills: bool
    gate: GateConfig = field(default_factory=GateConfig)
    memory_rules: bool = True


CONDITIONS: dict[str, Condition] = {
    "frozen": Condition(False, False),
    "memory": Condition(True, False),
    "skills": Condition(False, True),
    "both": Condition(True, True),
    # ablations used for the "when did self-evolution hurt" analysis
    "both_nogate": Condition(True, True, GateConfig(enabled=False)),
    "both_weakgate": Condition(True, True, GateConfig(min_cases=1)),
    "memory_norules": Condition(True, False, memory_rules=False),
}
MAIN = ["frozen", "memory", "skills", "both"]
ABLATIONS = ["both_nogate", "both_weakgate", "memory_norules"]


def make_llm(kind: str, seed: int):
    if kind == "sim":
        from ..llm.sim import SimLLM
        return SimLLM(seed=seed)
    if kind == "gemini":
        from ..llm.gemini_llm import GeminiLLM
        return GeminiLLM()
    from ..llm.anthropic_llm import AnthropicLLM
    return AnthropicLLM()


def _row(base: dict, task: Task, traj, extra: dict | None = None) -> dict:
    ok = traj.answer.strip() == task.expected.strip()
    traj.success = ok
    return {**base, "task_id": task.id, "kind": task.kind, "family": task.family, "success": ok,
            "answer": traj.answer, "expected": task.expected, "tokens_in": traj.tokens_in,
            "tokens_out": traj.tokens_out, "cost": traj.cost, "llm_calls": traj.llm_calls,
            "skills_used": traj.skills_used, "offered": [s["name"] for s in traj.offered_skills],
            "run_errors": traj.run_errors, "code": traj.final_code, **(extra or {})}


async def _run(cond_name: str, seed: int, n: int, llm_kind: str, every: int = 40, small: int = 0,
               families: list[str] | None = None) -> dict:
    cond = CONDITIONS[cond_name]
    stream = build_stream(seed, n, every, small, families)
    mem = LocalMemory(rules=cond.memory_rules)
    learner = None
    if cond.use_memory or cond.use_skills:
        learner = Learner(mem, LearnConfig(cond.use_memory, cond.use_skills, cond.gate))
    ctx = ToolContext(memory=mem, use_memory=cond.use_memory, use_skills=cond.use_skills)
    rows: list[dict] = []
    errors: list[str] = []
    timeline: list[dict] = []
    t0 = time.time()
    async with Client(build_server(ctx)) as c:
        tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                 for t in (await c.list_tools()).tools]
        agent = Agent(make_llm(llm_kind, seed), c, tools)

        async def attempt(task: Task):
            ctx.inputs, ctx.now = task.inputs, (lambda ts=task.ts: ts)
            try:
                return await agent.solve(task.id, task.prompt, task.kind)
            except Exception as e:                      # a model/API failure counts as a failed task, not a crashed run
                errors.append(f"{task.id}: {type(e).__name__}: {str(e)[:200]}")
                from ..agent.trajectory import Trajectory
                return Trajectory(task_id=task.id, prompt=task.prompt, kind=task.kind)

        for i, task in enumerate(stream.items):
            if i in stream.probes:
                for pset, tasks in stream.probes[i].items():
                    for pt in tasks:
                        rows.append(_row({"phase": "probe", "cond": cond_name, "seed": seed, "cp": i, "set": pset},
                                         pt, await attempt(pt)))
            if task.kind == "obs":
                if learner:
                    learner.observe_statement(task, task.ts)
            else:
                traj = await attempt(task)
                before = len(learner.bus.topics[CANDIDATES]) if learner and cond.use_skills else 0
                extra = {}
                row = _row({"phase": "stream", "cond": cond_name, "seed": seed, "step": i}, task, traj)
                if learner:
                    learner.observe_episode(task, traj, task.ts)
                    if cond.use_skills and len(learner.bus.topics[CANDIDATES]) > before:
                        extra = {"learn_tokens_in": estimate_tokens(task.prompt + traj.final_code) + 120,
                                 "learn_tokens_out": 40}
                rows.append({**row, **extra})
            st = mem.stats()["skills"]
            gl = learner.gate.log if learner and learner.gate else []
            timeline.append({"step": i, "ts": task.ts, "active": st.get("active", 0), "deprecated": st.get("deprecated", 0),
                             "superseded": st.get("superseded", 0),
                             "promoted": sum(r["type"] == "promoted" for r in gl),
                             "rejected": sum(r["type"] == "rejected" for r in gl),
                             "insufficient": sum(r["type"] == "insufficient" for r in gl),
                             "redundant": sum(r["type"] == "redundant" for r in gl)})
        final_probe_cp = stream.checkpoints[-1]
        if final_probe_cp not in [r.get("cp") for r in rows if r["phase"] == "probe"]:
            for pset, tasks in stream.probes.get(final_probe_cp, {}).items():
                for pt in tasks:
                    rows.append(_row({"phase": "probe", "cond": cond_name, "seed": seed, "cp": final_probe_cp,
                                      "set": pset}, pt, await attempt(pt)))
    return {"cond": cond_name, "seed": seed, "n": n, "llm": llm_kind, "seconds": time.time() - t0, "rows": rows,
            "timeline": timeline, "skills": mem.list_skills(), "events": mem.events(),
            "gate_log": learner.gate.log if learner and learner.gate else [], "errors": errors}


def run_one(cond_name: str, seed: int, n: int, llm_kind: str, out: str, every: int = 40, small: int = 0,
            families: list[str] | None = None) -> str:
    path = Path(out) / f"{cond_name}_s{seed}.json"
    if path.exists():
        return str(path)
    res = asyncio.run(_run(cond_name, seed, n, llm_kind, every, small, families))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    return str(path)


def run_many(conds: list[str], seeds: list[int], n: int, llm_kind: str, out: str, workers: int = 16,
             every: int = 40, small: int = 0, families: list[str] | None = None) -> list[str]:
    jobs = [(c, s) for s in seeds for c in conds]
    done = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, c, s, n, llm_kind, out, every, small, families): (c, s) for c, s in jobs}
        for f in as_completed(futs):
            done.append(f.result())
            print(f"[{len(done)}/{len(jobs)}] finished {futs[f][0]} seed={futs[f][1]}", flush=True)
    return done
