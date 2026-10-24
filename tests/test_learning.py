import asyncio

from mcp import Client

from skillforge.agent import Agent
from skillforge.envs.stream import build_stream
from skillforge.llm.sim import SimLLM
from skillforge.mcp_server.server import ToolContext, build_server
from skillforge.memory.client import LocalMemory
from skillforge.skills.distiller import extract_skill
from skillforge.skills.gate import GateConfig
from skillforge.skills.pipeline import LearnConfig, Learner


def test_extract_skill_renames_and_documents():
    code = "def solve(n):\n    return n if n < 2 else solve(n - 1) + solve(n - 2)\nprint(solve(INPUT['n']))\n"
    src, params = extract_skill(code, "fib", "Return the nth Fibonacci number.")
    assert params == ["n"] and "def fib(n)" in src and "solve" not in src and "Fibonacci" in src


async def run_stream(n_tasks, gate=GateConfig(), seed=1):
    mem = LocalMemory()
    cfg = LearnConfig(use_memory=True, use_skills=True, gate=gate)
    learner = Learner(mem, cfg)
    ctx = ToolContext(memory=mem, use_memory=True, use_skills=True)
    stream = build_stream(seed)
    correct = 0
    async with Client(build_server(ctx)) as c:
        tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                 for t in (await c.list_tools()).tools]
        agent = Agent(SimLLM(seed=seed), c, tools)
        for t in stream.items[:n_tasks]:
            now = t.ts
            ctx.now = lambda now=now: now
            if t.kind == "obs":
                learner.observe_statement(t, now)
                continue
            ctx.inputs = t.inputs
            tr = await agent.solve(t.id, t.prompt, t.kind)
            tr.success = tr.answer == t.expected
            correct += tr.success
            learner.observe_episode(t, tr, now)
    return mem, learner, correct


def test_skills_are_promoted_through_the_gate():
    mem, learner, _ = asyncio.run(run_stream(90))
    active = mem.list_skills("active")
    assert len(active) >= 3
    decisions = {r["type"] for r in learner.gate.log}
    assert "promoted" in decisions and "insufficient" in decisions
    assert all(s["gate"]["passed"] == s["gate"]["cases"] >= 3 for s in active)


def test_replay_cases_come_from_one_family():
    """Gate evidence must not mix families, or correct skills would be rejected for the wrong reason."""
    mem, learner, _ = asyncio.run(run_stream(90))
    fam = {e["id"]: e["meta"]["_family"] for e in mem.store.episodes.values()}
    from skillforge.envs.families import FAMILIES
    from skillforge.llm.textutil import cosine_overlap, stem_of
    impure = total = 0
    for e in mem.store.episodes.values():
        if e["kind"] != "code" or not e["success"]:
            continue
        cand = {"prompt": e["prompt"], "origin_case": {"inputs": e["meta"]["inputs"], "answer": e["answer"]},
                "params": list(e["meta"]["inputs"]), "origin_episode": e["id"]}
        for c in learner.gate.replay_cases(cand)[1:]:
            total += 1
            match = [x for x in mem.store.episodes.values() if x["meta"]["inputs"] == c["inputs"]]
            impure += any(m["meta"]["_family"] != e["meta"]["_family"] for m in match)
    assert total > 0 and impure / total < 0.05, f"impure replay cases: {impure}/{total}"
