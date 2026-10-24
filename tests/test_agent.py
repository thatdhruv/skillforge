import asyncio

from mcp import Client

from skillforge.agent import Agent
from skillforge.envs.stream import build_stream
from skillforge.llm.sim import SimLLM
from skillforge.mcp_server.server import ToolContext, build_server
from skillforge.memory.client import LocalMemory


async def solve_all(use_memory, use_skills, tasks, mem=None):
    mem = mem or LocalMemory()
    ctx = ToolContext(memory=mem, use_memory=use_memory, use_skills=use_skills)
    server = build_server(ctx)
    async with Client(server) as c:
        tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                 for t in (await c.list_tools()).tools]
        agent = Agent(SimLLM(seed=1, p_flaw=0.0, p_slip=0.0), c, tools)
        out = []
        for t in tasks:
            ctx.inputs = t.inputs
            out.append(await agent.solve(t.id, t.prompt, t.kind))
        return out, tools


def test_agent_solves_code_tasks_through_mcp():
    s = build_stream(3)
    tasks = [t for t in s.items if t.kind == "code"][:6]
    trajs, tools = asyncio.run(solve_all(False, False, tasks))
    assert {t["name"] for t in tools} == {"run_python"}
    assert all(tr.answer == t.expected for tr, t in zip(trajs, tasks))
    assert all(tr.tokens_in > 0 and tr.cost > 0 for tr in trajs)


def test_tool_surface_follows_condition():
    _, tools = asyncio.run(solve_all(True, True, []))
    assert {t["name"] for t in tools} == {"run_python", "memory_search", "skill_search"}
