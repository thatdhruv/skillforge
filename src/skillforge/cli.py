"""``skillforge`` command line: experiments, reports, and the long-running services used in Docker/Kubernetes."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time


def _http_memory():
    from .memory.client import HttpMemory
    return HttpMemory(os.environ.get("MEMORY_URL", "http://localhost:8080"))


def _kafka():
    from .bus import KafkaBus, wait_for_kafka
    brokers = os.environ.get("KAFKA_BROKERS", "localhost:19092")
    wait_for_kafka(brokers)
    return KafkaBus(brokers)


def cmd_distiller(a):
    from .bus import TRAJECTORIES  # noqa: F401
    from .skills.distiller import Distiller
    bus = _kafka()
    d = Distiller(bus)
    print("distiller consuming trajectories.finished", flush=True)
    while True:
        n = d.step(timeout=1.0)
        if n:
            bus.flush()
            print(f"proposed {n} candidate skill(s)", flush=True)


def cmd_gate(a):
    from .skills.gate import Gate, GateConfig
    bus = _kafka()
    g = Gate(_http_memory(), bus, GateConfig(min_cases=a.min_cases))
    print("gate consuming skills.candidates", flush=True)
    while True:
        n = g.step(timeout=1.0)
        if n:
            bus.flush()
            for r in g.log[-n:]:
                print(f"{r['type']:<12} {r['name']}  ({r['reason']})", flush=True)


def cmd_worker(a):
    """Runs the agent over a task stream against the real memory service, publishing trajectories to Kafka."""
    from mcp import Client

    from .agent import Agent
    from .envs.stream import build_stream
    from .eval.run import make_llm
    from .mcp_server.server import ToolContext, build_server
    from .skills.pipeline import LearnConfig, Learner

    mem, bus = _http_memory(), _kafka()
    learner = Learner(mem, LearnConfig(use_memory=True, use_skills=True), bus=bus, inline=False)
    ctx = ToolContext(memory=mem, use_memory=True, use_skills=True)

    async def main():
        stream = build_stream(a.seed, a.n)
        async with Client(build_server(ctx)) as c:
            tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                     for t in (await c.list_tools()).tools]
            agent = Agent(make_llm(a.llm, a.seed), c, tools)
            ok = tot = 0
            for i, task in enumerate(stream.items):
                if task.kind == "obs":
                    learner.observe_statement(task, task.ts)
                    continue
                ctx.inputs, ctx.now = task.inputs, (lambda ts=task.ts: ts)
                t = await agent.solve(task.id, task.prompt, task.kind)
                t.success = t.answer.strip() == task.expected.strip()
                learner.observe_episode(task, t, task.ts)
                bus.flush()
                ok, tot = ok + t.success, tot + 1
                print(f"[{i:03d}] {task.family:<14} {'ok ' if t.success else 'ERR'} acc={ok / tot:.2f} "
                      f"skills_used={t.skills_used}", flush=True)
                time.sleep(a.delay)

    asyncio.run(main())


def cmd_serve_mcp(a):
    from .mcp_server.server import ToolContext, build_server
    mem = _http_memory()
    server = build_server(ToolContext(memory=mem, use_memory=True, use_skills=True))
    if a.transport == "stdio":
        server.run()
    else:
        server.run(transport="streamable-http")


def cmd_eval(a):
    from .eval.analyze import summarize
    from .eval.run import ABLATIONS, MAIN, run_many
    conds = MAIN + ([] if a.no_ablations else ABLATIONS)
    run_many(conds, list(range(a.seeds)), a.n, a.llm, a.out, a.workers)
    summarize(a.out)


def cmd_analyze(a):
    from .eval.analyze import summarize
    summarize(a.out)


def cmd_report(a):
    from .dashboard.build import build
    from .dashboard.figures import make_all
    print(make_all(a.results, a.figures))
    print(build(a.results, a.html))


def cmd_demo(a):
    """Tiny end-to-end demonstration in one process; prints the skill library as it grows."""
    from .eval.run import CONDITIONS, _run
    res = asyncio.run(_run("both", a.seed, a.n, "sim"))
    ok = [r["success"] for r in res["rows"] if r["phase"] == "stream" and r["kind"] != "obs"]
    print(f"stream accuracy: {sum(ok) / len(ok):.2f} over {len(ok)} scored tasks")
    for s in res["skills"]:
        print(f"  {s['status']:<10} {s['name']} v{s['version']}  uses={s['uses']} gate={s['gate']}")
    json.dump(res["gate_log"][-5:], sys.stdout, indent=1)
    print()


def main(argv=None):
    p = argparse.ArgumentParser(prog="skillforge")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("distiller").set_defaults(f=cmd_distiller)
    g = sub.add_parser("gate"); g.add_argument("--min-cases", type=int, default=3); g.set_defaults(f=cmd_gate)
    w = sub.add_parser("worker")
    w.add_argument("--seed", type=int, default=0); w.add_argument("--n", type=int, default=240)
    w.add_argument("--llm", default="sim"); w.add_argument("--delay", type=float, default=0.0); w.set_defaults(f=cmd_worker)
    m = sub.add_parser("serve-mcp"); m.add_argument("--transport", default="stdio", choices=["stdio", "http"]); m.set_defaults(f=cmd_serve_mcp)
    e = sub.add_parser("eval")
    e.add_argument("--seeds", type=int, default=10); e.add_argument("--n", type=int, default=240)
    e.add_argument("--llm", default="sim", choices=["sim", "anthropic"]); e.add_argument("--out", default="results/main")
    e.add_argument("--workers", type=int, default=16); e.add_argument("--no-ablations", action="store_true"); e.set_defaults(f=cmd_eval)
    an = sub.add_parser("analyze"); an.add_argument("--out", default="results/main"); an.set_defaults(f=cmd_analyze)
    r = sub.add_parser("report")
    r.add_argument("--results", default="results"); r.add_argument("--figures", default="docs/images")
    r.add_argument("--html", default="docs/dashboard.html"); r.set_defaults(f=cmd_report)
    d = sub.add_parser("demo"); d.add_argument("--seed", type=int, default=0); d.add_argument("--n", type=int, default=100); d.set_defaults(f=cmd_demo)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
