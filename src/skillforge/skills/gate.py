"""Skill gate: a candidate joins the library only after it passes replay tests in the sandbox.

Replay cases are verified past successes (inputs + the answer the environment accepted) retrieved from episodic
memory. The gate refuses candidates with too little evidence, deduplicates behaviourally against the library, and
records why every decision was made.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..bus import CANDIDATES, SKILL_EVENTS, Bus
from ..envs.sandbox import run_python, screen
from ..llm.textutil import cosine_overlap, stem_of
from ..memory.client import Memory

BATCH = '''import json
_res = []
for _c in INPUT["cases"]:
    try:
        _res.append(str(NAME(**_c)))
    except Exception as _e:
        _res.append("EXC:" + repr(_e)[:40])
print(json.dumps(_res))
'''


@dataclass
class GateConfig:
    enabled: bool = True
    min_cases: int = 3          # distinct verified cases required (the origin case counts)
    max_cases: int = 10
    min_stem_overlap: float = 0.25
    timeout: float = 8.0


@dataclass
class GateResult:
    decision: str               # promoted | rejected | redundant | insufficient | invalid
    reason: str = ""
    cases: int = 0
    passed: int = 0
    redundant_with: str | None = None


def run_cases(source: str, name: str, cases: list[dict], skills: dict[str, str], timeout: float) -> list[str]:
    r = run_python(source + "\n" + BATCH.replace("NAME", name), {"cases": [c["inputs"] for c in cases]},
                   skills, timeout)
    if not r.ok:
        return ["EXC:" + (r.stderr.splitlines()[-1] if r.stderr else "failed")] * len(cases)
    try:
        return json.loads(r.answer)
    except Exception:
        return ["EXC:bad-output"] * len(cases)


class Gate:
    def __init__(self, memory: Memory, bus: Bus, cfg: GateConfig | None = None, group: str = "gate"):
        self.mem, self.bus, self.cfg = memory, bus, cfg or GateConfig()
        self.consumer = bus.consumer(CANDIDATES, group)
        self.log: list[dict] = []

    def replay_cases(self, cand: dict) -> list[dict]:
        stem = stem_of(cand["prompt"])
        cases = [cand["origin_case"]]
        seen = {json.dumps(cand["origin_case"]["inputs"], sort_keys=True)}
        for e in self.mem.similar_episodes(stem, k=40, only_success=True):
            meta = e.get("meta") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            inputs = meta.get("inputs")
            if not inputs or set(inputs) != set(cand["params"]) or e["id"] == cand["origin_episode"]:
                continue
            if cosine_overlap(stem_of(e.get("prompt", "")), stem) < self.cfg.min_stem_overlap:
                continue
            k = json.dumps(inputs, sort_keys=True)
            if k in seen:
                continue
            seen.add(k)
            cases.append({"inputs": inputs, "answer": e["answer"]})
            if len(cases) >= self.cfg.max_cases:
                break
        return cases

    def evaluate(self, cand: dict) -> GateResult:
        bad = screen(cand["source"])
        if bad:
            return GateResult("invalid", bad)
        if not self.cfg.enabled:
            return GateResult("promoted", "gate disabled")
        cases = self.replay_cases(cand)
        if len(cases) < self.cfg.min_cases:
            return GateResult("insufficient", f"{len(cases)} < {self.cfg.min_cases} verified cases", len(cases))
        skills = self.mem.active_sources()
        out = run_cases(cand["source"], cand["name"], cases, skills, self.cfg.timeout)
        ok = sum(o == c["answer"] for o, c in zip(out, cases))
        if ok < len(cases):
            return GateResult("rejected", f"failed {len(cases) - ok} of {len(cases)} replay cases", len(cases), ok)
        for s in self.mem.list_skills("active"):
            if set(s["params"]) != set(cand["params"]):
                continue
            theirs = run_cases(skills[s["name"]], s["name"], cases, skills, self.cfg.timeout)
            if all(o == c["answer"] for o, c in zip(theirs, cases)):
                return GateResult("redundant", f"existing skill {s['name']} already passes", len(cases), ok, s["name"])
        return GateResult("promoted", "passed all replay cases", len(cases), ok)

    def unique_name(self, cand: dict) -> str:
        """Same name + same signature = a new version. Same name + different signature = different skill."""
        name = cand["name"]
        for s in self.mem.list_skills():
            if s["name"] == name and set(s["params"]) != set(cand["params"]):
                i = 2
                taken = {x["name"] for x in self.mem.list_skills()}
                while f"{name}_{i}" in taken:
                    i += 1
                return f"{name}_{i}"
        return name

    def step(self, max_messages: int = 100, timeout: float = 0.0) -> int:
        n = 0
        for cand in self.consumer.poll(max_messages, timeout):
            res = self.evaluate(cand)
            name = cand["name"]
            if res.decision == "promoted":
                name = self.unique_name(cand)
                src = cand["source"]
                if name != cand["name"]:
                    pat = "(?<![A-Za-z0-9_])" + re.escape(cand["name"]) + "(?![A-Za-z0-9_])"
                    src = re.sub(pat, name, src)
                self.mem.put_skill(name, src, cand["docstring"], cand["params"], origin_episode=cand["origin_episode"],
                                   gate={"cases": res.cases, "passed": res.passed, "reason": res.reason},
                                   now=cand.get("ts", 0.0))
            rec = {"type": res.decision, "name": name, "reason": res.reason, "cases": res.cases, "passed": res.passed,
                   "ts": cand.get("ts", 0.0), "origin": cand["origin_episode"]}
            self.log.append(rec)
            self.bus.publish(SKILL_EVENTS, rec, key=name)
            n += 1
        self.consumer.commit()
        return n
