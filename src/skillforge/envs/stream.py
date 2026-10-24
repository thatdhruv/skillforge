"""Task stream: code tasks, fact observations/queries, and read-only probe sets at checkpoints."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from random import Random

from .families import FAMILIES, TRAIN_FAMILIES, TRANSFER_FAMILIES, make_instance

HOUR = 3600.0
ONCALL_TTL = 72 * HOUR
SERVICES = ["billing", "search", "auth", "orders", "catalog", "payments", "inventory", "notify", "reports",
            "gateway", "ledger", "media"]
REGIONS = ["us-east-1", "eu-west-1", "ap-south-1", "us-west-2", "eu-central-1"]
PEOPLE = ["alice", "bob", "chen", "dmitri", "esha", "farid"]


@dataclass
class Task:
    id: str
    kind: str                      # "code" | "fact" | "obs"
    family: str
    prompt: str
    expected: str = ""
    inputs: dict = field(default_factory=dict)
    ts: float = 0.0
    split: str = "train"
    meta: dict = field(default_factory=dict)


@dataclass
class Stream:
    items: list[Task]
    checkpoints: list[int]
    probes: dict[int, dict[str, list[Task]]]   # checkpoint -> probe-set name -> tasks


class World:
    """Ground truth for the fact tasks."""

    def __init__(self):
        self.region: dict[str, str] = {}
        self.port: dict[str, int] = {}
        self.oncall: dict[str, tuple[str, float]] = {}

    def endpoint(self, svc: str) -> str:
        return f"https://{svc}.{self.region[svc]}.example.com:{self.port[svc]}"

    def who(self, svc: str, now: float) -> str:
        v = self.oncall.get(svc)
        return v[0] if v and now < v[1] else "unknown"


def endpoint_task(tid: str, svc: str, world: World, ts: float) -> Task:
    prompt = (f"Using what we discussed earlier, give the endpoint URL for the {svc} service in the form "
              f"https://{svc}.<region>.example.com:<port>")
    return Task(tid, "fact", "endpoint", prompt, world.endpoint(svc), ts=ts, meta={"service": svc})


def oncall_task(tid: str, svc: str, world: World, ts: float) -> Task:
    prompt = (f"Per what we noted earlier, who is currently on call for the {svc} service? "
              f"Answer with the name only, or 'unknown' if nobody is currently assigned.")
    return Task(tid, "fact", "oncall", prompt, world.who(svc, ts), ts=ts, meta={"service": svc})


def _obs(tid: str, text: str, ts: float, doc_ts: float | None = None) -> Task:
    return Task(tid, "obs", "obs", text, ts=ts, meta={"doc_ts": ts if doc_ts is None else doc_ts})


def code_task(tid: str, fam_name: str, rng: Random, ts: float) -> Task:
    fam = FAMILIES[fam_name]
    inputs, prompt, expected = make_instance(fam, rng)
    return Task(tid, "code", fam_name, prompt, expected, inputs, ts, fam.split)


# -------------------------------------------------------------------- fact extraction (rule based)
_PATTERNS = [
    (re.compile(r"the (?P<s>[\w-]+) service runs in region (?P<o>[\w-]+)"), "region", None),
    (re.compile(r"the (?P<s>[\w-]+) service listens on port (?P<o>\d+)"), "port", None),
    (re.compile(r"(?P<s>[\w-]+) has been moved to region (?P<o>[\w-]+)"), "region", None),
    (re.compile(r"on-call engineer for (?P<s>[\w-]+) is (?P<o>\w+)"), "oncall", ONCALL_TTL),
    (re.compile(r"(?P<s>[\w-]+) used to run in region (?P<o>[\w-]+)"), "region", None),
]


def extract_facts(text: str, doc_ts: float) -> list[dict]:
    out = []
    for rx, pred, ttl in _PATTERNS:
        m = rx.search(text)
        if m:
            out.append({"subject": m["s"], "predicate": pred, "object": m["o"], "text": text.strip(),
                        "valid_from": doc_ts, "ttl_seconds": ttl})
    return out


# -------------------------------------------------------------------- builders


def _probes(seed: int, cp: int, world: World, early_fams: list[str]) -> dict[str, list[Task]]:
    ts = cp * HOUR
    r = Random(seed * 1009 + cp)
    heldin = [code_task(f"p{cp}-h{i}", f, r, ts) for i, f in enumerate(TRAIN_FAMILIES * 2)]
    transfer = [code_task(f"p{cp}-t{i}", f, r, ts) for i, f in enumerate(TRANSFER_FAMILIES * 4)]
    rr = Random(seed * 7 + 1)     # retention instances are identical at every checkpoint
    retention = [code_task(f"p{cp}-r{i}", f, rr, ts) for i, f in enumerate(early_fams * 4)]
    fact: list[Task] = []
    known = [s for s in SERVICES if s in world.region and s in world.port]
    for i, svc in enumerate(r.sample(known, min(4, len(known)))):
        fact.append(endpoint_task(f"p{cp}-f{i}", svc, world, ts))
    for i, svc in enumerate(r.sample(SERVICES, 3)):
        fact.append(oncall_task(f"p{cp}-o{i}", svc, world, ts))
    return {"heldin": heldin, "transfer": transfer, "retention": retention, "fact": fact}


def build_stream(seed: int, n: int = 240, checkpoint_every: int = 40) -> Stream:
    rng = Random(seed)
    world = World()
    items: list[Task] = []
    checkpoints = list(range(0, n + 1, checkpoint_every))
    probes: dict[int, dict[str, list[Task]]] = {}
    early: list[str] = Random(seed + 99).sample(TRAIN_FAMILIES, 4)   # the "old abilities" tracked for retention
    seen_early = 0

    init_obs: list[str] = []
    for svc in SERVICES:
        init_obs.append(("region", svc))
        init_obs.append(("port", svc))
    slots = set(rng.sample(range(0, 64), len(init_obs)))
    init_iter = iter(rng.sample(init_obs, len(init_obs)))
    ports = {s: rng.choice([80, 443, 8080, 8443, 9000]) for s in SERVICES}

    def snapshot(cp: int):
        probes[cp] = _probes(seed, cp, copy.deepcopy(world), early)

    for i in range(n):
        if i in checkpoints:
            snapshot(i)
        ts = i * HOUR
        tid = f"s{seed}-{i}"
        if i in slots:
            kind, svc = next(init_iter)
            if kind == "region":
                r_ = rng.choice(REGIONS)
                world.region[svc] = r_
                items.append(_obs(tid, f"FYI: the {svc} service runs in region {r_}.", ts))
            else:
                world.port[svc] = ports[svc]
                items.append(_obs(tid, f"FYI: the {svc} service listens on port {ports[svc]}.", ts))
            continue
        roll = rng.random()
        known = [s for s in SERVICES if s in world.region and s in world.port]
        if i >= 64 and roll < 0.14:                                  # move a service
            svc = rng.choice(list(world.region))
            r_ = rng.choice([x for x in REGIONS if x != world.region[svc]])
            world.region[svc] = r_
            items.append(_obs(tid, f"Update: {svc} has been moved to region {r_}.", ts))
        elif i >= 64 and roll < 0.22:                                # on-call rotation (expires)
            svc, who = rng.choice(SERVICES), rng.choice(PEOPLE)
            world.oncall[svc] = (who, ts + ONCALL_TTL)
            items.append(_obs(tid, f"This week's on-call engineer for {svc} is {who}.", ts))
        elif i >= 64 and roll < 0.26 and world.region:               # late-arriving stale note
            svc = rng.choice(list(world.region))
            old = rng.choice([x for x in REGIONS if x != world.region[svc]])
            items.append(_obs(tid, f"Archived note: {svc} used to run in region {old}.", ts,
                              doc_ts=max(0.0, ts - 30 * 24 * HOUR)))
        elif known and roll < 0.40:
            items.append(endpoint_task(tid, rng.choice(known), world, ts))
        elif i >= 64 and roll < 0.46:
            items.append(oncall_task(tid, rng.choice(SERVICES), world, ts))
        else:
            fam = rng.choice(TRAIN_FAMILIES)
            if seen_early < 12:                       # teach the tracked families first
                fam = early[seen_early % 4]
                seen_early += 1
            items.append(code_task(tid, fam, rng, ts))
    if n in checkpoints:
        snapshot(n)
    return Stream(items, checkpoints, probes)
