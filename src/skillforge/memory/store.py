"""Reference implementation of the memory rules (episodic, semantic, procedural).

The Go service in ``memory-service/`` implements the same contract on Postgres + pgvector. Both are run against
``tests/conformance/*.json`` so they cannot drift. Times are plain floats (seconds) supplied by the caller, which
makes expiry deterministic in experiments; the service defaults to wall-clock time when ``now`` is omitted.
"""
from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from typing import Any

RRF_K = 60
MIN_SIMILARITY = 0.15
DEPRECATE_MIN_USES = 5
DEPRECATE_BELOW = 0.5


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", s.lower())


class MemoryStore:
    def __init__(self, rules: bool = True):
        self.rules = rules          # False = ablation: append-only facts, no conflict handling or expiry
        self._ids = itertools.count(1)
        self.episodes: dict[str, dict] = {}
        self.facts: list[dict] = []
        self.skills: list[dict] = []
        self.events: list[dict] = []

    # ------------------------------------------------------------------ episodic
    def add_episode(self, ep: dict) -> dict:
        eid = ep.get("id") or f"ep-{next(self._ids)}"
        rec = {**ep, "id": eid}
        self.episodes[eid] = rec
        return {"id": eid}

    def get_episode(self, eid: str) -> dict | None:
        return self.episodes.get(eid)

    def search_episodes(self, embedding: list[float], k: int = 10, only_success: bool = False) -> list[dict]:
        cands = [e for e in self.episodes.values() if e.get("embedding") and (e.get("success") or not only_success)]
        cands.sort(key=lambda e: -_cos(embedding, e["embedding"]))
        return [{k_: v for k_, v in e.items() if k_ != "embedding"} | {"score": _cos(embedding, e["embedding"])}
                for e in cands[:k]]

    # ------------------------------------------------------------------ semantic
    def _alive(self, f: dict, now: float) -> bool:
        return f["status"] == "active" and (f["expires_at"] is None or f["expires_at"] > now)

    def put_fact(self, subject: str, predicate: str, object: str, text: str = "", embedding: list[float] | None = None,
                 confidence: float = 0.8, valid_from: float | None = None, ttl_seconds: float | None = None,
                 source_episode: str | None = None, now: float = 0.0) -> dict:
        vf = now if valid_from is None else valid_from
        if not self.rules:
            ttl_seconds = None
        new = {"id": f"f-{next(self._ids)}", "subject": subject, "predicate": predicate, "object": object,
               "text": text or f"{subject} {predicate} {object}", "embedding": embedding, "confidence": confidence,
               "valid_from": vf, "expires_at": None if ttl_seconds is None else vf + ttl_seconds,
               "last_confirmed": now, "status": "active", "superseded_by": None, "source_episode": source_episode}
        if not self.rules:
            self.facts.append(new)
            return {"action": "created", "fact_id": new["id"]}
        cur = next((f for f in self.facts if f["status"] == "active" and f["subject"] == subject
                    and f["predicate"] == predicate and (f["expires_at"] is None or f["expires_at"] > now)), None)
        if cur is None:
            if new["expires_at"] is not None and new["expires_at"] <= now:
                new["status"] = "history"          # arrived already expired
                self.facts.append(new)
                return {"action": "history", "fact_id": new["id"]}
            self.facts.append(new)
            return {"action": "created", "fact_id": new["id"]}
        if cur["object"] == object:
            cur["confidence"] = min(1.0, cur["confidence"] + 0.05)
            cur["last_confirmed"] = now
            if ttl_seconds is not None and new["expires_at"] is not None:
                cur["expires_at"] = max(cur["expires_at"] or 0.0, new["expires_at"])
            return {"action": "reinforced", "fact_id": cur["id"]}
        if vf > cur["valid_from"] or (vf == cur["valid_from"] and confidence > cur["confidence"]):
            cur["status"], cur["superseded_by"] = "superseded", new["id"]
            self.facts.append(new)
            return {"action": "superseded", "fact_id": new["id"], "replaced": cur["id"]}
        if vf < cur["valid_from"]:
            new["status"] = "history"              # late-arriving stale information never wins
            self.facts.append(new)
            return {"action": "history", "fact_id": new["id"]}
        return {"action": "rejected", "fact_id": cur["id"]}

    def search_facts(self, query: str, embedding: list[float], k: int = 5, now: float = 0.0,
                     mode: str = "hybrid") -> list[dict]:
        live = [f for f in self.facts if self._alive(f, now)]
        return self._hybrid(live, query, embedding, k, lambda f: f["text"], now, mode)

    # ------------------------------------------------------------------ procedural
    def put_skill(self, name: str, source: str, docstring: str, params: list[str], embedding: list[float] | None,
                  origin_episode: str | None = None, gate: dict | None = None, now: float = 0.0) -> dict:
        prev = [s for s in self.skills if s["name"] == name]
        version = max((s["version"] for s in prev), default=0) + 1
        for s in prev:
            if s["status"] == "active":
                s["status"] = "superseded"
                self._event(now, "superseded", name, s["version"])
        rec = {"name": name, "version": version, "source": source, "docstring": docstring, "params": params,
               "embedding": embedding, "origin_episode": origin_episode, "gate": gate or {}, "status": "active",
               "uses": 0, "successes": 0, "created_at": now, "last_used": None}
        self.skills.append(rec)
        self._event(now, "created", name, version)
        return {"name": name, "version": version, "status": "active"}

    def search_skills(self, query: str, embedding: list[float], k: int = 5, now: float = 0.0) -> list[dict]:
        live = [s for s in self.skills if s["status"] == "active"]
        return self._hybrid(live, query, embedding, k, lambda s: f'{s["name"]} {s["docstring"]}', now)

    def list_skills(self, status: str | None = None) -> list[dict]:
        out = [s for s in self.skills if status is None or s["status"] == status]
        return [{k: v for k, v in s.items() if k != "embedding"} for s in out]

    def active_sources(self) -> dict[str, str]:
        return {s["name"]: s["source"] for s in self.skills if s["status"] == "active"}

    def record_usage(self, name: str, version: int, success: bool, now: float = 0.0) -> dict:
        s = next((x for x in self.skills if x["name"] == name and x["version"] == version), None)
        if s is None:
            return {"status": "unknown"}
        s["uses"] += 1
        s["successes"] += int(success)
        s["last_used"] = now
        self._event(now, "used", name, version, success=success)
        if s["status"] == "active" and s["uses"] >= DEPRECATE_MIN_USES and s["successes"] / s["uses"] < DEPRECATE_BELOW:
            s["status"] = "deprecated"
            self._event(now, "deprecated", name, version)
        return {"status": s["status"], "uses": s["uses"], "successes": s["successes"]}

    def reap(self, now: float, skill_idle_ttl: float | None = None) -> dict:
        expired = 0
        for f in self.facts:
            if f["status"] == "active" and f["expires_at"] is not None and f["expires_at"] <= now:
                f["status"] = "expired"
                expired += 1
        retired = 0
        if skill_idle_ttl:
            for s in self.skills:
                last = s["last_used"] if s["last_used"] is not None else s["created_at"]
                if s["status"] == "active" and now - last > skill_idle_ttl:
                    s["status"] = "retired"
                    self._event(now, "retired", s["name"], s["version"])
                    retired += 1
        return {"facts_expired": expired, "skills_retired": retired}

    # ------------------------------------------------------------------ shared
    def _event(self, ts: float, typ: str, name: str, version: int, **extra: Any) -> None:
        self.events.append({"ts": ts, "type": typ, "name": name, "version": version, **extra})

    def stats(self) -> dict:
        c = Counter(s["status"] for s in self.skills)
        f = Counter(x["status"] for x in self.facts)
        return {"episodes": len(self.episodes), "facts": dict(f), "skills": dict(c)}

    def _hybrid(self, items: list[dict], query: str, emb: list[float], k: int, textf, now: float,
                mode: str = "hybrid") -> list[dict]:
        if not items:
            return []
        vec = sorted((i for i in items if i.get("embedding") and _cos(emb, i["embedding"]) >= MIN_SIMILARITY),
                     key=lambda i: -_cos(emb, i["embedding"]))[:50]
        q = set(_tok(query))
        docs = [(i, _tok(textf(i))) for i in items]
        n = len(docs)
        avg = sum(len(d) for _, d in docs) / n or 1.0
        df = Counter(t for _, d in docs for t in set(d))
        scored = []
        for i, d in docs:
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg))
            if s > 0:
                scored.append((s, i))
        kw = [i for _, i in sorted(scored, key=lambda x: -x[0])[:50]]
        fused: dict[int, float] = {}
        by_id = {id(i): i for i in items}
        for ranked in ([vec] if mode == "vector" else [kw] if mode == "keyword" else [vec, kw]):
            for r, i in enumerate(ranked):
                fused[id(i)] = fused.get(id(i), 0.0) + 1.0 / (RRF_K + r + 1)
        top = sorted(fused, key=lambda x: -fused[x])[:k]
        return [{**{k_: v for k_, v in by_id[t].items() if k_ != "embedding"}, "score": fused[t]} for t in top]
