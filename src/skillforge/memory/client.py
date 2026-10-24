"""Memory clients: ``LocalMemory`` (in-process reference) and ``HttpMemory`` (Go service + Postgres/pgvector).

Both expose the same methods and embed text client-side, so they are interchangeable.
"""
from __future__ import annotations

from typing import Protocol

import httpx

from .embed import HashEmbedder
from .store import MemoryStore


class Memory(Protocol):
    def add_episode(self, ep: dict, text: str) -> str: ...
    def similar_episodes(self, text: str, k: int = 10, only_success: bool = True) -> list[dict]: ...
    def put_fact(self, subject: str, predicate: str, object: str, text: str, **kw) -> dict: ...
    def search_facts(self, query: str, k: int = 5, now: float = 0.0) -> list[dict]: ...
    def put_skill(self, name: str, source: str, docstring: str, params: list[str], **kw) -> dict: ...
    def search_skills(self, query: str, k: int = 5, now: float = 0.0) -> list[dict]: ...
    def list_skills(self, status: str | None = None) -> list[dict]: ...
    def active_sources(self) -> dict[str, str]: ...
    def record_usage(self, name: str, version: int, success: bool, now: float = 0.0) -> dict: ...
    def reap(self, now: float, skill_idle_ttl: float | None = None) -> dict: ...
    def stats(self) -> dict: ...
    def events(self) -> list[dict]: ...


class LocalMemory:
    def __init__(self, embedder: HashEmbedder | None = None):
        self.store = MemoryStore()
        self.emb = embedder or HashEmbedder()

    def add_episode(self, ep: dict, text: str) -> str:
        return self.store.add_episode({**ep, "embedding": self.emb.embed(text)})["id"]

    def similar_episodes(self, text, k=10, only_success=True):
        return self.store.search_episodes(self.emb.embed(text), k, only_success)

    def put_fact(self, subject, predicate, object, text, **kw):
        return self.store.put_fact(subject, predicate, object, text, self.emb.embed(text), **kw)

    def search_facts(self, query, k=5, now=0.0):
        return self.store.search_facts(query, self.emb.embed(query), k, now)

    def put_skill(self, name, source, docstring, params, **kw):
        return self.store.put_skill(name, source, docstring, params, self.emb.embed(f"{name} {docstring}"), **kw)

    def search_skills(self, query, k=5, now=0.0):
        return self.store.search_skills(query, self.emb.embed(query), k, now)

    def list_skills(self, status=None):
        return self.store.list_skills(status)

    def active_sources(self):
        return self.store.active_sources()

    def record_usage(self, name, version, success, now=0.0):
        return self.store.record_usage(name, version, success, now)

    def reap(self, now, skill_idle_ttl=None):
        return self.store.reap(now, skill_idle_ttl)

    def stats(self):
        return self.store.stats()

    def events(self):
        return list(self.store.events)


class HttpMemory:
    def __init__(self, base_url: str, embedder: HashEmbedder | None = None, timeout: float = 20.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)
        self.emb = embedder or HashEmbedder()

    def _post(self, path: str, body: dict) -> dict | list:
        r = self.http.post(path, json=body)
        r.raise_for_status()
        return r.json()

    def add_episode(self, ep, text):
        return self._post("/v1/episodes", {**ep, "embedding": self.emb.embed(text)})["id"]

    def similar_episodes(self, text, k=10, only_success=True):
        return self._post("/v1/episodes/search", {"embedding": self.emb.embed(text), "k": k,
                                                   "only_success": only_success})

    def put_fact(self, subject, predicate, object, text, **kw):
        return self._post("/v1/facts", {"subject": subject, "predicate": predicate, "object": object, "text": text,
                                        "embedding": self.emb.embed(text), **kw})

    def search_facts(self, query, k=5, now=0.0):
        return self._post("/v1/facts/search", {"query": query, "embedding": self.emb.embed(query), "k": k,
                                                "now": now})

    def put_skill(self, name, source, docstring, params, **kw):
        return self._post("/v1/skills", {"name": name, "source": source, "docstring": docstring, "params": params,
                                          "embedding": self.emb.embed(f"{name} {docstring}"), **kw})

    def search_skills(self, query, k=5, now=0.0):
        return self._post("/v1/skills/search", {"query": query, "embedding": self.emb.embed(query), "k": k,
                                                 "now": now})

    def list_skills(self, status=None):
        r = self.http.get("/v1/skills", params={"status": status} if status else None)
        r.raise_for_status()
        return r.json()

    def active_sources(self):
        return {s["name"]: s["source"] for s in self.list_skills("active")}

    def record_usage(self, name, version, success, now=0.0):
        return self._post(f"/v1/skills/{name}/usage", {"version": version, "success": success, "now": now})

    def reap(self, now, skill_idle_ttl=None):
        return self._post("/v1/admin/reap", {"now": now, "skill_idle_ttl": skill_idle_ttl})

    def stats(self):
        r = self.http.get("/v1/stats")
        r.raise_for_status()
        return r.json()

    def reset(self):
        self.http.post("/v1/admin/reset").raise_for_status()

    def events(self):
        r = self.http.get("/v1/skills/events")
        r.raise_for_status()
        return r.json()
