"""Run JSON conformance fixtures against any ``Memory`` implementation."""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES = sorted((Path(__file__).parent / "conformance").glob("*.json"))


def run_fixture(mem, fx: dict) -> None:
    for i, step in enumerate(fx["steps"]):
        op, a, exp = step["op"], dict(step["args"]), step["expect"]
        where = f'{fx["name"]}#{i} {op}'
        if op == "put_fact":
            r = mem.put_fact(a.pop("subject"), a.pop("predicate"), a.pop("object"), a.pop("text"), **a)
            assert r["action"] == exp["action"], where + f" -> {r}"
        elif op == "search_facts":
            r = mem.search_facts(a["query"], a["k"], a["now"])
            check_list(r, exp, where)
        elif op == "put_skill":
            r = mem.put_skill(a.pop("name"), a.pop("source"), a.pop("docstring"), a.pop("params"), **a)
            for k, v in exp.items():
                assert r[k] == v, where + f" {k}: {r}"
        elif op == "list_skills":
            r = mem.list_skills(a.get("status"))
            assert [s["name"] for s in r] == exp["names"], where
            assert [s["version"] for s in r] == exp["versions"], where
        elif op == "search_skills":
            check_list(mem.search_skills(a["query"], a["k"], a["now"]), exp, where)
        elif op == "record_usage":
            r = mem.record_usage(a["name"], a["version"], a["success"], a["now"])
            assert r["status"] == exp["status"], where + f" -> {r}"
        else:
            raise AssertionError(f"unknown op {op}")


def check_list(r, exp, where):
    assert len(r) == exp["count"], where + f" count {len(r)}"
    for k, v in exp.get("top", {}).items():
        assert r[0][k] == v, where + f" top.{k}={r[0][k]}"
