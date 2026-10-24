"""Distiller: finished trajectories -> candidate skills (executable Python functions with docstrings).

Consumes ``trajectories.finished``; publishes ``skills.candidates``. Only successful trajectories whose final code
defines ``solve`` are eligible. Naming and documenting is pluggable: a heuristic namer works offline, an LLM namer
(`LLMNamer`) is used with a real model.
"""
from __future__ import annotations

import ast
import json
import re

from ..bus import CANDIDATES, TRAJECTORIES, Bus
from ..llm.base import LLM, Usage
from ..llm.textutil import STOP, stem_of


class HeuristicNamer:
    def name_and_doc(self, prompt: str, code: str) -> tuple[str, str]:
        stem = stem_of(prompt)
        first = re.split(r"(?<=[.?!])\s", stem)[0]
        toks = [w for w in re.findall(r"[a-z]+", first.lower()) if w not in STOP and len(w) > 2][:4]
        name = "_".join(toks) or "skill"
        return name, first.strip()


class LLMNamer:
    """Asks a real model for a name and a docstring. Token use is recorded in ``usage`` (learning overhead)."""

    def __init__(self, llm: LLM):
        self.llm, self.usage = llm, Usage()

    def name_and_doc(self, prompt: str, code: str) -> tuple[str, str]:
        msg = ("Name and document a reusable Python function that generalises this solved task. Reply with JSON only: "
               '{"name": "snake_case_name", "docstring": "one sentence stating what it computes and its inputs"}\n\n'
               f"TASK:\n{stem_of(prompt)}\n\nSOLUTION:\n{code}")
        r = self.llm.chat("You write precise function names and docstrings.", [{"role": "user", "content": msg}], [])
        self.usage.add(r)
        try:
            d = json.loads(r.text[r.text.index("{"): r.text.rindex("}") + 1])
            name = re.sub(r"\W+", "_", d["name"]).strip("_").lower() or "skill"
            return name, d["docstring"].strip()
        except Exception:
            return HeuristicNamer().name_and_doc(prompt, code)


def extract_skill(code: str, name: str, doc: str) -> tuple[str, list[str]] | None:
    """Keep imports and function definitions, rename ``solve`` to ``name`` and attach the docstring."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    keep: list[ast.stmt] = []
    entry: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
        elif isinstance(node, ast.FunctionDef):
            keep.append(node)
            if node.name == "solve":
                entry = node
    if entry is None:
        return None
    params = [a.arg for a in entry.args.args]

    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self, n):
            self.generic_visit(n)
            if n.name == "solve":
                n.name = name
            return n

        def visit_Name(self, n):
            if n.id == "solve":
                n.id = name
            return n

    mod = ast.Module(body=[Rename().visit(n) for n in keep], type_ignores=[])
    ast.fix_missing_locations(mod)
    for n in mod.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            body = n.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body = body[1:]
            n.body = [ast.Expr(ast.Constant(doc))] + body
    return ast.unparse(mod) + "\n", params


class Distiller:
    def __init__(self, bus: Bus, namer=None, group: str = "distiller"):
        self.bus, self.namer = bus, namer or HeuristicNamer()
        self.consumer = bus.consumer(TRAJECTORIES, group)

    def step(self, max_messages: int = 100, timeout: float = 0.0) -> int:
        n = 0
        for t in self.consumer.poll(max_messages, timeout):
            if t.get("kind") != "code" or not t.get("success") or not t.get("final_code"):
                continue
            name, doc = self.namer.name_and_doc(t["prompt"], t["final_code"])
            got = extract_skill(t["final_code"], name, doc)
            if got is None:
                continue
            source, params = got
            self.bus.publish(CANDIDATES, {
                "name": name, "source": source, "docstring": doc, "params": params, "prompt": t["prompt"],
                "origin_episode": t["task_id"], "origin_case": {"inputs": t["inputs"], "answer": t["answer"]},
                "ts": t.get("ts", 0.0)}, key=name)
            n += 1
        self.consumer.commit()
        return n
