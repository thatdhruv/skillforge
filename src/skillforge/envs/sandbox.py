"""Sandboxed Python execution used by the agent's ``run_python`` tool and by the skill gate.

Isolation here is defence in depth, not a security boundary: an import allow-list and an AST screen
(no ``open``/``exec``/dunder access), a wall-clock timeout and an isolated interpreter (``-I -S``).
In the Kubernetes deployment the sandbox additionally runs in a pod with a read-only root filesystem,
no service-account token and a deny-all NetworkPolicy (see ``deploy/helm``).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field

ALLOWED_IMPORTS = {
    "re", "math", "json", "itertools", "collections", "datetime", "string", "functools", "heapq", "bisect",
    "statistics", "decimal", "fractions", "operator", "typing", "csv", "textwrap", "io", "skills", "calendar",
}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals",
                   "vars", "getattr", "setattr", "delattr", "exit", "quit"}
MAX_OUTPUT = 4000

_RUNNER = r'''
import sys, json, types, traceback
payload = json.loads(sys.stdin.read())
skills_src = payload["skills"]
used = []
cache = {}
mod = types.ModuleType("skills")
def _load(name):
    if name in cache:
        return cache[name]
    ns = {"__name__": "skill_" + name}
    exec(compile(skills_src[name], "<skill:%s>" % name, "exec"), ns)
    fn = ns[name]
    def wrapper(*a, **k):
        used.append(name)
        return fn(*a, **k)
    wrapper.__doc__ = fn.__doc__
    wrapper.__name__ = name
    cache[name] = wrapper
    return wrapper
def _ga(name):
    if name in skills_src:
        return _load(name)
    raise AttributeError(name)
mod.__getattr__ = _ga
sys.modules["skills"] = mod
code = payload["code"]
try:
    exec(compile(code, "<agent>", "exec"), {"INPUT": payload["inputs"], "__name__": "__main__"})
except SystemExit:
    pass
except BaseException:
    tb = traceback.format_exc().strip().splitlines()
    sys.stderr.write("\n".join(tb[-4:]) + "\n")
    sys.stderr.write("__EXC__\n")
finally:
    sys.stdout.flush()
    sys.stderr.write("__SKILLS_USED__" + json.dumps(used) + "\n")
'''


@dataclass
class RunResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    skills_used: list[str] = field(default_factory=list)
    seconds: float = 0.0
    timed_out: bool = False
    violation: str | None = None

    @property
    def answer(self) -> str:
        lines = [ln for ln in self.stdout.strip().splitlines() if ln.strip()]
        return lines[-1].strip() if lines else ""


def screen(code: str) -> str | None:
    """Return a human-readable violation or ``None``."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{a.name}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"access to dunder attribute '{node.attr}' is not allowed"
    return None


def run_python(code: str, inputs: dict | None = None, skills: dict[str, str] | None = None,
               timeout: float = 5.0) -> RunResult:
    bad = screen(code)
    if bad:
        return RunResult(ok=False, stderr=bad, violation=bad)
    payload = json.dumps({"code": code, "inputs": inputs or {}, "skills": skills or {}})
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", _RUNNER], input=payload, capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return RunResult(ok=False, stderr="timeout", timed_out=True, seconds=time.perf_counter() - t0)
    secs = time.perf_counter() - t0
    err = p.stderr
    used: list[str] = []
    keep = []
    for ln in err.splitlines():
        if ln.startswith("__SKILLS_USED__"):
            used = json.loads(ln[len("__SKILLS_USED__"):])
        else:
            keep.append(ln)
    stderr = "\n".join(keep)
    failed = "__EXC__" in stderr or p.returncode != 0
    stderr = stderr.replace("__EXC__", "").strip()
    return RunResult(ok=not failed, stdout=p.stdout[:MAX_OUTPUT], stderr=stderr[:MAX_OUTPUT], skills_used=used,
                     seconds=secs)
