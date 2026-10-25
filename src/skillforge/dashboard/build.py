"""Bake summary.json (+ LongMemEval, harm cases) into one self-contained dashboard.html."""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template.html")


def build(results: str, out: str, run: str = "main") -> str:
    r = Path(results)
    data = {
        "summary": json.loads((r / run / "summary.json").read_text()),
        "harm_cases": json.loads((r / run / "harm_cases.json").read_text()),
        "longmemeval": json.loads((r / "longmemeval.json").read_text()) if (r / "longmemeval.json").exists() and run == "main" else None,
    }
    # keep the page small: examples only carry what the page renders
    data["harm_cases"] = [{k: e[k] for k in ("cond", "family", "cause", "skills_used", "answer", "expected")}
                          for e in data["harm_cases"][:40]]
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html, encoding="utf-8")
    return out
