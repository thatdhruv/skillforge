"""Turn raw run files into the numbers behind every figure: curves, transfer, retention, cost, harm."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

import numpy as np

from ..envs.families import FAMILIES
from ..llm.base import cost_usd
from ..skills.gate import run_cases

SCORED = ("code", "fact")
BIN = 20


def load_runs(d: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(d).glob("*_s[0-9]*.json"))]


def _ci(per_seed: list[float]) -> tuple[float, float, float]:
    a = np.array(per_seed, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boots = [rng.choice(a, len(a)).mean() for _ in range(1000)]
    return float(a.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _row_cost(r: dict, model: str) -> float:
    return r["cost"] + cost_usd(model, r.get("learn_tokens_in", 0), r.get("learn_tokens_out", 0))


def _by(runs: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in runs:
        out[r["cond"]].append(r)
    return out


def learning_curves(by: dict, n: int) -> dict:
    out = {}
    for cond, runs in by.items():
        bins = list(range(0, n, BIN))
        acc, tok, cost, accv = [], [], [], []
        for b in bins:
            a_s, t_s, c_s = [], [], []
            for run in runs:
                rows = [r for r in run["rows"] if r["phase"] == "stream" and r["kind"] in SCORED
                        and b <= r["step"] < b + BIN]
                if rows:
                    a_s.append(np.mean([r["success"] for r in rows]))
                    t_s.append(np.mean([r["tokens_in"] + r["tokens_out"] + r.get("learn_tokens_in", 0)
                                        + r.get("learn_tokens_out", 0) for r in rows]))
                    c_s.append(np.mean([_row_cost(r, run["llm"]) for r in rows]))
            acc.append(_ci(a_s))
            tok.append(float(np.mean(t_s)) if t_s else None)
            cost.append(float(np.mean(c_s)) if c_s else None)
        out[cond] = {"x": [b + BIN / 2 for b in bins], "acc": [a[0] for a in acc], "lo": [a[1] for a in acc],
                     "hi": [a[2] for a in acc], "tokens": tok, "cost": cost}
    return out


def probe_curves(by: dict) -> dict:
    out: dict = {}
    for cond, runs in by.items():
        sets = sorted({r["set"] for run in runs for r in run["rows"] if r["phase"] == "probe"})
        out[cond] = {}
        for s in sets:
            cps = sorted({r["cp"] for run in runs for r in run["rows"] if r["phase"] == "probe" and r["set"] == s})
            pts = []
            for cp in cps:
                per_seed = []
                for run in runs:
                    rows = [r for r in run["rows"] if r["phase"] == "probe" and r["set"] == s and r["cp"] == cp]
                    if rows:
                        per_seed.append(np.mean([r["success"] for r in rows]))
                m, lo, hi = _ci(per_seed)
                pts.append({"cp": cp, "mean": m, "lo": lo, "hi": hi})
            out[cond][s] = pts
    return out


def paired_diff(by: dict, cond: str, base: str, pset: str, cp: int) -> dict:
    """Cluster bootstrap over seeds of (cond - base) accuracy on identical probe tasks."""
    def per_seed(c):
        return {run["seed"]: np.mean([r["success"] for r in run["rows"]
                                       if r["phase"] == "probe" and r["set"] == pset and r["cp"] == cp])
                for run in by[c]}
    a, b = per_seed(cond), per_seed(base)
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    rng = np.random.default_rng(1)
    boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    return {"mean": float(d.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "seeds": len(d)}


def overall(by: dict) -> dict:
    out = {}
    for cond, runs in by.items():
        model = runs[0]["llm"]
        half = runs[0]["n"] // 2
        rec: dict = {}
        for name, lo in (("stream_all", 0), ("stream_second_half", half)):
            per = [np.mean([r["success"] for r in run["rows"] if r["phase"] == "stream" and r["kind"] in SCORED
                            and r["step"] >= lo]) for run in runs]
            rec[name] = _ci(per)
        for kind in SCORED:
            per = [np.mean([r["success"] for r in run["rows"] if r["phase"] == "stream" and r["kind"] == kind
                            and r["step"] >= half]) for run in runs]
            rec[f"{kind}_second_half"] = _ci(per)
        rows = [r for run in runs for r in run["rows"] if r["phase"] == "stream" and r["kind"] in SCORED]
        rec["tokens_per_task"] = float(np.mean([r["tokens_in"] + r["tokens_out"] + r.get("learn_tokens_in", 0)
                                                + r.get("learn_tokens_out", 0) for r in rows]))
        rec["cost_per_task"] = float(np.mean([_row_cost(r, model) for r in rows]))
        rec["learning_overhead_share"] = float(
            sum(cost_usd(model, r.get("learn_tokens_in", 0), r.get("learn_tokens_out", 0)) for r in rows)
            / max(1e-12, sum(_row_cost(r, model) for r in rows)))
        out[cond] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in rec.items()}
    return out


# ------------------------------------------------------------------ harm analysis (paired against the frozen agent)


def _origin_family(run: dict, skill_name: str) -> str | None:
    fam_of = {r["task_id"]: r["family"] for r in run["rows"]}
    cands = [s for s in run["skills"] if s["name"] == skill_name]
    if not cands:
        return None
    return fam_of.get(cands[-1].get("origin_episode"))


def harm(by: dict) -> dict:
    frozen = {(r["seed"], r["task_id"]): r for run in by["frozen"] for r in run["rows"] if r["kind"] == "code"}
    out, examples = {}, []
    for cond, runs in by.items():
        if cond == "frozen":
            continue
        helped = harmed = 0
        causes: Counter = Counter()
        for run in runs:
            for r in run["rows"]:
                if r["kind"] != "code":
                    continue
                f = frozen.get((r["seed"], r["task_id"]))
                if f is None:
                    continue
                if r["success"] and not f["success"]:
                    helped += 1
                elif f["success"] and not r["success"]:
                    harmed += 1
                    if r["skills_used"]:
                        fam = _origin_family(run, r["skills_used"][0])
                        cause = "flawed skill passed the gate" if fam == r["family"] else "right-looking skill, wrong task"
                    elif r["offered"]:
                        cause = "other (skill library present)"
                    else:
                        cause = "other"
                    causes[cause] += 1
                    if len(examples) < 400:
                        examples.append({"cond": cond, "seed": r["seed"], "task_id": r["task_id"], "family": r["family"],
                                         "phase": r["phase"], "cause": cause, "skills_used": r["skills_used"],
                                         "answer": r["answer"], "expected": r["expected"], "code": r["code"],
                                         "frozen_code": f["code"]})
        total = sum(1 for run in runs for r in run["rows"] if r["kind"] == "code")
        out[cond] = {"helped": helped, "harmed": harmed, "net": helped - harmed, "tasks": total,
                     "harm_rate": harmed / max(1, total), "causes": dict(causes)}
    return {"summary": out, "examples": examples}


def memory_failures(by: dict) -> dict:
    out = {}
    for cond, runs in by.items():
        if "memory" not in cond and "both" not in cond:
            continue
        c: Counter = Counter()
        for run in runs:
            for r in run["rows"]:
                if r["kind"] != "fact":
                    continue
                c["total"] += 1
                if r["success"]:
                    c["correct"] += 1
                elif r["answer"].strip() == "unknown":
                    c["abstained (not found or expired)"] += 1
                else:
                    c["stale or wrong value"] += 1
        out[cond] = dict(c)
    return out


def skill_audit(runs: list[dict], per_family: int = 25) -> dict:
    """Ground-truth precision of promoted skills: run each on fresh instances of its origin family."""
    res: dict = defaultdict(lambda: {"skills": 0, "flawed": 0})
    for run in runs:
        if run["cond"] in ("frozen", "memory", "memory_norules"):
            continue
        fam_of = {r["task_id"]: r["family"] for r in run["rows"]}
        sources = {s["name"]: s["source"] for s in run["skills"]}
        for s in run["skills"]:
            fam = fam_of.get(s.get("origin_episode"))
            if fam not in FAMILIES:
                continue
            f, rng = FAMILIES[fam], Random(1234)
            cases = []
            for _ in range(per_family):
                inp = f.gen(rng)
                cases.append({"inputs": inp, "answer": str(f.truth(**inp))})
            out = run_cases(s["source"], s["name"], cases, sources, 10.0)
            ok = sum(o == c["answer"] for o, c in zip(out, cases)) / len(cases)
            res[run["cond"]]["skills"] += 1
            res[run["cond"]]["flawed"] += int(ok < 1.0)
    return {k: {**v, "flawed_share": v["flawed"] / max(1, v["skills"])} for k, v in res.items()}


def library(by: dict) -> dict:
    out = {}
    for cond, runs in by.items():
        if not runs[0]["timeline"] or not any(t["promoted"] or t["active"] for r in runs for t in r["timeline"]):
            continue
        steps = len(runs[0]["timeline"])
        keys = ("active", "deprecated", "superseded", "promoted", "rejected", "insufficient", "redundant")
        out[cond] = {k: [float(np.mean([r["timeline"][i][k] for r in runs])) for i in range(steps)] for k in keys}
        final = [s for r in runs for s in r["skills"]]
        out[cond]["final"] = {
            "skills_created": len(final) / len(runs),
            "active": sum(s["status"] == "active" for s in final) / len(runs),
            "deprecated": sum(s["status"] == "deprecated" for s in final) / len(runs),
            "mean_uses": float(np.mean([s["uses"] for s in final])) if final else 0.0,
        }
    return out


def summarize(d: str | Path, audit: bool = True) -> dict:
    runs = load_runs(d)
    by = _by(runs)
    n = runs[0]["n"]
    cps = sorted({r["cp"] for r in runs[0]["rows"] if r["phase"] == "probe"})
    last = cps[-1]
    summary = {
        "llm": runs[0]["llm"], "seeds": sorted({r["seed"] for r in runs}), "n": n, "checkpoints": cps,
        "conditions": sorted(by), "overall": overall(by), "learning_curves": learning_curves(by, n),
        "probes": probe_curves(by), "library": library(by), "memory_failures": memory_failures(by),
    }
    harm_res = harm(by) if "frozen" in by else {"summary": {}, "examples": []}
    summary["harm"] = harm_res["summary"]
    if audit:
        summary["skill_audit"] = skill_audit(runs)
    if "frozen" in by:
        summary["vs_frozen"] = {
            c: {s: paired_diff(by, c, "frozen", s, last) for s in ("heldin", "transfer", "retention", "fact")}
            for c in by if c != "frozen"}
    p = Path(d)
    (p / "summary.json").write_text(json.dumps(summary, indent=1))
    (p / "harm_cases.json").write_text(json.dumps(harm_res["examples"], indent=1))
    return summary
