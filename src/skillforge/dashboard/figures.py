"""Static figures (PNG) for the README, drawn from ``summary.json``."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots 1-4 of the reference palette, in fixed order, one colour per condition
COLORS = {"frozen": "#2a78d6", "memory": "#eb6834", "skills": "#1baf7a", "both": "#8a5cf0"}
GREY = {"both_nogate": "#6b6a66", "both_weakgate": "#9a9893", "memory_norules": "#b4b2ac"}
LABEL = {"frozen": "Frozen", "memory": "Memory only", "skills": "Skills only", "both": "Both",
         "both_nogate": "Both, no gate", "both_weakgate": "Both, 1-case gate", "memory_norules": "Memory, no rules"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160})


def _col(c: str) -> str:
    return COLORS.get(c) or GREY.get(c, "#888")


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)


def learning_curves(s: dict, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.3, 1]})
    lc = s["learning_curves"]
    for c in ("frozen", "memory", "skills", "both"):
        d = lc[c]
        ax[0].plot(d["x"], d["acc"], color=_col(c), lw=2, marker="o", ms=4, label=LABEL[c])
        ax[0].fill_between(d["x"], d["lo"], d["hi"], color=_col(c), alpha=0.12, lw=0)
        ax[1].plot(d["x"], [t / 1000 if t else None for t in d["tokens"]], color=_col(c), lw=2, marker="o", ms=4)
    ax[0].set_title("Accuracy on the online task stream", loc="left", color=INK, fontsize=11)
    ax[0].set_xlabel("stream step"); ax[0].set_ylabel("accuracy (20-step window)"); ax[0].set_ylim(0.3, 1.02)
    ax[0].legend(frameon=False, loc="lower right")
    ax[1].set_title("Tokens per task (incl. learning overhead)", loc="left", color=INK, fontsize=11)
    ax[1].set_xlabel("stream step"); ax[1].set_ylabel("thousand tokens")
    fig.suptitle(f"Mean over {len(s['seeds'])} seeds, 95% bootstrap CI  ·  LLM backend: {s['llm']}", x=0.01, ha="left",
                 color=MUTED, fontsize=9, y=0.02)
    _save(fig, out, "learning_curves.png")


def cost_curve(s: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for c in ("frozen", "memory", "skills", "both"):
        d = s["learning_curves"][c]
        ax.plot(d["x"], [x * 1000 if x else None for x in d["cost"]], color=_col(c), lw=2, marker="o", ms=4, label=LABEL[c])
    ax.set_title("Cost per task (USD x 1000)", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("stream step"); ax.set_ylabel("$ per 1000 tasks")
    ax.legend(frameon=False)
    _save(fig, out, "cost_curve.png")


def probes(s: dict, out: Path) -> None:
    sets = [("heldin", "Trained families, fresh instances"), ("transfer", "Transfer: unseen composite families"),
            ("retention", "Retention: earliest-taught families"), ("fact", "Facts about the changing world")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (key, title) in zip(axes.flat, sets):
        for c in ("frozen", "memory", "skills", "both"):
            pts = s["probes"][c][key]
            x = [p["cp"] for p in pts]
            ax.plot(x, [p["mean"] for p in pts], color=_col(c), lw=2, marker="o", ms=4, label=LABEL[c])
            ax.fill_between(x, [p["lo"] for p in pts], [p["hi"] for p in pts], color=_col(c), alpha=0.1, lw=0)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_ylim(-0.02, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("checkpoint (stream step)")
    axes[0][0].legend(frameon=False, loc="lower right")
    fig.text(0.01, 0.0, "Read-only probes: no learning happens while they run. Same tasks for every condition.",
             color=MUTED, fontsize=9)
    fig.tight_layout()
    _save(fig, out, "probes.png")


def harm(s: dict, out: Path) -> None:
    h = s["harm"]
    order = [c for c in ("memory", "skills", "both", "both_weakgate", "both_nogate") if c in h]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    y = range(len(order))
    ax.barh([i + 0.2 for i in y], [h[c]["helped"] for c in order], height=0.38, color="#1baf7a", label="fixed a task the frozen agent failed")
    ax.barh([i - 0.2 for i in y], [h[c]["harmed"] for c in order], height=0.38, color="#e34948", label="broke a task the frozen agent solved")
    ax.set_yticks(list(y)); ax.set_yticklabels([LABEL[c] for c in order]); ax.invert_yaxis()
    for i, c in enumerate(order):
        ax.text(h[c]["helped"] + 1, i + 0.2, str(h[c]["helped"]), va="center", color=MUTED, fontsize=9)
        ax.text(h[c]["harmed"] + 1, i - 0.2, str(h[c]["harmed"]), va="center", color=MUTED, fontsize=9)
    ax.set_xlabel("code tasks, summed over all seeds (paired by task)")
    ax.set_title("Where self-evolution helped and where it hurt", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    _save(fig, out, "harm.png")


def library(s: dict, out: Path) -> None:
    lib = s["library"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    d = lib["both"]
    x = range(len(d["active"]))
    ax[0].plot(x, d["active"], color="#1baf7a", lw=2, label="active")
    ax[0].plot(x, d["promoted"], color="#2a78d6", lw=2, ls="--", label="promoted (cumulative)")
    ax[0].plot(x, d["deprecated"], color="#e34948", lw=2, label="deprecated")
    ax[0].plot(x, d["superseded"], color="#eda100", lw=2, label="superseded by a new version")
    ax[0].set_title("Skill library growth ('Both', mean of seeds)", loc="left", color=INK, fontsize=11)
    ax[0].set_xlabel("stream step"); ax[0].set_ylabel("skills"); ax[0].legend(frameon=False, fontsize=9)
    names = [c for c in ("skills", "both", "both_weakgate", "both_nogate") if c in lib]
    w = 0.2
    for j, kind, col in ((0, "promoted", "#1baf7a"), (1, "rejected", "#e34948"), (2, "insufficient", "#9a9893"), (3, "redundant", "#2a78d6")):
        ax[1].bar([i + (j - 1.5) * w for i in range(len(names))], [lib[c][kind][-1] for c in names], w * 0.9, color=col, label=kind)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels([LABEL[c] for c in names], fontsize=8)
    ax[1].set_title("What the gate decided (per run)", loc="left", color=INK, fontsize=11)
    ax[1].legend(frameon=False, fontsize=9); ax[1].grid(axis="x", visible=False)
    _save(fig, out, "library.png")


def longmemeval(lme: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    modes = ("vector", "keyword", "hybrid")
    cols = {"vector": "#2a78d6", "keyword": "#eb6834", "hybrid": "#1baf7a"}
    for ax, emb in zip(axes, ("hash", "neural")):
        if emb not in lme:
            continue
        r = lme[emb]
        ks = [1, 3, 5, 10]
        for m in modes:
            v = [r["modes"][m]["overall"][f"recall@{k}"]["mean"] for k in ks]
            lo = [r["modes"][m]["overall"][f"recall@{k}"]["lo"] for k in ks]
            hi = [r["modes"][m]["overall"][f"recall@{k}"]["hi"] for k in ks]
            ax.plot(ks, v, color=cols[m], lw=2, marker="o", ms=4, label=m)
            ax.fill_between(ks, lo, hi, color=cols[m], alpha=0.1, lw=0)
        ax.plot(ks, [r["random_baseline"][f"recall@{k}"] for k in ks], color="#9a9893", ls=":", label="random")
        ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_title(f"{'Hashing' if emb == 'hash' else 'Neural (bge-small)'} embedder, n={r['questions']}", loc="left",
                     color=INK, fontsize=11)
        ax.set_xlabel("k (sessions)"); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("recall_any@k"); axes[0].legend(frameon=False, loc="lower right")
    _save(fig, out, "longmemeval.png")


def make_all(results: str, out: str) -> list[str]:
    r, o = Path(results), Path(out)
    o.mkdir(parents=True, exist_ok=True)
    s = json.loads((r / "main" / "summary.json").read_text())
    for f in (learning_curves, cost_curve, probes, harm, library):
        f(s, o)
    lme = r / "longmemeval.json"
    if lme.exists():
        longmemeval(json.loads(lme.read_text()), o)
    return sorted(p.name for p in o.glob("*.png"))
