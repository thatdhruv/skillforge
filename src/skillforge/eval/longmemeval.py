"""Memory retrieval on LongMemEval (Wu et al., 2024; ``longmemeval_s_cleaned``): ~50 chat sessions per question.

Every turn of the haystack is written to the memory service as a fact; the question is the query. We report
session-level ``recall_any@k`` (is any evidence session among the top-k distinct sessions retrieved?) and turn-level
hit@10 (is an evidence turn, flagged ``has_answer``, in the top 10 turns?), for vector-only, keyword-only and
hybrid (RRF) retrieval, and for a hashing embedder vs a neural one. This measures the *retrieval* half of memory;
end-to-end answer accuracy needs an LLM reader and judge (see ``--qa``) and is not reported without one.
"""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import Random

import numpy as np

from ..memory.embed import FastEmbedder, HashEmbedder
from ..memory.store import MemoryStore

KS = (1, 3, 5, 10)
MODES = ("vector", "keyword", "hybrid")


def load(path: str, per_type: int | None, seed: int = 0) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = [q for q in data if not q["question_id"].endswith("_abs")]
    if per_type:
        rng, by = Random(seed), defaultdict(list)
        for q in data:
            by[q["question_type"]].append(q)
        data = [q for t in sorted(by) for q in rng.sample(by[t], min(per_type, len(by[t])))]
    return data


def _turns(q: dict):
    for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
        for i, t in enumerate(sess):
            yield sid, i, f"{t['role']}: {t['content']}"[:1200], bool(t.get("has_answer"))


def _make_embedder(kind: str):
    return FastEmbedder() if kind == "neural" else HashEmbedder()


def eval_chunk(args) -> list[dict]:
    qs, emb_kind, modes = args
    emb = _make_embedder(emb_kind)
    out = []
    for q in qs:
        turns = list(_turns(q))
        texts = [t[2] for t in turns]
        vecs = emb.embed_many(texts) if hasattr(emb, "embed_many") else [emb.embed(t) for t in texts]
        store = MemoryStore()
        for (sid, i, text, _), v in zip(turns, vecs):
            store.put_fact(sid, f"turn{i}", "x", text, v, now=0.0)
        gold_turns = {(sid, f"turn{i}") for sid, i, _, h in turns if h}
        qv = emb.embed(q["question"])
        gold = set(q["answer_session_ids"])
        rec = {"id": q["question_id"], "type": q["question_type"], "sessions": len(q["haystack_session_ids"])}
        for mode in modes:
            hits = store.search_facts(q["question"], qv, 60, 0.0, mode)
            sessions: list[str] = []
            for h in hits:
                if h["subject"] not in sessions:
                    sessions.append(h["subject"])
            for k in KS:
                rec[f"{mode}@{k}"] = int(bool(gold & set(sessions[:k])))
            rec[f"{mode}:turn@10"] = int(bool(gold_turns & {(h["subject"], h["predicate"]) for h in hits[:10]}))
        out.append(rec)
    return out


def run(path: str, per_type: int | None, emb_kind: str, workers: int, modes=MODES) -> dict:
    qs = load(path, per_type)
    chunks = [qs[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(eval_chunk, [(c, emb_kind, modes) for c in chunks if c]) for r in part]
    return aggregate(rows, modes, emb_kind)


def aggregate(rows: list[dict], modes, emb_kind: str) -> dict:
    types = sorted({r["type"] for r in rows})
    rng = np.random.default_rng(0)

    def stat(sel, key):
        v = np.array([r[key] for r in sel], dtype=float)
        boots = [rng.choice(v, len(v)).mean() for _ in range(500)]
        return {"mean": float(v.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}

    res = {"embedder": emb_kind, "questions": len(rows), "types": types, "modes": {}}
    for mode in modes:
        m = {"overall": {f"recall@{k}": stat(rows, f"{mode}@{k}") for k in KS}}
        m["overall"]["turn_hit@10"] = stat(rows, f"{mode}:turn@10")
        m["by_type"] = {t: {f"recall@{k}": stat([r for r in rows if r["type"] == t], f"{mode}@{k}")["mean"] for k in KS}
                        for t in types}
        res["modes"][mode] = m
    res["random_baseline"] = {f"recall@{k}": float(np.mean([min(1.0, k / r["sessions"]) for r in rows])) for k in KS}
    return res
