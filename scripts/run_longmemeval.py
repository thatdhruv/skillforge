"""python scripts/run_longmemeval.py --data data/longmemeval_s_cleaned.json --out results/longmemeval.json"""
import argparse
import json
import time
from pathlib import Path

from skillforge.eval.longmemeval import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    ap.add_argument("--out", default="results/longmemeval.json")
    ap.add_argument("--neural-per-type", type=int, default=25)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = {}
    t = time.time()
    out["hash"] = run(a.data, None, "hash", a.workers)
    print("hash done", round(time.time() - t), "s", flush=True)
    out["neural"] = run(a.data, a.neural_per_type, "neural", 4)
    print("neural done", round(time.time() - t), "s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
