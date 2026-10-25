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
    ap.add_argument("--neural-per-type", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    t = time.time()
    if "hash" not in out:
        out["hash"] = run(a.data, None, "hash", a.workers)
        out_path.write_text(json.dumps(out, indent=1))
        print("hash done", round(time.time() - t), "s", flush=True)
    if a.neural_per_type:
        out["neural"] = run(a.data, a.neural_per_type, "neural", 3)
        out_path.write_text(json.dumps(out, indent=1))
        print("neural done", round(time.time() - t), "s", flush=True)
