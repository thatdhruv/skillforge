"""Reproduce the main experiment: python scripts/run_experiment.py --seeds 10 --n 240 --out results/main"""
import argparse

from skillforge.eval.analyze import summarize
from skillforge.eval.run import ABLATIONS, MAIN, run_many

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--llm", default="sim", choices=["sim", "anthropic"])
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-ablations", action="store_true")
    a = ap.parse_args()
    conds = MAIN + ([] if a.no_ablations else ABLATIONS)
    run_many(conds, list(range(a.seeds)), a.n, a.llm, a.out, a.workers)
    summarize(a.out)
    print("summary written to", a.out)
