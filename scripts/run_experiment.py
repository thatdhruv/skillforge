"""Reproduce the main experiment: python scripts/run_experiment.py --seeds 10 --n 240 --out results/main"""
import argparse

from skillforge.eval.analyze import summarize
from skillforge.eval.run import ABLATIONS, MAIN, run_many

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--llm", default="sim", choices=["sim", "anthropic", "gemini"])
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--every", type=int, default=40, help="checkpoint spacing")
    ap.add_argument("--small-probes", type=int, default=0, help="0 full, 1 reduced, 2 tiny probe sets")
    ap.add_argument("--conds", default="", help="comma list; default = main (+ablations)")
    a = ap.parse_args()
    conds = a.conds.split(",") if a.conds else MAIN + ([] if a.no_ablations else ABLATIONS)
    run_many(conds, list(range(a.seeds)), a.n, a.llm, a.out, a.workers, a.every, a.small_probes)
    summarize(a.out)
    print("summary written to", a.out)
