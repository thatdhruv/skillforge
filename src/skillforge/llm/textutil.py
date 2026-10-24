from __future__ import annotations

import math
import re

STOP = set("""a an the of to in on for and or is are be it its this that these those with as at by from into your you
must define print answer given below return what which how many number each every any all if then than can has have
code variable input predefined dict write""".split())


def words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        out.append(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


def cosine_overlap(a: str, b: str) -> float:
    A, B = set(words(a)), set(words(b))
    if not A or not B:
        return 0.0
    return len(A & B) / math.sqrt(len(A) * len(B))


def stem_of(prompt: str) -> str:
    """The natural-language part of a task prompt (no code contract, no data)."""
    head = prompt.split("\nYour code must")[0]
    return head.split("\nINPUT =")[0].strip()
