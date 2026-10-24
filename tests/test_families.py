import json
from random import Random

import pytest

from skillforge.envs.families import FAMILIES, TRAIN_FAMILIES, make_instance
from skillforge.envs.sandbox import run_python, screen


def script(fam, src):
    body = src.replace("FN", "solve")
    return body + "\nprint(solve(**INPUT))\n"


@pytest.mark.parametrize("name", TRAIN_FAMILIES)
def test_reference_matches_truth_and_bug_is_partial(name):
    fam = FAMILIES[name]
    rng = Random(7)
    n, bug_fail = 12, 0
    for _ in range(n):
        inputs, prompt, expected = make_instance(fam, rng)
        assert "INPUT = " in prompt
        r = run_python(script(fam, fam.ref_src), inputs)
        assert r.ok, r.stderr
        assert r.answer == expected
        for b in fam.bug_srcs:
            rb = run_python(script(fam, b), inputs)
            if not rb.ok or rb.answer != expected:
                bug_fail += 1
    if fam.bug_srcs:
        rate = bug_fail / (n * len(fam.bug_srcs))
        assert 0.05 < rate < 0.9, f"{name} bug trigger rate {rate}"


def test_screen_blocks_escape_attempts():
    assert screen("import os") is not None
    assert screen("open('x')") is not None
    assert screen("x = ().__class__") is not None
    assert screen("import re\nprint(1)") is None


def test_timeout():
    r = run_python("while True: pass", timeout=0.5)
    assert r.timed_out and not r.ok


def test_skills_module_and_usage_tracking():
    skills = {"double": 'def double(x):\n    "Double x."\n    return 2 * x\n'}
    r = run_python("from skills import double\nprint(double(INPUT['v']))", {"v": 21}, skills)
    assert r.ok and r.answer == "42" and r.skills_used == ["double"]


def test_composite_truth():
    fam = FAMILIES["roman_add"]
    inputs, _, expected = make_instance(fam, Random(1))
    src = (FAMILIES["roman"].ref_src.replace("FN", "int_to_roman") + FAMILIES["roman_to_int"].ref_src.replace("FN", "roman_to_int")
           + fam.glue_src + "\nprint(solve(**INPUT))")
    assert run_python(src, inputs).answer == expected
