import json

import pytest

from conformance import FIXTURES, run_fixture
from skillforge.memory.client import LocalMemory


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_reference_implementation_conforms(path):
    run_fixture(LocalMemory(), json.loads(path.read_text()))


def test_idle_skill_retirement_and_stats():
    m = LocalMemory()
    m.put_skill("f", "def f(x):\n    return x\n", "Identity.", ["x"], now=0)
    assert m.reap(now=100, skill_idle_ttl=50)["skills_retired"] == 1
    assert m.stats()["skills"] == {"retired": 1}
