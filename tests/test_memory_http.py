"""Runs the same conformance fixtures against the Go service (set MEMORY_URL, e.g. http://localhost:8080)."""
import json
import os

import pytest

from conformance import FIXTURES, run_fixture
from skillforge.memory.client import HttpMemory

URL = os.environ.get("MEMORY_URL")
pytestmark = pytest.mark.skipif(not URL, reason="MEMORY_URL not set")


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_go_service_conforms(path):
    mem = HttpMemory(URL)
    mem.reset()
    run_fixture(mem, json.loads(path.read_text()))
