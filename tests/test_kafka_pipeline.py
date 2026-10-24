"""End-to-end through Redpanda + the Go memory service. Needs `docker compose up -d postgres memory redpanda`.

    KAFKA_BROKERS=localhost:19092 MEMORY_URL=http://localhost:8080 pytest tests/test_kafka_pipeline.py
"""
import os
import time
import uuid
from random import Random

import pytest

from skillforge.envs.families import FAMILIES, make_instance
from skillforge.llm.textutil import stem_of

BROKERS, URL = os.environ.get("KAFKA_BROKERS"), os.environ.get("MEMORY_URL")
pytestmark = pytest.mark.skipif(not (BROKERS and URL), reason="KAFKA_BROKERS / MEMORY_URL not set")


def test_trajectories_become_a_gated_skill_via_redpanda():
    from skillforge.bus import KafkaBus, wait_for_kafka
    from skillforge.memory.client import HttpMemory
    from skillforge.skills.distiller import Distiller
    from skillforge.skills.gate import Gate

    wait_for_kafka(BROKERS)
    mem, bus = HttpMemory(URL), KafkaBus(BROKERS)
    mem.reset()
    fam, rng = FAMILIES["base_conv"], Random(5)
    run = uuid.uuid4().hex[:6]
    code = fam.ref_src.replace("FN", "solve") + "\nprint(solve(**INPUT))\n"

    # four verified successes of the same family; the last one is the trajectory that gets distilled
    for i in range(4):
        inputs, prompt, expected = make_instance(fam, rng)
        tid = f"{run}-{i}"
        mem.add_episode({"id": tid, "task_id": tid, "kind": "code", "success": True, "prompt": prompt,
                         "answer": expected, "meta": {"inputs": inputs}}, stem_of(prompt))
        if i == 3:
            bus.publish("trajectories.finished", {"task_id": tid, "prompt": prompt, "kind": "code", "success": True,
                                                  "final_code": code, "inputs": inputs, "answer": expected, "ts": 1.0})
    bus.flush()

    d, g = Distiller(bus, group=f"d-{run}"), Gate(mem, bus, group=f"g-{run}")
    end = time.time() + 40
    proposed = 0
    while time.time() < end and not proposed:
        proposed += d.step(timeout=1.0)
    bus.flush()
    assert proposed >= 1
    end = time.time() + 40
    while time.time() < end and not any(r["type"] == "promoted" for r in g.log):
        g.step(timeout=1.0)
    assert any(r["type"] == "promoted" for r in g.log), g.log
    active = mem.list_skills("active")
    assert active and active[0]["gate"]["passed"] >= 3
