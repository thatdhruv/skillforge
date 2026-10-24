from skillforge.envs.stream import build_stream, extract_facts


def test_stream_shape_and_determinism():
    a, b = build_stream(1), build_stream(1)
    assert [t.prompt for t in a.items] == [t.prompt for t in b.items]
    kinds = {t.kind for t in a.items}
    assert kinds == {"code", "fact", "obs"}
    assert a.checkpoints[0] == 0 and a.checkpoints[-1] == 240
    pr = a.probes[120]
    assert len(pr["transfer"]) == 16 and len(pr["retention"]) == 16 and pr["fact"]
    # retention instances are identical across checkpoints
    assert [t.prompt for t in a.probes[40]["retention"]] == [t.prompt for t in a.probes[200]["retention"]]


def test_fact_extraction():
    f = extract_facts("This week's on-call engineer for auth is bob.", 5.0)
    assert f[0]["predicate"] == "oncall" and f[0]["object"] == "bob" and f[0]["ttl_seconds"]
    assert extract_facts("Archived note: auth used to run in region eu-west-1.", 1.0)[0]["valid_from"] == 1.0
