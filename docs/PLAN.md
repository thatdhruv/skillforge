# SkillForge — architecture decisions and phase plan

## Goal

An agent that turns its own experience into **verified, reusable skills** and **typed memory**,
plus an evaluation that answers one question honestly: *does the improvement survive a controlled comparison?*

## Architecture decisions

| # | Decision | Why | Rejected |
|---|----------|-----|----------|
| 1 | **LangGraph** agent (`plan → act → observe → finish`) | Explicit graph, easy to trace trajectories | Hand-rolled loop |
| 2 | Tools served over **MCP** (FastMCP). In-memory transport for experiments, stdio/HTTP in deployment | Matches the brief; one tool surface for every LLM backend | Direct function calls |
| 3 | **Pluggable `LLM` interface**: `AnthropicLLM` (real) and `SimLLM` (deterministic stand-in) | No API key is needed to reproduce the pipeline; the real model is a one-env-var switch. Every report states which backend produced it | Mocking at the HTTP layer |
| 4 | **Redpanda** (Kafka API) event bus, with an in-memory bus behind the same interface | Redpanda is a single binary, so it is light on Kubernetes and laptops. In-memory keeps tests and experiments deterministic | Kafka + ZooKeeper |
| 5 | **Memory service in Go** + **Postgres/pgvector** | Brief asks for Go or Java; Go gives a small static binary and a simple `pgx` stack | Java/Spring |
| 6 | Hybrid retrieval = pgvector cosine **+** Postgres FTS, fused with **reciprocal rank fusion** | Shows both retrieval skills and is robust when one signal is weak | Vector-only |
| 7 | **Python reference implementation of the memory rules** + shared JSON conformance fixtures run against both Python and Go | Experiments need thousands of fast, deterministic runs; production needs Postgres. Fixtures stop the two from drifting | Duplicating logic untested |
| 8 | **Skill gate** = sandboxed replay tests against verified past cases, then promote | The only road into the library; deprecation is driven by observed failures | Trusting LLM-written code |
| 9 | Evaluation is a **2×2 factorial** (memory × skills) over one fixed task stream, multiple seeds | Isolates each component's contribution and their interaction | A single before/after run |
| 10 | Docker Compose for local, **Helm** for Kubernetes, **Terraform** for a minimal EKS | Brief | — |

## Memory write/update rules (the contract)

* **Episodic** — append-only trajectories. Retained; never mutated.
* **Semantic** facts are keyed by `(subject, predicate)`:
  * same object ⇒ *reinforce* (confidence↑, `last_confirmed` refreshed);
  * different object, newer `valid_from` ⇒ *supersede* (old fact kept as history, never returned by search);
  * different object, older `valid_from` ⇒ stored as *history* only (a late-arriving stale fact must not win);
  * tie on time ⇒ higher confidence wins, otherwise the incumbent stays and the write is *rejected*.
  * every fact may carry a TTL; expired facts are filtered at read time and reaped in the background.
* **Procedural** skills are versioned per name. `active → deprecated` when `uses ≥ 5` and success rate `< 50 %`.
  A new version supersedes the previous active one.

## Phases

| Phase | Deliverable | Exit criterion |
|-------|-------------|----------------|
| 0 | Repo, plan, toolchain | Plan committed |
| 1 | Task environments + sandbox + LLM interface (+ `SimLLM`) | Each task family has a verified reference solution; bugs are reproducible |
| 2 | Memory rules (Python) + Go service + Postgres/pgvector | Conformance fixtures pass on Python **and** Go/Postgres |
| 3 | MCP tool server + LangGraph agent | Agent solves tasks end to end through MCP |
| 4 | Event bus, distiller, skill gate, library lifecycle | A trajectory becomes a gated, versioned skill; failures deprecate it |
| 5 | Evaluation harness: 4 conditions, transfer, retention, LongMemEval | One command reproduces every number |
| 6 | Dashboard and figures | Learning curves, failure cases, library growth/retention |
| 7 | Dockerfiles, Compose, Helm, Terraform (EKS) | `helm lint`/`template`, `terraform validate` pass; Compose stack runs |
| 8 | README with diagrams and screenshots; GitHub repo | Fresh clone works from README |
