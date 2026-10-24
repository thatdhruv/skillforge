"""Experience pipeline glue: episode -> bus -> distiller -> gate -> library, plus usage tracking and fact writing."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..agent.trajectory import Trajectory
from ..bus import TRAJECTORIES, Bus, InMemoryBus
from ..envs.stream import Task, extract_facts
from ..llm.textutil import stem_of
from ..memory.client import Memory
from .distiller import Distiller
from .gate import Gate, GateConfig


@dataclass
class LearnConfig:
    use_memory: bool = False
    use_skills: bool = False
    gate: GateConfig = field(default_factory=GateConfig)


class Learner:
    """Turns finished episodes into memory. ``step_bus`` runs the consumers inline; in a deployment the distiller and
    the gate are separate services reading the same topics."""

    def __init__(self, memory: Memory, cfg: LearnConfig, bus: Bus | None = None, namer=None, inline: bool = True):
        self.mem, self.cfg = memory, cfg
        self.bus = bus or InMemoryBus()
        self.inline = inline
        self.distiller = Distiller(self.bus, namer) if cfg.use_skills and inline else None
        self.gate = Gate(memory, self.bus, cfg.gate) if cfg.use_skills and inline else None

    # -------------------------------------------------------------- observation (statements, no scoring)
    def observe_statement(self, task: Task, now: float) -> list[dict]:
        out = []
        if self.cfg.use_memory:
            for f in extract_facts(task.prompt, task.meta.get("doc_ts", now)):
                out.append(self.mem.put_fact(f["subject"], f["predicate"], f["object"], f["text"],
                                             valid_from=f["valid_from"], ttl_seconds=f["ttl_seconds"],
                                             source_episode=task.id, now=now))
        return out

    # -------------------------------------------------------------- finished task
    def observe_episode(self, task: Task, traj: Trajectory, now: float) -> None:
        self.mem.add_episode({
            "id": task.id, "task_id": task.id, "kind": task.kind, "success": bool(traj.success), "prompt": task.prompt,
            "answer": traj.answer, "tokens_in": traj.tokens_in, "tokens_out": traj.tokens_out, "cost": traj.cost,
            "trajectory": traj.to_dict(), "created_at": now,
            "meta": {"inputs": task.inputs, "_family": task.family}},      # _family is for offline analysis only
            stem_of(task.prompt))
        if not self.cfg.use_skills:
            return
        active = {s["name"]: s["version"] for s in self.mem.list_skills("active")}
        offered = {s["name"]: s["version"] for s in traj.offered_skills}
        for name in set(traj.skills_used):
            ver = offered.get(name) or active.get(name)
            if ver:
                self.mem.record_usage(name, ver, bool(traj.success), now)
        self.bus.publish(TRAJECTORIES, {
            "task_id": task.id, "prompt": task.prompt, "kind": task.kind, "success": bool(traj.success),
            "final_code": traj.final_code, "inputs": task.inputs,
            "answer": traj.answer, "ts": now}, key=task.id)
        if self.inline:
            self.step_bus()

    def step_bus(self) -> None:
        if self.distiller:
            self.distiller.step()
            self.gate.step()
