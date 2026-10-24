from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Trajectory:
    task_id: str
    prompt: str
    kind: str = "code"
    steps: list[dict] = field(default_factory=list)       # {"text", "tool_calls":[{name,args}], "results":[str]}
    answer: str = ""
    success: bool | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    llm_calls: int = 0
    offered_skills: list[dict] = field(default_factory=list)   # {name, version}
    skills_used: list[str] = field(default_factory=list)
    final_code: str = ""                                       # last code that ran without error
    run_errors: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
