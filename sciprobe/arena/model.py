"""Arena data model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# BenchmarkQED AutoQ classes, plus two sciprobe additions:
#   connection - multi-hop questions whose answer requires linking >= 2 documents
#   control    - facts that exist only in the private corpus (memorisation control)
QUESTION_CLASSES = (
    "data_local", "data_global", "activity_local", "activity_global", "data_linked", "connection", "control",
)


@dataclass
class Question:
    id: str
    text: str
    qclass: str = "unknown"
    assertions: list[str] = field(default_factory=list)
    reference: str | None = None
    required_docs: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    doc_id: str
    text: str
    score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmAnswer:
    arm: str
    question_id: str
    text: str
    contexts: list[Context] = field(default_factory=list)
    latency_ms: float | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
