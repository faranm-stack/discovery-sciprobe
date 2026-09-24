"""Core data model: oracle tasks, references, verdicts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

ANSWER_INSTRUCTION = (
    "End your reply with exactly one final line of the form "
    'ANSWER: {"value": <number or string>, "unit": "<unit>"}. '
    'If the input is invalid, physically impossible, or cannot be answered, use '
    '"value": null and add a "reason" field instead of guessing.'
)


class Kind(str, Enum):
    NUMERIC = "numeric"  # value +/- tolerance, unit-aware
    RANGE = "range"  # lo <= value <= hi, unit-aware
    EXACT = "exact"  # case/whitespace-insensitive string or integer match
    FORMULA = "formula"  # molecular formula, element-count comparison
    SMILES = "smiles"  # same molecule after RDKit canonicalisation (any valid SMILES spelling passes)
    CUSTOM = "custom"  # OracleTask.checker(reply_text) decides


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NO_ANSWER = "NO_ANSWER"  # nothing parseable in the reply
    REFUSED = "REFUSED"  # agent declined when an answer was expected
    CORRECT_REFUSAL = "CORRECT_REFUSAL"  # negative control handled correctly
    HALLUCINATED = "HALLUCINATED"  # negative control answered with a value
    ERROR = "ERROR"  # transport / adapter error

    @property
    def ok(self) -> bool:
        return self in (Status.PASS, Status.CORRECT_REFUSAL)


@dataclass
class Confusion:
    """A known wrong answer, so a failure can be explained, not just flagged."""

    value: Any
    label: str
    unit: str | None = None


@dataclass
class Reference:
    kind: Kind
    value: Any = None
    unit: str | None = None
    rel_tol: float = 0.0
    abs_tol: float = 0.0
    lo: float | None = None
    hi: float | None = None
    source: str = ""

    def describe(self) -> str:
        if self.kind == Kind.RANGE:
            return f"[{self.lo}, {self.hi}] {self.unit or ''}".strip()
        if self.kind == Kind.NUMERIC:
            tol = []
            if self.abs_tol:
                tol.append(f"±{self.abs_tol:g}")
            if self.rel_tol:
                tol.append(f"±{self.rel_tol * 100:g}%")
            return f"{self.value:g} {self.unit or ''} ({' or '.join(tol) or 'exact'})".strip()
        return str(self.value)


@dataclass
class OracleTask:
    id: str
    domain: str
    agents: list[str]
    question: str
    reference: Reference | None
    confusions: list[Confusion] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    expect_refusal: bool = False
    notes: str = ""
    # For Kind.CUSTOM: fn(answer_value, full_reply_text) -> (passed, score 0..1, diagnosis)
    checker: Any = field(default=None, repr=False, compare=False)

    @property
    def prompt(self) -> str:
        return f"{self.question.strip()}\n\n{ANSWER_INSTRUCTION}"


@dataclass
class AgentReply:
    text: str
    latency_ms: float | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    task_id: str
    agent: str
    domain: str
    status: Status
    score: float
    expected: str
    got: str
    diagnosis: str = ""
    latency_ms: float | None = None
    raw_text: str = ""
    tags: list[str] = field(default_factory=list)
    provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d
