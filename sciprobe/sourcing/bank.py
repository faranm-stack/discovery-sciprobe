"""Question bank: everything we can ask each agent, in priority order.

1. Listed prompts  - starter-kit samplePrompts, README usage tables, PR/custom
                     question files. Usually open-ended with no answer key, so
                     they are graded by fact-checking, rubric coverage and
                     provenance rather than a single number.
2. Built-in packs  - curated oracle tasks, invariance probes, negative controls.
3. Live oracles    - fresh questions with answer keys fetched from PubChem,
                     RCSB, UniProt, ChEMBL, ClinicalTrials.gov at build time.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..arena.judge import AssertionChecker, LexicalAssertionChecker
from ..model import AgentReply
from ..verify.factcheck import fact_check
from ..verify.provenance import provenance
from .harvest import AgentProfile, ListedPrompt

OPEN_INSTRUCTION = ("Answer as you normally would. Cite database identifiers (e.g. PubChem CID, ChEMBL ID, PDB ID, "
                    "UniProt accession, NCT number, PMID or DOI) for any facts you retrieve, and give SMILES in backticks.")


@dataclass
class OpenItem:
    id: str
    agent: str
    prompt: str
    source: str
    source_path: str = ""
    expected: str = ""
    input_files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def full_prompt(self) -> str:
        files = f"\n\nInput files: {', '.join(self.input_files)}" if self.input_files else ""
        return f"{self.prompt}{files}\n\n{OPEN_INSTRUCTION}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rubric_elements(expected: str) -> list[str]:
    parts = re.split(r",\s*(?:and\s+)?|;\s*|\s+and\s+(?=[a-z])", (expected or "").strip().rstrip("."))
    return [p.strip() for p in parts if len(p.strip()) > 3]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def open_items_from_profiles(profiles: dict[str, AgentProfile], include_file_prompts: bool = False) -> list[OpenItem]:
    items: list[OpenItem] = []
    for name, prof in profiles.items():
        for lp in prof.listed:
            tags = [lp.source] + (["needs-input-files"] if lp.input_files else [])
            if lp.input_files and not include_file_prompts:
                continue
            items.append(OpenItem(f"open.{lp.source}.{name}.{_slug(lp.prompt)}", name, lp.prompt, lp.source,
                                  lp.source_path, lp.expected, lp.input_files, tags))
    return items


def load_custom_questions(path: str | Path) -> list[OpenItem]:
    """JSONL/JSON of {agent, prompt, expected?, id?} - e.g. a PR's question set."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    rows = [json.loads(l) for l in text.splitlines() if l.strip()] if p.suffix == ".jsonl" else json.loads(text)
    return [OpenItem(r.get("id") or f"open.custom.{r['agent']}.{_slug(r['prompt'])}", r["agent"], r["prompt"], "custom",
                     str(p), r.get("expected", ""), r.get("input_files", []), ["custom"]) for r in rows]


def evaluate_open(item: OpenItem, reply: AgentReply, checker: AssertionChecker | None = None,
                  offline: bool = False) -> dict[str, Any]:
    base = {"id": item.id, "agent": item.agent, "source": item.source, "prompt": item.prompt,
            "expected": item.expected, "raw_text": reply.text or "", "latency_ms": reply.latency_ms}
    if reply.error:
        return {**base, "status": "ERROR", "score": 0.0, "diagnosis": reply.error, "provenance": "n/a"}
    if not (reply.text or "").strip():
        return {**base, "status": "NO_REPLY", "score": 0.0, "diagnosis": "empty reply", "provenance": "n/a"}

    fc = fact_check(reply.text, item.prompt) if not offline else None
    elems = rubric_elements(item.expected)
    chk = checker or LexicalAssertionChecker(threshold=0.5)
    covered = [e for e in elems if chk.check(item.prompt, reply.text, e)]
    coverage = (len(covered) / len(elems)) if elems else None
    prov = provenance(reply)

    parts = [x for x in ((fc.score if fc else None), coverage) if x is not None]
    score = sum(parts) / len(parts) if parts else None
    bad = [c for c in (fc.checks if fc else []) if c.status in ("NOT_FOUND", "INVALID", "MISMATCH")]
    if bad:
        status = "FLAGGED"
    elif score is None:
        status = "UNVERIFIABLE"
    else:
        status = "OK" if score >= 0.5 else "WEAK"
    diag = "; ".join(f"{c.kind} {c.value}: {c.status}{' - ' + c.detail if c.detail else ''}" for c in bad[:6])
    if elems and coverage is not None and coverage < 1:
        missing = [e for e in elems if e not in covered]
        diag = "; ".join(x for x in (diag, f"rubric missing: {missing}") if x)
    return {**base, "status": status, "score": score, "diagnosis": diag, "provenance": prov,
            "factcheck": fc.to_dict() if fc else None, "rubric_coverage": coverage, "rubric": elems}
