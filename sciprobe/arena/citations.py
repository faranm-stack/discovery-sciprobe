"""Citation forensics: are cited sources real, retrieved, used, and supportive?

Metrics per answer:
  precision    cited ∩ retrieved / cited     (cites things it actually saw)
  utilization  cited ∩ retrieved / retrieved (the "58.5% vs 20.9%" metric)
  out_of_corpus  cited ids that are not in the corpus at all (fabrication signal)
  support      fraction of cited sentences whose content words are found in the
               cited source text (lexical grounding; a cheap faithfulness proxy)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .model import ArmAnswer
from .text import content_tokens, sentences

_DOI = re.compile(r"10\.\d{4,9}/[^\s\]\)\[,;\"'<>]+")
_BRACKET = re.compile(r"\[([^\[\]\n]{1,160})\]")


def norm_id(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", s)
    return s.rstrip(".)")


def cited_ids(text: str, context_ids: list[str] | None = None) -> list[str]:
    """Citation ids in order of first appearance. Numeric [n] maps to the n-th context."""
    found: list[str] = []
    ctx = context_ids or []
    for m in _BRACKET.finditer(text or ""):
        for tok in re.split(r"[,;]", m.group(1)):
            tok = tok.strip()
            if not tok or tok.upper() == "NO_SOURCES":
                continue
            if tok.isdigit() and ctx and 1 <= int(tok) <= len(ctx):
                tok = ctx[int(tok) - 1]
            found.append(norm_id(tok))
    for m in _DOI.finditer(text or ""):
        found.append(norm_id(m.group(0)))
    seen: set[str] = set()
    return [x for x in found if not (x in seen or seen.add(x))]


@dataclass
class CitationReport:
    arm: str
    question_id: str
    n_cited: int
    n_retrieved: int
    precision: float | None
    utilization: float | None
    out_of_corpus: list[str] = field(default_factory=list)
    unretrieved: list[str] = field(default_factory=list)
    support: float | None = None
    uncited_sentence_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze(answer: ArmAnswer, corpus_ids: set[str] | None = None, support_threshold: float = 0.5) -> CitationReport:
    ctx_ids = [norm_id(c.doc_id) for c in answer.contexts]
    retrieved = set(ctx_ids)
    cited = cited_ids(answer.text, ctx_ids)
    cset = set(cited)
    hit = cset & retrieved
    corpus = {norm_id(c) for c in corpus_ids} if corpus_ids else None

    ctx_text = {norm_id(c.doc_id): c.text for c in answer.contexts}
    sents = sentences(answer.text)
    cited_sents = supported = uncited = 0
    for s in sents:
        ids = cited_ids(s, ctx_ids)
        words = set(content_tokens(_BRACKET.sub(" ", s)))
        if not words:
            continue
        if not ids:
            uncited += 1
            continue
        src = " ".join(ctx_text.get(i, "") for i in ids)
        if not src:
            continue
        cited_sents += 1
        if len(words & set(content_tokens(src))) / len(words) >= support_threshold:
            supported += 1
    total = cited_sents + uncited
    return CitationReport(
        arm=answer.arm,
        question_id=answer.question_id,
        n_cited=len(cset),
        n_retrieved=len(retrieved),
        precision=(len(hit) / len(cset)) if cset and retrieved else None,
        utilization=(len(hit) / len(retrieved)) if retrieved else None,
        out_of_corpus=sorted(cset - corpus) if corpus is not None else [],
        unretrieved=sorted(cset - retrieved) if retrieved else [],
        support=(supported / cited_sents) if cited_sents else None,
        uncited_sentence_fraction=(uncited / total) if total else None,
    )
