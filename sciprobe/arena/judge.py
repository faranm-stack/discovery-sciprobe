"""Judges: assertion checking and counterbalanced pairwise comparison.

Pairwise follows BenchmarkQED AutoE: every pair is judged in BOTH orders; if the
two verdicts disagree the result is a tie (position bias cannot manufacture a
win). The fraction of order-consistent verdicts is reported as a judge-quality
signal, alongside the recommended A/A test (an arm against a rerun of itself).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .text import content_tokens

# Default criteria mirror BenchmarkQED's pairwise metrics (wording is ours).
CRITERIA: dict[str, str] = {
    "comprehensiveness": "How much detail does the answer provide to cover all aspects of the question?",
    "diversity": "How varied and rich is the answer in offering different perspectives and insights?",
    "empowerment": "How well does the answer help the reader understand the topic and make informed judgements?",
    "relevance": "How directly does the answer address the specific question asked?",
}

ChatFn = Callable[[str, str], str]  # (system, user) -> text


class AssertionChecker(Protocol):
    name: str

    def check(self, question: str, answer: str, assertion: str) -> bool: ...


class PairwiseJudge(Protocol):
    name: str

    def judge(self, question: str, first: str, second: str, criterion: str) -> str:
        """Return "1", "2" or "tie" for the answers in the order shown."""
        ...


@dataclass
class PairVerdict:
    winner: str  # "a" | "b" | "tie"
    consistent: bool


def counterbalanced(judge: PairwiseJudge, question: str, a: str, b: str, criterion: str) -> PairVerdict:
    v1 = judge.judge(question, a, b, criterion)
    v2 = judge.judge(question, b, a, criterion)
    w1 = {"1": "a", "2": "b"}.get(v1, "tie")
    w2 = {"1": "b", "2": "a"}.get(v2, "tie")
    if w1 == w2:
        return PairVerdict(w1, True)
    return PairVerdict("tie", False)


# ------------------------------------------------------------------ offline (deterministic)

class LexicalAssertionChecker:
    """Heuristic: an assertion passes if the answer contains most of its key
    terms (numbers and gene/protein-like symbols are mandatory). Deterministic
    and free - good for smoke tests; use an LLM checker for real studies."""

    name = "lexical"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def check(self, question: str, answer: str, assertion: str) -> bool:
        a_words = set(content_tokens(answer))
        key = [w for w in content_tokens(assertion) if w not in {"response", "answer", "should", "mention", "state",
                                                                   "report", "that", "include", "describe", "explain"}]
        if not key:
            return False
        must = [w for w in key if any(ch.isdigit() for ch in w)]
        if any(m not in a_words for m in must):
            return False
        return sum(w in a_words for w in key) / len(key) >= self.threshold


class AssertionCountJudge:
    """Pairwise judge that prefers the answer satisfying more of the question's
    assertions. Needs no LLM and has no position or length bias."""

    name = "assertion-count"

    def __init__(self, checker: AssertionChecker, assertions_for: Callable[[str], list[str]]):
        self.checker, self.assertions_for = checker, assertions_for

    def judge(self, question: str, first: str, second: str, criterion: str) -> str:
        asserts = self.assertions_for(question)
        s1 = sum(self.checker.check(question, first, x) for x in asserts)
        s2 = sum(self.checker.check(question, second, x) for x in asserts)
        return "1" if s1 > s2 else "2" if s2 > s1 else "tie"


# ------------------------------------------------------------------ LLM judges

_PAIR_SYSTEM = ("You are an impartial expert judge of scientific answers. Judge ONLY on the stated criterion. "
                "Ignore answer length and style unless the criterion requires it. Do not favour an answer because "
                "of its position.")
_PAIR_USER = """Criterion: {name} - {desc}

Question:
{question}

Answer 1:
{first}

Answer 2:
{second}

Reply with a one-sentence rationale, then a final line exactly 'WINNER: 1', 'WINNER: 2' or 'WINNER: TIE'."""

_ASSERT_SYSTEM = "You verify whether an answer satisfies a factual assertion. Be strict: paraphrase is fine, vagueness is not."
_ASSERT_USER = """Question: {question}

Answer:
{answer}

Assertion: {assertion}

Does the answer satisfy the assertion? Reply with a final line exactly 'VERDICT: YES' or 'VERDICT: NO'."""


class LLMPairwiseJudge:
    def __init__(self, chat: ChatFn, criteria: dict[str, str] | None = None, name: str = "llm-pairwise"):
        self.chat, self.criteria, self.name = chat, criteria or CRITERIA, name

    def judge(self, question: str, first: str, second: str, criterion: str) -> str:
        out = self.chat(_PAIR_SYSTEM, _PAIR_USER.format(name=criterion, desc=self.criteria.get(criterion, criterion),
                                                        question=question, first=first, second=second))
        m = re.findall(r"WINNER:\s*(1|2|TIE)", out or "", re.IGNORECASE)
        return m[-1].lower() if m else "tie"


class LLMAssertionChecker:
    def __init__(self, chat: ChatFn, name: str = "llm-assertion"):
        self.chat, self.name = chat, name

    def check(self, question: str, answer: str, assertion: str) -> bool:
        out = self.chat(_ASSERT_SYSTEM, _ASSERT_USER.format(question=question, answer=answer, assertion=assertion))
        m = re.findall(r"VERDICT:\s*(YES|NO)", out or "", re.IGNORECASE)
        return bool(m) and m[-1].upper() == "YES"
