"""Pull a structured answer out of free-text agent replies."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from . import units

_ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*(\{[^{}]*\})", re.IGNORECASE | re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{[^`]*?\})\s*```", re.IGNORECASE | re.DOTALL)
_NUMBER_RE = re.compile(
    r"(?<![\w.])([-+\u2212]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+\u2212]?\d+)?|[-+\u2212]?\.\d+)"
    r"(?:\s*(?:x|\u00d7|\*)\s*10\^?\s*\(?([-+\u2212]?\d+)\)?)?"
)
_REFUSAL_RE = re.compile(
    r"invalid smiles|not a valid|cannot be parsed|could not (?:be )?pars|unable to (?:parse|compute|find|process)|"
    r"does not exist|no such (?:entry|structure|id)|not found|physically (?:impossible|meaningless)|"
    r"chemically (?:impossible|invalid)|violates? valen|hypervalent|cannot answer|can't answer|"
    r"nuclei (?:overlap|coincide)|zero (?:bond )?(?:length|distance)",
    re.IGNORECASE,
)

_UNIT_TOKENS: list[str] = sorted(
    {k for k in units._UNITS if k} | set(units._CASE_SENSITIVE), key=len, reverse=True
)


@dataclass
class Extracted:
    value: Any = None
    unit: str | None = None
    refused: bool = False
    reason: str = ""
    method: str = "none"

    @property
    def found(self) -> bool:
        return self.value is not None or self.refused


def _to_float(s: str) -> float:
    return float(s.replace(",", "").replace("\u2212", "-"))


def _coerce(value: Any) -> Any:
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in {"null", "none", "n/a", "na", "invalid", ""}:
            return None
        if re.fullmatch(r"[-+\u2212]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?", v):
            try:
                return _to_float(v)
            except ValueError:
                return v
    return value


def _parse_obj(blob: str) -> dict | None:
    for candidate in (blob, blob.replace("'", '"')):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _from_obj(obj: dict, method: str) -> Extracted:
    raw_value = obj.get("value", obj.get("answer"))
    value = _coerce(raw_value)
    unit = obj.get("unit") or obj.get("units")
    reason = str(obj.get("reason") or "")
    return Extracted(value=value, unit=unit, refused=value is None, reason=reason, method=method)


def _unit_after(text: str, pos: int) -> str | None:
    tail = text[pos : pos + 24].lstrip()
    for tok in _UNIT_TOKENS:
        if tok in units._CASE_SENSITIVE:
            hit = tail.startswith(tok)
        else:
            hit = tail[: len(tok)].lower() == tok
        if hit:
            nxt = tail[len(tok) : len(tok) + 1]
            if not nxt or not (nxt.isalnum() or nxt == "_"):
                return tok
    return None


def scan_numbers(text: str) -> list[tuple[float, str | None]]:
    out: list[tuple[float, str | None]] = []
    for m in _NUMBER_RE.finditer(text):
        try:
            val = _to_float(m.group(1))
        except ValueError:
            continue
        if m.group(2):
            val *= 10 ** int(m.group(2).replace("\u2212", "-"))
        out.append((val, _unit_after(text, m.end())))
    return out


def extract(text: str, expected_dimension: str | None = None) -> Extracted:
    """Best-effort structured answer extraction.

    Order: explicit ``ANSWER: {...}`` line -> fenced JSON with a value ->
    last number carrying a unit of the expected dimension -> refusal phrases.
    """
    if not text:
        return Extracted()
    for pattern, method in ((_ANSWER_RE, "answer_line"), (_FENCED_JSON_RE, "json_block")):
        matches = pattern.findall(text)
        for blob in reversed(matches):
            obj = _parse_obj(blob)
            if obj is not None and ("value" in obj or "answer" in obj):
                return _from_obj(obj, method)

    if _REFUSAL_RE.search(text):
        return Extracted(refused=True, reason=_REFUSAL_RE.search(text).group(0), method="refusal_phrase")

    numbers = scan_numbers(text)
    if expected_dimension:
        for val, unit in reversed(numbers):
            if unit is not None and units.is_known(unit) and units.dimension(unit) == expected_dimension:
                return Extracted(value=val, unit=unit, method="number_with_unit")
        if expected_dimension == "dimensionless" and numbers:
            val, unit = numbers[-1]
            return Extracted(value=val, unit=unit, method="last_number")
    elif numbers:
        val, unit = numbers[-1]
        return Extracted(value=val, unit=unit, method="last_number")
    return Extracted()
