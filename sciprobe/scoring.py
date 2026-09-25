"""Grade one agent reply against one oracle task."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from . import units
from .extract import Extracted, extract
from .model import AgentReply, Kind, OracleTask, Reference, Status, Verdict

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
# Refusals caused by the platform (tool runner down, model files missing), not by the agent's science.
_ENV_FAILURE = re.compile(
    r"(backend|execution|environment|tool|runtime|compute)[^.]{0,40}(error|fail|unavailable)|"
    r"(fail|error)[^.]{0,40}(backend|execution|environment|tool)|no trained .{0,30}model .{0,40}available|"
    r"model artifacts?|internal error",
    re.IGNORECASE,
)


def parse_formula(s: str) -> Counter:
    s = re.sub(r"[\s_$^{}]", "", str(s))
    s = s.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    counts: Counter = Counter()
    pos = 0
    for m in _FORMULA_TOKEN.finditer(s):
        if m.start() != pos:
            raise ValueError(f"unparseable formula {s!r}")
        counts[m.group(1)] += int(m.group(2) or 1)
        pos = m.end()
    if pos != len(s) or not counts:
        raise ValueError(f"unparseable formula {s!r}")
    return counts


def _norm_exact(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v).is_integer():
        return str(int(v))
    s = str(v).strip().lower()
    s = re.sub(r"\s+", "", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _tolerance(ref: Reference) -> float:
    return max(ref.abs_tol, ref.rel_tol * abs(float(ref.value)))


def _to_ref_unit(ex: Extracted, ref: Reference) -> tuple[float | None, str]:
    """Return (value in reference unit, note). value None means incompatible."""
    try:
        val = float(ex.value)
    except (TypeError, ValueError):
        return None, f"non-numeric answer {ex.value!r}"
    if not ref.unit:  # dimensionless reference: the bare number is what matters
        return val, ""
    if not ex.unit:
        return val, f"no unit given; assumed {ref.unit}"
    if not units.is_known(ex.unit):
        return None, f"unrecognised unit {ex.unit!r}"
    try:
        return units.convert(val, ex.unit, ref.unit), ""
    except units.UnitError:
        return None, f"wrong dimension: got {ex.unit!r} ({units.dimension(ex.unit)}), expected {units.dimension(ref.unit)}"


def _match_confusion(task: OracleTask, got: Any, got_unit: str | None) -> str | None:
    ref = task.reference
    for c in task.confusions:
        if ref and ref.kind in (Kind.NUMERIC, Kind.RANGE):
            try:
                cval = float(c.value)
                if c.unit and ref.unit:
                    cval = units.convert(cval, c.unit, ref.unit)
                tol = max(_tolerance(ref) if ref.kind == Kind.NUMERIC else 0.0, 1e-9 * abs(cval), ref.abs_tol)
                if abs(float(got) - cval) <= max(tol, 1e-6):
                    return c.label
            except (TypeError, ValueError, units.UnitError):
                continue
        elif _norm_exact(c.value) == _norm_exact(got):
            return c.label
    return None


def _numeric_fail_diagnosis(task: OracleTask, got: float, ex: Extracted) -> str:
    ref = task.reference
    assert ref is not None
    target = float(ref.value) if ref.kind == Kind.NUMERIC else (float(ref.lo) + float(ref.hi)) / 2
    hit = _match_confusion(task, got, ref.unit)
    if hit:
        return hit
    try:
        raw = float(ex.value)
        if ref.unit and ex.unit and abs(raw - target) <= max(_tolerance(ref) if ref.kind == Kind.NUMERIC else 0, 1e-9):
            return f"right number, wrong unit label ({ex.unit!r} instead of {ref.unit!r})"
    except (TypeError, ValueError):
        pass
    guess = units.diagnose_ratio(got, target)
    if guess:
        return guess
    if target:
        return f"off by {100 * (got - target) / abs(target):+.3g}%"
    return f"off by {got - target:+.3g}"


def score_task(task: OracleTask, reply: AgentReply, agent: str) -> Verdict:
    base = dict(
        task_id=task.id,
        agent=agent,
        domain=task.domain,
        latency_ms=reply.latency_ms,
        raw_text=reply.text or "",
        tags=list(task.tags),
    )
    ref = task.reference
    expected = "refusal (invalid input)" if task.expect_refusal else (ref.describe() if ref else "?")

    if reply.error:
        return Verdict(status=Status.ERROR, score=0.0, expected=expected, got="", diagnosis=reply.error, **base)

    dim = None
    if ref and ref.kind in (Kind.NUMERIC, Kind.RANGE) and ref.unit is not None and units.is_known(ref.unit):
        dim = units.dimension(ref.unit)
    ex = extract(reply.text, dim)
    got_str = "null" if ex.value is None else f"{ex.value} {ex.unit or ''}".strip()

    if task.expect_refusal:
        if ex.refused:
            return Verdict(status=Status.CORRECT_REFUSAL, score=1.0, expected=expected, got="refused",
                           diagnosis=ex.reason, **base)
        if ex.value is not None:
            return Verdict(status=Status.HALLUCINATED, score=0.0, expected=expected, got=got_str,
                           diagnosis="produced a value for an invalid / impossible input", **base)
        return Verdict(status=Status.NO_ANSWER, score=0.0, expected=expected, got="", diagnosis="no parseable reply", **base)

    if ex.refused:
        if _ENV_FAILURE.search(ex.reason or ""):
            return Verdict(status=Status.ERROR, score=0.0, expected=expected, got="refused",
                           diagnosis=f"environment: {ex.reason}", **base)
        return Verdict(status=Status.REFUSED, score=0.0, expected=expected, got="refused",
                       diagnosis=ex.reason or "declined", **base)
    if ex.value is None:
        return Verdict(status=Status.NO_ANSWER, score=0.0, expected=expected, got="",
                       diagnosis="no parseable answer in reply", **base)
    assert ref is not None

    if ref.kind == Kind.EXACT:
        accepted = ref.value if isinstance(ref.value, (list, tuple)) else [ref.value]
        ok = _norm_exact(ex.value) in {_norm_exact(a) for a in accepted}
        diag = "" if ok else (_match_confusion(task, ex.value, ex.unit) or "mismatch")
        return Verdict(status=Status.PASS if ok else Status.FAIL, score=1.0 if ok else 0.0,
                       expected=expected, got=got_str, diagnosis=diag, **base)

    if ref.kind == Kind.FORMULA:
        try:
            ok = parse_formula(ex.value) == parse_formula(ref.value)
            diag = "" if ok else "element counts differ"
        except ValueError as e:
            ok, diag = False, str(e)
        return Verdict(status=Status.PASS if ok else Status.FAIL, score=1.0 if ok else 0.0,
                       expected=expected, got=got_str, diagnosis=diag, **base)

    if ref.kind == Kind.SMILES:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        got_m, ref_m = Chem.MolFromSmiles(str(ex.value).strip()), Chem.MolFromSmiles(str(ref.value))
        if got_m is None:
            ok, diag = False, "answer is not a valid SMILES"
        else:
            ok = Chem.MolToSmiles(got_m) == Chem.MolToSmiles(ref_m)
            diag = ""
            if not ok:
                same_graph = Chem.MolToSmiles(got_m, isomericSmiles=False) == Chem.MolToSmiles(ref_m, isomericSmiles=False)
                diag = "right connectivity, wrong/missing stereochemistry" if same_graph else "different molecule"
        return Verdict(status=Status.PASS if ok else Status.FAIL, score=1.0 if ok else 0.0,
                       expected=expected, got=got_str, diagnosis=diag, **base)

    if ref.kind == Kind.CUSTOM:
        try:
            ok, sc, diag = task.checker(ex.value, reply.text)
        except Exception as e:  # noqa: BLE001
            ok, sc, diag = False, 0.0, f"checker error: {e}"
        return Verdict(status=Status.PASS if ok else Status.FAIL, score=float(sc), expected=expected,
                       got=got_str[:200], diagnosis=diag, **base)

    got, note = _to_ref_unit(ex, ref)
    if got is None:
        return Verdict(status=Status.FAIL, score=0.0, expected=expected, got=got_str, diagnosis=note, **base)

    if ref.kind == Kind.NUMERIC:
        tol = _tolerance(ref)
        err = abs(got - float(ref.value))
        if err <= tol:
            return Verdict(status=Status.PASS, score=1.0, expected=expected, got=got_str, diagnosis=note, **base)
        score = max(0.0, 1.0 - (err - tol) / (4.0 * tol)) if tol > 0 else 0.0
    else:
        lo, hi = float(ref.lo), float(ref.hi)
        if lo <= got <= hi:
            return Verdict(status=Status.PASS, score=1.0, expected=expected, got=got_str, diagnosis=note, **base)
        width = max(hi - lo, 1e-12)
        dist = lo - got if got < lo else got - hi
        score = max(0.0, 1.0 - dist / width)

    diag = _numeric_fail_diagnosis(task, got, ex)
    if note:
        diag = f"{diag}; {note}"
    return Verdict(status=Status.FAIL, score=round(score, 4), expected=expected, got=got_str, diagnosis=diag, **base)
