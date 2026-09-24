"""Question / answer I/O, including BenchmarkQED (AutoQ / AutoE) interop.

BenchmarkQED formats (from the microsoft/benchmark-qed repo):
  questions/assertions: JSON list of {question_id, question_text, assertions: [{statement, ...}], ...}
  answers:              JSON list of {question_id, question | question_text, answer, metrics?}
  AutoE pairwise:       one directory per condition containing <question_set>_answers.json
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

from .model import QUESTION_CLASSES, ArmAnswer, Context, Question


def _records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("questions", "data", "records", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return data


def _files(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted([*p.glob("*.json"), *p.glob("*.jsonl")]))
        else:
            out.append(p)
    return out


def _class_from_name(name: str) -> str:
    low = name.lower()
    return next((c for c in sorted(QUESTION_CLASSES, key=len, reverse=True) if c in low), "unknown")


def _maybe_literal(v: Any) -> Any:
    # BenchmarkQED sometimes serialises nested lists as Python-literal strings.
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                return v
    return v


def _assertions(raw: Any) -> list[str]:
    raw = _maybe_literal(raw)
    if not raw:
        return []
    out = []
    for a in raw if isinstance(raw, list) else [raw]:
        if isinstance(a, dict):
            s = a.get("statement") or a.get("assertion") or a.get("text")
            if s:
                out.append(str(s))
        elif a:
            out.append(str(a))
    return out


def load_questions(paths: Iterable[str | Path]) -> list[Question]:
    qs: dict[str, Question] = {}
    for f in _files(paths):
        default_cls = _class_from_name(f.stem)
        for r in _records(f):
            qid = str(r.get("question_id") or r.get("id") or r.get("qid") or "")
            text = r.get("question_text") or r.get("question") or r.get("text") or ""
            if not qid or not text:
                continue
            q = qs.get(qid) or Question(id=qid, text=str(text))
            explicit = r.get("qclass") or r.get("question_type") or r.get("class")
            if explicit:
                q.qclass = str(explicit)
            elif q.qclass == "unknown":
                q.qclass = default_cls
            q.assertions = q.assertions or _assertions(r.get("assertions"))
            q.reference = q.reference or r.get("reference") or r.get("reference_answer")
            q.required_docs = q.required_docs or [str(d) for d in (_maybe_literal(r.get("required_docs")) or [])]
            qs[qid] = q
    return list(qs.values())


def load_answers(path: str | Path, arm: str) -> list[ArmAnswer]:
    """Load one arm's answers (BenchmarkQED answer JSON, or JSONL with optional contexts)."""
    out: list[ArmAnswer] = []
    for f in _files([path]):
        for r in _records(f):
            qid = str(r.get("question_id") or r.get("id") or "")
            if not qid:
                continue
            ctxs = []
            for c in _maybe_literal(r.get("contexts") or r.get("retrieved") or []) or []:
                if isinstance(c, dict):
                    ctxs.append(Context(doc_id=str(c.get("doc_id") or c.get("doi") or c.get("id") or ""),
                                        text=str(c.get("text") or c.get("content") or ""), score=c.get("score")))
                else:
                    ctxs.append(Context(doc_id=str(c), text=""))
            out.append(ArmAnswer(arm=arm, question_id=qid, text=str(r.get("answer") or r.get("response") or ""),
                                 contexts=ctxs, latency_ms=r.get("latency_ms"), error=r.get("error"),
                                 meta={"source": str(f)}))
    return out


def save_answers_jsonl(answers: Iterable[ArmAnswer], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for a in answers:
            fh.write(json.dumps(a.to_dict()) + "\n")


def export_autoe(questions: list[Question], answers: dict[str, list[ArmAnswer]], out_dir: str | Path,
                 base: str, criteria_trials: int = 4) -> Path:
    """Write AutoE's expected layout + a pairwise settings.yaml skeleton.

    out_dir/input/<arm>/<qclass>_answers.json  and  out_dir/settings.yaml
    """
    out = Path(out_dir)
    qmap = {q.id: q for q in questions}
    classes = sorted({q.qclass for q in questions})
    for arm, ans in answers.items():
        by_cls: dict[str, list[dict[str, Any]]] = {}
        for a in ans:
            q = qmap.get(a.question_id)
            if q is None:
                continue
            by_cls.setdefault(q.qclass, []).append({"question_id": q.id, "question_text": q.text, "answer": a.text})
        d = out / "input" / arm
        d.mkdir(parents=True, exist_ok=True)
        for cls, rows in by_cls.items():
            (d / f"{cls}_answers.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    others = [a for a in answers if a != base]
    lines = ["# Generated by sciprobe arena. Fill in llm_config, then run:",
             "#   benchmark-qed autoe pairwise-scores settings.yaml output",
             "base:", f"  name: {base}", f"  answer_base_path: input/{base}", "", "others:"]
    for o in others:
        lines += [f"  - name: {o}", f"    answer_base_path: input/{o}"]
    lines += ["", "question_sets:"] + [f"  - {c}" for c in classes]
    lines += ["", f"trials: {criteria_trials}", "", "llm_config:", "  auth_type: api_key", "  model: gpt-4.1",
              "  api_key: ${OPENAI_API_KEY}", "  llm_provider: openai.chat", "  concurrent_requests: 20", ""]
    (out / "settings.yaml").write_text("\n".join(lines), encoding="utf-8")
    return out
