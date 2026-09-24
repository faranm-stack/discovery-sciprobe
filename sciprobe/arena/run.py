"""Run an arena: collect answers from every arm, check assertions, analyse
citations, judge pairwise (counterbalanced), and compute paired statistics."""
from __future__ import annotations

import itertools
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import citations as cit
from . import stats
from .judge import CRITERIA, AssertionChecker, AssertionCountJudge, PairwiseJudge, counterbalanced
from .model import ArmAnswer, Question


@dataclass
class ArenaResult:
    questions: list[Question]
    answers: dict[str, dict[str, ArmAnswer]]
    assertion_results: dict[str, dict[str, list[bool]]]
    citation_reports: dict[str, dict[str, cit.CitationReport]]
    pair_verdicts: list[dict[str, Any]]
    arm_summary: list[dict[str, Any]] = field(default_factory=list)
    pair_summary: list[dict[str, Any]] = field(default_factory=list)
    lift: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "warnings": self.warnings,
            "arm_summary": self.arm_summary,
            "pair_summary": self.pair_summary,
            "lift_vs_baseline": self.lift,
            "questions": [q.__dict__ for q in self.questions],
            "answers": {a: {q: x.to_dict() for q, x in d.items()} for a, d in self.answers.items()},
            "assertion_results": self.assertion_results,
            "citations": {a: {q: r.to_dict() for q, r in d.items()} for a, d in self.citation_reports.items()},
            "pair_verdicts": self.pair_verdicts,
        }


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def run_arena(
    questions: list[Question],
    arms: list,
    assertion_checker: AssertionChecker | None = None,
    pairwise_judge: PairwiseJudge | None = None,
    criteria: list[str] | None = None,
    baseline: str | None = None,
    corpus_ids: set[str] | None = None,
    workers: int = 4,
    seed: int = 0,
) -> ArenaResult:
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arm names must be unique: {names}")
    if baseline and baseline not in names:
        raise ValueError(f"baseline {baseline!r} is not one of {names}")

    # 1. answers (each call is independent; arms own their isolation)
    jobs = [(arm, q) for arm in arms for q in questions]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        outs = list(pool.map(lambda j: j[0].answer(j[1]), jobs))
    answers: dict[str, dict[str, ArmAnswer]] = {n: {} for n in names}
    for (arm, q), a in zip(jobs, outs):
        answers[arm.name][q.id] = a

    # 2. assertions
    assertion_results: dict[str, dict[str, list[bool]]] = {n: {} for n in names}
    if assertion_checker:
        checks = [(n, q, x) for n in names for q in questions if q.assertions for x in q.assertions]

        def _check(t):
            n, q, x = t
            a = answers[n][q.id]
            return (n, q.id, False if a.error or not a.text else assertion_checker.check(q.text, a.text, x))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for n, qid, ok in pool.map(_check, checks):
                assertion_results[n].setdefault(qid, []).append(bool(ok))

    # 3. citations
    citation_reports = {n: {q.id: cit.analyze(answers[n][q.id], corpus_ids) for q in questions} for n in names}

    # 4. pairwise
    judge = pairwise_judge
    if judge is None and assertion_checker is not None:
        text_to_assertions = {q.text: q.assertions for q in questions}
        judge = AssertionCountJudge(assertion_checker, lambda t: text_to_assertions.get(t, []))
        crit = criteria or ["assertions"]
    else:
        crit = criteria or list(CRITERIA)
    verdicts: list[dict[str, Any]] = []
    if judge is not None:
        tasks = [(a, b, c, q) for a, b in itertools.combinations(names, 2) for c in crit for q in questions]

        def _pair(t):
            a, b, c, q = t
            ta, tb = answers[a][q.id], answers[b][q.id]
            if ta.error or tb.error:
                return None
            v = counterbalanced(judge, q.text, ta.text, tb.text, c)
            return {"a": a, "b": b, "criterion": c, "question_id": q.id, "qclass": q.qclass,
                    "winner": v.winner, "consistent": v.consistent}

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            verdicts = [v for v in pool.map(_pair, tasks) if v]

    res = ArenaResult(questions, answers, assertion_results, citation_reports, verdicts,
                      meta={"arms": names, "baseline": baseline, "criteria": crit if judge else [],
                            "judge": getattr(judge, "name", None), "assertion_checker": getattr(assertion_checker, "name", None),
                            "n_questions": len(questions), "seed": seed})
    _summarise(res, arms, baseline, seed)
    return res


def _assert_frac(res: ArenaResult, arm: str, qid: str) -> float | None:
    r = res.assertion_results.get(arm, {}).get(qid)
    return (sum(r) / len(r)) if r else None


def _summarise(res: ArenaResult, arms: list, baseline: str | None, seed: int) -> None:
    qclass = {q.id: q.qclass for q in res.questions}
    classes = sorted({q.qclass for q in res.questions})

    for arm in arms:
        n = arm.name
        ans = res.answers[n]
        reps = res.citation_reports[n]
        per_class = {}
        for c in classes:
            vals = [_assert_frac(res, n, q) for q in ans if qclass[q] == c]
            per_class[c] = _mean([v for v in vals if v is not None])
        lat = [a.latency_ms for a in ans.values() if a.latency_ms is not None]
        res.arm_summary.append({
            "arm": n,
            "controlled": bool(getattr(arm, "controlled", False)),
            "answered": sum(1 for a in ans.values() if a.text and not a.error),
            "errors": sum(1 for a in ans.values() if a.error),
            "assertion_pass": _mean([_assert_frac(res, n, q) for q in ans]),
            "assertion_pass_by_class": per_class,
            "citation_precision": _mean([r.precision for r in reps.values()]),
            "citation_utilization": _mean([r.utilization for r in reps.values()]),
            "citation_support": _mean([r.support for r in reps.values()]),
            "mean_cited": _mean([r.n_cited for r in reps.values()]),
            "mean_retrieved": _mean([r.n_retrieved for r in reps.values()]),
            "out_of_corpus_citations": sum(len(r.out_of_corpus) for r in reps.values()),
            "median_latency_ms": statistics.median(lat) if lat else None,
        })

    # pairwise: question is the unit; criteria reported separately
    groups: dict[tuple, list[dict]] = {}
    for v in res.pair_verdicts:
        for cls in ("all", v["qclass"]):
            groups.setdefault((v["a"], v["b"], v["criterion"], cls), []).append(v)
    rows = []
    for (a, b, c, cls), vs in sorted(groups.items()):
        wa = sum(v["winner"] == "a" for v in vs)
        wb = sum(v["winner"] == "b" for v in vs)
        ties = len(vs) - wa - wb
        n = len(vs)
        lo, hi = stats.wilson_ci(wa + 0.5 * ties, n)
        rows.append({"a": a, "b": b, "criterion": c, "qclass": cls, "n": n, "wins_a": wa, "wins_b": wb, "ties": ties,
                     "win_rate_a": (wa + 0.5 * ties) / n if n else None, "ci95": [round(lo, 3), round(hi, 3)],
                     "p_sign": stats.sign_test(wa, wb),
                     "judge_consistency": sum(v["consistent"] for v in vs) / n if n else None})
    main = [r for r in rows if r["qclass"] == "all"]
    for r, adj in zip(main, stats.holm([r["p_sign"] for r in main])):
        r["p_holm"] = adj
    res.pair_summary = rows

    # paired lift in assertion pass rate vs the baseline arm
    if baseline:
        for arm in arms:
            if arm.name == baseline:
                continue
            diffs = []
            for q in res.questions:
                x, y = _assert_frac(res, arm.name, q.id), _assert_frac(res, baseline, q.id)
                if x is not None and y is not None:
                    diffs.append(x - y)
            if not diffs:
                continue
            ups, downs = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
            res.lift.append({"arm": arm.name, "baseline": baseline, "n": len(diffs),
                             "mean_diff": statistics.fmean(diffs),
                             "ci95": [round(x, 3) for x in stats.bootstrap_mean_ci(diffs, seed=seed)],
                             "better": ups, "worse": downs, "same": len(diffs) - ups - downs,
                             "p_sign": stats.sign_test(ups, downs)})

    _warnings(res, arms, baseline)


def _warnings(res: ArenaResult, arms: list, baseline: str | None) -> None:
    w = res.warnings
    controlled = [a for a in arms if getattr(a, "controlled", False)]
    uncontrolled = [a.name for a in arms if not getattr(a, "controlled", False)]
    if controlled and uncontrolled:
        w.append(f"Arms {uncontrolled} are end-to-end systems (own model/prompt/agent). Comparisons with them "
                 "mix retrieval with generation differences; only controlled arms isolate retrieval.")
    gens = {a.generator.name for a in controlled if hasattr(a, "generator")}
    if len(gens) > 1:
        w.append(f"Controlled arms use different generators {sorted(gens)} - retrieval is no longer the only variable.")
    n = len(res.questions)
    need = stats.min_wins_for_significance(n)
    w.append(f"With {n} questions, an arm needs >= {need} question-level wins (no ties) for p < 0.05; "
             f"~{stats.questions_needed(0.6)} questions are needed to detect a 60% true win rate with 80% power.")
    cons = [r["judge_consistency"] for r in res.pair_summary if r["qclass"] == "all" and r["judge_consistency"] is not None]
    if cons and min(cons) < 0.8:
        w.append(f"Judge order-consistency as low as {min(cons):.0%}: strong position bias; run an A/A test and "
                 "consider a stronger judge model.")
    if baseline:
        b = next((s for s in res.arm_summary if s["arm"] == baseline), None)
        ctrl = (b or {}).get("assertion_pass_by_class", {}).get("control")
        if ctrl is not None and ctrl > 0.3:
            w.append(f"The no-retrieval baseline passes {ctrl:.0%} of 'control' (private-corpus) assertions - "
                     "those facts may be public or memorised; the control is leaking.")
    for s in res.arm_summary:
        if s["out_of_corpus_citations"]:
            w.append(f"{s['arm']} cited {s['out_of_corpus_citations']} id(s) that are not in the corpus.")
