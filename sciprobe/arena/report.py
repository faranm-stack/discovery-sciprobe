"""Arena report (Markdown + JSON)."""
from __future__ import annotations

import json
from pathlib import Path

from .run import ArenaResult


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def _num(x, fmt: str = ".1f") -> str:
    return "—" if x is None else format(x, fmt)


def _p(x) -> str:
    return "—" if x is None else ("<0.001" if x < 0.001 else f"{x:.3f}")


def to_markdown(res: ArenaResult) -> str:
    m = res.meta
    classes = sorted({q.qclass for q in res.questions})
    out = [
        "# SciProbe retrieval arena",
        "",
        f"{m['n_questions']} questions · arms: {', '.join(m['arms'])} · baseline: {m.get('baseline') or '—'} · "
        f"judge: {m.get('judge') or '—'} · assertion checker: {m.get('assertion_checker') or '—'}",
        "",
        "## Arms",
        "",
        "| Arm | Controlled | Answered | Errors | Assertion pass | "
        + " | ".join(classes) + " | Cited/retrieved | Citation precision | Utilization | Support | Out-of-corpus | Median latency |",
        "|---|---|---:|---:|---:|" + "---:|" * len(classes) + "---|---:|---:|---:|---:|---:|",
    ]
    for s in res.arm_summary:
        by = s["assertion_pass_by_class"]
        out.append(
            f"| {s['arm']} | {'yes' if s['controlled'] else 'no'} | {s['answered']} | {s['errors']} | "
            f"**{_pct(s['assertion_pass'])}** | " + " | ".join(_pct(by.get(c)) for c in classes) + " | "
            f"{_num(s['mean_cited'])}/{_num(s['mean_retrieved'])} | {_pct(s['citation_precision'])} | "
            f"{_pct(s['citation_utilization'])} | {_pct(s['citation_support'])} | {s['out_of_corpus_citations']} | "
            f"{_num(s['median_latency_ms'], '.0f')} ms |")

    if res.lift:
        out += ["", "## Lift over the no-retrieval baseline (assertion pass rate, paired by question)", "",
                "| Arm | n | Mean diff | 95% CI | Better / worse / same | p (sign) |", "|---|---:|---:|---|---|---:|"]
        for r in res.lift:
            out.append(f"| {r['arm']} | {r['n']} | {r['mean_diff']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | "
                       f"{r['better']} / {r['worse']} / {r['same']} | {_p(r['p_sign'])} |")

    main = [r for r in res.pair_summary if r["qclass"] == "all"]
    if main:
        out += ["", "## Pairwise (counterbalanced; question = unit of analysis)", "",
                "| A vs B | Criterion | A wins | B wins | Ties | A win rate | 95% CI | p (sign) | p (Holm) | Judge consistency |",
                "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
        for r in main:
            out.append(f"| {r['a']} vs {r['b']} | {r['criterion']} | {r['wins_a']} | {r['wins_b']} | {r['ties']} | "
                       f"{_pct(r['win_rate_a'])} | [{_pct(r['ci95'][0])}, {_pct(r['ci95'][1])}] | {_p(r['p_sign'])} | "
                       f"{_p(r.get('p_holm'))} | {_pct(r['judge_consistency'])} |")
        by_cls = [r for r in res.pair_summary if r["qclass"] != "all"]
        if by_cls:
            out += ["", "### By question class", "", "| A vs B | Criterion | Class | A–B–tie | A win rate | p (sign) |",
                    "|---|---|---|---|---:|---:|"]
            for r in by_cls:
                out.append(f"| {r['a']} vs {r['b']} | {r['criterion']} | {r['qclass']} | "
                           f"{r['wins_a']}–{r['wins_b']}–{r['ties']} | {_pct(r['win_rate_a'])} | {_p(r['p_sign'])} |")

    if res.warnings:
        out += ["", "## Validity notes", ""] + [f"- {w}" for w in res.warnings]
    return "\n".join(out) + "\n"


def write(res: ArenaResult, out_dir: str | Path) -> dict[str, Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = {"json": d / "arena.json", "md": d / "arena.md"}
    paths["json"].write_text(json.dumps(res.to_dict(), indent=2, default=str), encoding="utf-8")
    paths["md"].write_text(to_markdown(res), encoding="utf-8")
    return paths
