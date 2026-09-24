"""Scorecards: per-agent SciScore plus JSON / Markdown / HTML reports."""
from __future__ import annotations

import html
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .model import Status
from .runner import RunResult

WEIGHTS = {"accuracy": 0.45, "invariance": 0.25, "controls": 0.15, "factuality": 0.15}


@dataclass
class AgentCard:
    agent: str
    oracle_n: int = 0
    oracle_pass: int = 0
    oracle_score: float = 0.0
    inv_n: int = 0
    inv_pass: int = 0
    ctrl_n: int = 0
    ctrl_pass: int = 0
    open_n: int = 0
    open_scored: int = 0
    open_score: float = 0.0
    open_flagged: int = 0
    errors: int = 0

    def component(self, name: str) -> float | None:
        if name == "accuracy":
            return self.oracle_score / self.oracle_n if self.oracle_n else None
        if name == "invariance":
            return self.inv_pass / self.inv_n if self.inv_n else None
        if name == "factuality":
            return self.open_score / self.open_scored if self.open_scored else None
        return self.ctrl_pass / self.ctrl_n if self.ctrl_n else None

    @property
    def sciscore(self) -> float | None:
        parts = [(WEIGHTS[k], v) for k in WEIGHTS if (v := self.component(k)) is not None]
        if not parts:
            return None
        return 100.0 * sum(w * v for w, v in parts) / sum(w for w, _ in parts)


def scorecards(result: RunResult) -> list[AgentCard]:
    cards: dict[str, AgentCard] = defaultdict(lambda: AgentCard(""))
    for v in result.verdicts:
        c = cards[v.agent]
        c.agent = v.agent
        if v.status == Status.ERROR:
            c.errors += 1
        if v.status in (Status.CORRECT_REFUSAL, Status.HALLUCINATED) or v.expected.startswith("refusal"):
            c.ctrl_n += 1
            c.ctrl_pass += int(v.status.ok)
        else:
            c.oracle_n += 1
            c.oracle_pass += int(v.status.ok)
            c.oracle_score += v.score
    for r in result.invariance:
        c = cards[r.agent]
        c.agent = r.agent
        c.inv_n += 1
        c.inv_pass += int(r.status == Status.PASS)
        if r.status == Status.ERROR:
            c.errors += 1
    for o in result.open:
        c = cards[o["agent"]]
        c.agent = o["agent"]
        c.open_n += 1
        c.open_flagged += int(o["status"] == "FLAGGED")
        if o["status"] == "ERROR":
            c.errors += 1
        if o.get("score") is not None:
            c.open_scored += 1
            c.open_score += o["score"]
    return sorted(cards.values(), key=lambda c: (-(c.sciscore or -1), c.agent))


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.0f}%" if pct else f"{x:.1f}"


def to_markdown(result: RunResult) -> str:
    cards = scorecards(result)
    m = result.meta
    lines = [
        "# SciProbe scorecard",
        "",
        f"adapter `{m.get('adapter')}` · {m.get('started_utc')} · {m.get('n_jobs')} prompts · "
        f"{m.get('duration_s')} s · seed {m.get('seed')} · sciprobe {m.get('sciprobe_version')}",
        "",
        "| Agent | SciScore | Accuracy (oracle) | Invariance | Negative controls | Factuality (open) | Errors |",
        "|---|---:|---|---|---|---|---:|",
    ]
    for c in cards:
        lines.append(
            f"| {c.agent} | **{_fmt(c.sciscore)}** | {c.oracle_pass}/{c.oracle_n} ({_fmt(c.component('accuracy'), True)}) | "
            f"{c.inv_pass}/{c.inv_n} | {c.ctrl_pass}/{c.ctrl_n} | "
            f"{_fmt(c.component('factuality'), True)} of {c.open_n} ({c.open_flagged} flagged) | {c.errors} |"
        )
    lines += ["", "SciScore = " + " + ".join(f"{w:.0%} {k}" for k, w in WEIGHTS.items())
              + " (renormalised over components that ran).", ""]

    bad = [v for v in result.verdicts if not v.status.ok]
    lines += ["## Oracle findings", ""]
    if not bad:
        lines.append("All oracle tasks passed.")
    else:
        lines += ["| Agent | Task | Status | Expected | Got | Diagnosis | Provenance |", "|---|---|---|---|---|---|---|"]
        for v in sorted(bad, key=lambda v: (v.agent, v.task_id)):
            lines.append(f"| {v.agent} | `{v.task_id}` | {v.status.value} | {v.expected} | {v.got or '—'} | "
                         f"{v.diagnosis} | {v.provenance or '—'} |")
    if result.open:
        lines += ["", "## Open-ended prompts (fact-checked against PubChem / ChEMBL / PDB / UniProt / NCT / PubMed / Crossref)", "",
                  "| Agent | Source | Prompt | Status | Score | Claims | Rubric | Provenance | Findings |",
                  "|---|---|---|---|---:|---|---:|---|---|"]
        for o in sorted(result.open, key=lambda o: (o["agent"], o["id"])):
            counts = (o.get("factcheck") or {}).get("counts") or {}
            claims = ", ".join(f"{k.lower()} {n}" for k, n in counts.items()) or "—"
            prompt = o["prompt"] if len(o["prompt"]) <= 70 else o["prompt"][:67] + "..."
            lines.append(f"| {o['agent']} | {o['source']} | {prompt} | {o['status']} | {_fmt(o.get('score'), True)} | "
                         f"{claims} | {_fmt(o.get('rubric_coverage'), True)} | {o.get('provenance', '—')} | "
                         f"{o.get('diagnosis') or '—'} |")
    lines += ["", "## Invariance probes", ""]
    if not result.invariance:
        lines.append("No invariance probes ran.")
    else:
        lines += ["| Agent | Probe | Relation | Status | Spread | Tol | Diagnosis |", "|---|---|---|---|---:|---:|---|"]
        for r in sorted(result.invariance, key=lambda r: (r.agent, r.probe_id)):
            spread = "—" if r.spread is None else f"{r.spread:.3g}"
            lines.append(f"| {r.agent} | `{r.probe_id}` | {r.relation} | {r.status.value} | {spread} | {r.tolerance:.3g} | {r.diagnosis} |")
    return "\n".join(lines) + "\n"


_CSS = """
body{font-family:Segoe UI,system-ui,sans-serif;margin:2rem;color:#1b1b1b;background:#fafafa}
h1{margin-bottom:.2rem} .meta{color:#666;margin-bottom:1.5rem}
table{border-collapse:collapse;margin:1rem 0;background:#fff;box-shadow:0 1px 3px #0001;font-size:.9rem}
th,td{padding:.45rem .7rem;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
th{background:#f0f2f5} .score{font-weight:700;font-size:1.05rem}
.PASS,.CORRECT_REFUSAL{color:#107c10;font-weight:600} .FAIL,.HALLUCINATED{color:#c50f1f;font-weight:600}
.NO_ANSWER,.REFUSED{color:#ca5010;font-weight:600} .ERROR{color:#8a8886;font-weight:600}
.bar{height:8px;background:#e1e1e1;border-radius:4px;width:120px;display:inline-block;vertical-align:middle;margin-left:.4rem}
.bar>span{display:block;height:100%;border-radius:4px}
details pre{white-space:pre-wrap;max-width:60rem;background:#f6f6f6;padding:.5rem}
"""


def _bar(score: float | None) -> str:
    if score is None:
        return ""
    color = "#107c10" if score >= 80 else "#ca5010" if score >= 50 else "#c50f1f"
    return f'<span class="bar"><span style="width:{score:.0f}%;background:{color}"></span></span>'


def to_html(result: RunResult) -> str:
    e = html.escape
    cards = scorecards(result)
    m = result.meta
    rows = "".join(
        f"<tr><td>{e(c.agent)}</td><td class=score>{_fmt(c.sciscore)}{_bar(c.sciscore)}</td>"
        f"<td>{c.oracle_pass}/{c.oracle_n}</td><td>{c.inv_pass}/{c.inv_n}</td><td>{c.ctrl_pass}/{c.ctrl_n}</td><td>{c.errors}</td></tr>"
        for c in cards
    )
    vrows = "".join(
        f"<tr><td>{e(v.agent)}</td><td><code>{e(v.task_id)}</code></td><td class={v.status.value}>{v.status.value}</td>"
        f"<td>{e(v.expected)}</td><td>{e(v.got or '—')}</td><td>{e(v.diagnosis)}"
        f"<details><summary>reply</summary><pre>{e(v.raw_text[:4000])}</pre></details></td></tr>"
        for v in sorted(result.verdicts, key=lambda v: (v.status.ok, v.agent, v.task_id))
    )
    irows = "".join(
        f"<tr><td>{e(r.agent)}</td><td><code>{e(r.probe_id)}</code></td><td>{e(r.relation)}</td>"
        f"<td class={r.status.value}>{r.status.value}</td><td>{'—' if r.spread is None else f'{r.spread:.3g}'}</td>"
        f"<td>{r.tolerance:.3g}</td><td>{e(r.diagnosis)}<details><summary>variants</summary><pre>"
        + e("\n".join(f"{x['label']}: {x['raw']}  {x['note']}\n    {x['input']}" for x in r.values))
        + "</pre></details></td></tr>"
        for r in sorted(result.invariance, key=lambda r: (r.status == Status.PASS, r.agent, r.probe_id))
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>SciProbe scorecard</title><style>{_CSS}</style></head>
<body><h1>SciProbe scorecard</h1>
<div class=meta>adapter <b>{e(str(m.get('adapter')))}</b> · {e(str(m.get('started_utc')))} · {m.get('n_jobs')} prompts ·
{m.get('duration_s')} s · seed {m.get('seed')}</div>
<table><tr><th>Agent</th><th>SciScore</th><th>Accuracy</th><th>Invariance</th><th>Neg. controls</th><th>Errors</th></tr>{rows}</table>
<h2>Oracle tasks</h2><table><tr><th>Agent</th><th>Task</th><th>Status</th><th>Expected</th><th>Got</th><th>Diagnosis</th></tr>{vrows}</table>
<h2>Invariance probes</h2><table><tr><th>Agent</th><th>Probe</th><th>Relation</th><th>Status</th><th>Spread</th><th>Tol</th><th>Diagnosis</th></tr>{irows}</table>
</body></html>"""


def write_reports(result: RunResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "results.json", "md": out / "scorecard.md", "html": out / "scorecard.html"}
    paths["json"].write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    paths["md"].write_text(to_markdown(result), encoding="utf-8")
    paths["html"].write_text(to_html(result), encoding="utf-8")
    return paths
