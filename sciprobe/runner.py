"""Select, plan, and execute oracle tasks and invariance probes."""
from __future__ import annotations

import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from . import __version__
from .invariance import InvarianceProbe, InvarianceResult, Variant, evaluate, variants
from .model import AgentReply, OracleTask, Verdict
from .scoring import score_task


@dataclass
class Job:
    kind: str  # "oracle" | "invariance" | "open"
    item_id: str
    agent: str  # logical agent name used in packs
    target: str  # deployed agent name actually asked
    prompt: str
    variant: Variant | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.item_id, "agent": self.target, "logical_agent": self.agent,
                "variant": self.variant.label if self.variant else None, "prompt": self.prompt}


@dataclass
class RunResult:
    verdicts: list[Verdict]
    invariance: list[InvarianceResult]
    meta: dict[str, Any] = field(default_factory=dict)
    open: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "verdicts": [v.to_dict() for v in self.verdicts],
                "invariance": [r.to_dict() for r in self.invariance], "open": self.open}


def _targets(item_agents: list[str], agents: list[str] | None) -> list[str]:
    if not agents or "*" in agents:
        return list(item_agents)
    wanted = {a.lower() for a in agents}
    return [a for a in item_agents if a.lower() in wanted]


def select(
    tasks: list[OracleTask],
    probes: list[InvarianceProbe],
    tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    ids: list[str] | None = None,
) -> tuple[list[OracleTask], list[InvarianceProbe]]:
    def keep(item_id: str, item_tags: list[str]) -> bool:
        if ids and not any(item_id == i or item_id.startswith(i.rstrip("*")) for i in ids):
            return False
        if tags and not set(tags) & set(item_tags):
            return False
        if exclude_tags and set(exclude_tags) & set(item_tags):
            return False
        return True

    return [t for t in tasks if keep(t.id, t.tags)], [p for p in probes if keep(p.id, p.tags)]


def plan(
    tasks: list[OracleTask],
    probes: list[InvarianceProbe],
    agents: list[str] | None = None,
    agent_map: dict[str, str] | None = None,
    seed: int = 0,
    open_items: list | None = None,
) -> list[Job]:
    amap = agent_map or {}
    jobs: list[Job] = []
    for t in tasks:
        for a in _targets(t.agents, agents):
            jobs.append(Job("oracle", t.id, a, amap.get(a, a), t.prompt))
    for p in probes:
        vs = variants(p, seed)
        for a in _targets(p.agents, agents):
            for v in vs:
                jobs.append(Job("invariance", p.id, a, amap.get(a, a), v.prompt, v))
    for o in open_items or []:
        for a in _targets([o.agent], agents):
            jobs.append(Job("open", o.id, a, amap.get(a, a), o.full_prompt))
    return jobs


def run(
    tasks: list[OracleTask],
    probes: list[InvarianceProbe],
    adapter,
    agents: list[str] | None = None,
    agent_map: dict[str, str] | None = None,
    workers: int = 4,
    seed: int = 0,
    progress: Callable[[int, int, Job, AgentReply], None] | None = None,
    open_items: list | None = None,
    open_offline: bool = False,
) -> RunResult:
    from .sourcing.bank import evaluate_open
    from .verify.provenance import provenance

    jobs = plan(tasks, probes, agents, agent_map, seed, open_items)
    started = time.time()
    replies: list[AgentReply | None] = [None] * len(jobs)
    done = 0
    lock = threading.Lock()

    def work(i: int) -> None:
        nonlocal done
        replies[i] = adapter.ask(jobs[i].target, jobs[i].prompt)
        with lock:
            done += 1
            n = done
        if progress:
            progress(n, len(jobs), jobs[i], replies[i])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, range(len(jobs))))

    task_by_id = {t.id: t for t in tasks}
    probe_by_id = {p.id: p for p in probes}
    open_by_id = {o.id: o for o in open_items or []}
    verdicts: list[Verdict] = []
    opened: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[tuple[Variant, AgentReply]]] = {}
    for job, rep in zip(jobs, replies):
        assert rep is not None
        if job.kind == "oracle":
            v = score_task(task_by_id[job.item_id], rep, job.agent)
            v.provenance = provenance(rep)
            verdicts.append(v)
        elif job.kind == "open":
            opened.append(evaluate_open(open_by_id[job.item_id], rep, offline=open_offline))
        else:
            grouped.setdefault((job.item_id, job.agent), []).append((job.variant, rep))
    inv = [evaluate(probe_by_id[pid], agent, reps) for (pid, agent), reps in grouped.items()]

    meta = {
        "sciprobe_version": __version__,
        "adapter": getattr(adapter, "name", type(adapter).__name__),
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 2),
        "seed": seed,
        "n_jobs": len(jobs),
        "python": platform.python_version(),
        "agent_map": agent_map or {},
    }
    return RunResult(verdicts, inv, meta, opened)
