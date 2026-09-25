"""Command-line interface: ``python -m sciprobe {list,plan,run}``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import packs, report, runner
from .adapters import (
    DiscoveryAdapter,
    FaultInjectingAdapter,
    RDKitReferenceAgent,
    RecordingAdapter,
    ReplayAdapter,
)


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _agent_map(pairs: list[str] | None) -> dict[str, str]:
    """``logical=deployed`` pairs; ``@path`` reads one pair per line (# comments allowed)."""
    out = {}
    expanded: list[str] = []
    for p in pairs or []:
        if p.startswith("@"):
            for line in Path(p[1:]).read_text(encoding="utf-8-sig").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    expanded.append(line)
        else:
            expanded.append(p)
    for p in expanded:
        k, _, v = p.partition("=")
        if not v:
            raise SystemExit(f"--agent-map expects logical=deployed, got {p!r}")
        out[k.strip()] = v.strip()
    return out


def _selection(args):
    names = _csv(args.packs) or list(packs.PACKS)
    if getattr(args, "live", 0) and "live" not in names:
        names.append("live")
    tasks, probes = packs.load(names, live_n=getattr(args, "live", 0) or 3, seed=args.seed)
    return runner.select(tasks, probes, _csv(args.tags), _csv(args.exclude_tags), _csv(args.ids))


def _open_items(args) -> list:
    from .sourcing.bank import load_custom_questions, open_items_from_profiles
    from .sourcing.harvest import harvest, open_catalog

    items = []
    if getattr(args, "catalog", None):
        profiles = harvest(open_catalog(args.catalog), _csv(args.agents))
        items += open_items_from_profiles(profiles, include_file_prompts=args.include_file_prompts)
    for q in getattr(args, "questions", None) or []:
        items += load_custom_questions(q)
    return items


def _add_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--packs", help=f"comma list from {packs.PACKS + packs.OPTIONAL_PACKS}")
    p.add_argument("--tags", help="only items with any of these tags (e.g. core)")
    p.add_argument("--exclude-tags", help="drop items with any of these tags (e.g. expensive)")
    p.add_argument("--ids", help="comma list of task/probe ids or id prefixes ending in *")
    p.add_argument("--agents", help="comma list of logical agent names (default: all in selected items)")
    p.add_argument("--agent-map", action="append", metavar="LOGICAL=DEPLOYED", help="rename agents for your deployment")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", type=int, default=0, metavar="N",
                   help="add N fresh live-oracle draws per authority (PubChem, RCSB, UniProt, ChEMBL, ClinicalTrials.gov)")
    p.add_argument("--catalog", help="harvest listed prompts: 'github' (microsoft/discovery@main), "
                                     "'github:owner/repo@ref', or a local checkout / PR branch path")
    p.add_argument("--questions", action="append", help="custom question file (JSON/JSONL of {agent, prompt, expected?})")
    p.add_argument("--include-file-prompts", action="store_true",
                   help="also run listed prompts that reference input files (the agent must have them)")


def cmd_list(args) -> int:
    tasks, probes = _selection(args)
    print(f"{len(tasks)} oracle tasks, {len(probes)} invariance probes\n")
    for t in tasks:
        exp = "REFUSE" if t.expect_refusal else (t.reference.describe() if t.reference else "?")
        print(f"  [oracle] {t.id:<40} {','.join(t.agents):<34} {exp}  {' '.join('#' + g for g in t.tags)}")
    for p in probes:
        print(f"  [invar ] {p.id:<40} {','.join(p.agents):<34} {p.relation.value}  {' '.join('#' + g for g in p.tags)}")
    return 0


def cmd_plan(args) -> int:
    tasks, probes = _selection(args)
    jobs = runner.plan(tasks, probes, _csv(args.agents), _agent_map(args.agent_map), args.seed, _open_items(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for j in jobs:
            fh.write(json.dumps(j.to_dict()) + "\n")
    print(f"wrote {len(jobs)} prompts to {out}")
    return 0


def _build_adapter(args):
    if args.adapter == "reference":
        adapter = RDKitReferenceAgent()
    elif args.adapter == "faulty":
        adapter = FaultInjectingAdapter(RDKitReferenceAgent(), {
            "exact_mass_for_mw": 0.5, "prefix_slip": 0.1, "input_order_noise": 0.6,
            "hallucinate_invalid": 1.0, "drop_answer": 0.05}, seed=args.seed)
    elif args.adapter == "replay":
        adapter = ReplayAdapter(args.replay or None)
        print(f"replay: loaded {len(adapter)} recorded replies", file=sys.stderr)
    elif args.adapter == "discovery":
        if not args.project or not (args.workspace_uri or args.workspace_arm_id):
            raise SystemExit("--adapter discovery needs --project and --workspace-uri or --workspace-arm-id")
        adapter = DiscoveryAdapter(args.project, args.workspace_uri, args.workspace_arm_id,
                                   investigation=args.investigation, poll_timeout_s=args.timeout)
    else:  # pragma: no cover
        raise SystemExit(f"unknown adapter {args.adapter}")
    if args.record:
        adapter = RecordingAdapter(adapter, args.record)
    return adapter


def cmd_run(args) -> int:
    tasks, probes = _selection(args)
    adapter = _build_adapter(args)

    def progress(n, total, job, rep):
        flag = "ERR" if rep.error else "ok "
        print(f"  [{n:>3}/{total}] {flag} {job.target:<20} {job.item_id} {job.variant.label if job.variant else ''}",
              file=sys.stderr)

    try:
        result = runner.run(tasks, probes, adapter, _csv(args.agents), _agent_map(args.agent_map),
                            workers=args.workers, seed=args.seed, progress=None if args.quiet else progress,
                            open_items=_open_items(args))
    finally:
        cleanup = getattr(adapter, "cleanup", None)
        if cleanup and args.cleanup != "none":
            summary = cleanup(args.cleanup)
            if summary:
                inv = getattr(getattr(adapter, "inner", adapter), "investigation", "?")
                print(f"cleanup ({summary['mode']}): deleted {summary['deleted_conversations']} conversation(s), "
                      f"investigation {inv!r} deleted={summary['investigation_deleted']}", file=sys.stderr)
                for f in summary["failed"]:
                    print(f"  cleanup failed: {f}", file=sys.stderr)
    paths = report.write_reports(result, args.out)
    print(report.to_markdown(result))
    print(f"reports: {', '.join(str(p) for p in paths.values())}", file=sys.stderr)
    if args.fail_under is not None:
        low = [c for c in report.scorecards(result) if c.sciscore is not None and c.sciscore < args.fail_under]
        if low:
            print(f"FAIL: {', '.join(f'{c.agent}={c.sciscore:.1f}' for c in low)} below {args.fail_under}", file=sys.stderr)
            return 1
    return 0


def cmd_bank(args) -> int:
    """Coverage map: what can we ask each agent, from which source, and where must we step in?"""
    from .sourcing.bank import open_items_from_profiles
    from .sourcing.harvest import harvest, open_catalog

    catalog = open_catalog(args.catalog or "github")
    profiles = harvest(catalog, _csv(args.agents))
    tasks, probes = _selection(args)
    listed_all = open_items_from_profiles(profiles, include_file_prompts=True)
    rows, bank = [], {"catalog": catalog.label, "agents": {}}
    for name, prof in sorted(profiles.items()):
        oracle = [t for t in tasks if name in t.agents and not t.expect_refusal and "live" not in t.tags]
        live = [t for t in tasks if name in t.agents and "live" in t.tags]
        ctrl = [t for t in tasks if name in t.agents and t.expect_refusal]
        inv = [p for p in probes if name in p.agents]
        listed = [o for o in listed_all if o.agent == name]
        kit = sum(o.source == "kit" for o in listed)
        verifiable = len(oracle) + len(live) + len(ctrl) + len(inv)
        gap = "STEP IN: no verifiable checks yet" if not verifiable else ("listed prompts only via fact-check" if not oracle else "")
        rows.append((name, ",".join(prof.tags) or "-", len(listed) - kit, kit, len(prof.example_inputs),
                     len(oracle), len(inv), len(ctrl), len(live), gap))
        bank["agents"][name] = {
            "description": prof.description, "tags": prof.tags, "example_inputs": prof.example_inputs,
            "listed_prompts": [o.to_dict() for o in listed], "oracle_tasks": [t.id for t in oracle],
            "live_tasks": [t.id for t in live], "controls": [t.id for t in ctrl], "invariance": [p.id for p in inv],
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    print(f"catalog {catalog.label}: {len(profiles)} agents\n")
    print("| Agent | Tags | README prompts | Kit prompts | Example inputs | Oracles | Invariance | Controls | Live | Gap |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


def cmd_factcheck(args) -> int:
    from .verify import fact_check

    texts: list[tuple[str, str, str | None]] = []
    if args.text:
        texts.append((args.text, Path(args.text).read_text(encoding="utf-8"), args.prompt))
    if args.replay is not None:
        rp = ReplayAdapter([args.replay] if args.replay else None)
        for (agent, prompt), reps in rp._store.items():
            if args.agents and agent not in {a.lower() for a in _csv(args.agents)}:
                continue
            for r in reps:
                texts.append((f"{agent}: {prompt[:60]}", r.text, prompt))
    worst = 0
    for label, text, prompt in texts:
        rep = fact_check(text, prompt)
        if not rep.checks:
            continue
        worst = max(worst, rep.n_bad)
        print(f"\n## {label}\n{rep.counts}  score={rep.score if rep.score is None else round(rep.score, 2)}")
        for c in rep.checks:
            print(f"  {c.status:<10} {c.kind:<8} {c.value:<24} {(c.label or '')[:60]}  {c.detail}")
    return 1 if (args.strict and worst) else 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="sciprobe", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list oracle tasks and invariance probes")
    _add_filters(p)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("plan", help="write the exact prompts to JSONL (for external execution + replay)")
    _add_filters(p)
    p.add_argument("--out", default="sciprobe-out/plan.jsonl")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="ask agents, score, and write reports")
    _add_filters(p)
    p.add_argument("--adapter", choices=["reference", "faulty", "replay", "discovery"], default="reference")
    p.add_argument("--replay", action="append", help="JSONL file or dir (default ~/.md-toolbox/conversations)")
    p.add_argument("--project", help="Discovery project (discovery adapter)")
    p.add_argument("--workspace-uri", help="Discovery workspace data-plane URI")
    p.add_argument("--workspace-arm-id", help="Discovery workspace ARM id (resolves the data-plane URI)")
    p.add_argument("--investigation", help="investigation name (default: unique sciprobe-<utc>-<rand> per run)")
    p.add_argument("--cleanup", choices=["all", "conversations", "none"], default="all",
                   help="after the run, delete what sciprobe created in Discovery (default: all; "
                        "pre-existing investigations are never deleted). Use --record to keep transcripts locally.")
    p.add_argument("--timeout", type=float, default=900.0, help="per-prompt poll timeout (s)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--record", help="append every exchange to this JSONL for later replay")
    p.add_argument("--out", default="sciprobe-out")
    p.add_argument("--fail-under", type=float, help="exit 1 if any agent's SciScore is below this")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("bank", help="coverage map of listed prompts, oracles and gaps per agent")
    _add_filters(p)
    p.add_argument("--out", default="sciprobe-out/bank.json")
    p.set_defaults(fn=cmd_bank)

    p = sub.add_parser("factcheck", help="verify identifiers, SMILES, formulas and MWs in agent replies")
    p.add_argument("--text", help="a file containing one reply")
    p.add_argument("--prompt", help="the question (enables on-topic checks)")
    p.add_argument("--replay", nargs="?", const="", help="fact-check recorded replies (default ~/.md-toolbox/conversations)")
    p.add_argument("--agents")
    p.add_argument("--strict", action="store_true", help="exit 1 if any claim fails")
    p.set_defaults(fn=cmd_factcheck)

    from .arena.cli import register as register_arena

    register_arena(sub)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
