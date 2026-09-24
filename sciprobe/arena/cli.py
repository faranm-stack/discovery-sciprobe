"""``sciprobe arena {demo,score,export-autoe,power}``."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import demo as demo_mod
from . import report, stats
from .arms import AzureOpenAIGenerator, ReplayArm
from .judge import LexicalAssertionChecker, LLMAssertionChecker, LLMPairwiseJudge
from .questions import export_autoe, load_answers, load_questions
from .run import run_arena


def _arm_specs(specs: list[str]) -> dict[str, str]:
    out = {}
    for s in specs or []:
        name, _, path = s.partition("=")
        if not path:
            raise SystemExit(f"--answers expects ARM=PATH, got {s!r}")
        out[name.strip()] = path.strip()
    return out


def _aoai(args):
    if not (args.aoai_endpoint and args.aoai_deployment):
        raise SystemExit("LLM judging needs --aoai-endpoint and --aoai-deployment")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    tp = None
    if not key:
        from ..adapters.discovery import TokenProvider

        tp = TokenProvider("https://cognitiveservices.azure.com/.default", "SCIPROBE_AOAI_TOKEN")
    return AzureOpenAIGenerator(args.aoai_endpoint, args.aoai_deployment, api_key=key, token_provider=tp)


def _finish(res, out) -> int:
    paths = report.write(res, out)
    print(report.to_markdown(res))
    print(f"reports: {', '.join(str(p) for p in paths.values())}", file=sys.stderr)
    return 0


def cmd_demo(args) -> int:
    res = run_arena(demo_mod.QUESTIONS, demo_mod.arms(), assertion_checker=LexicalAssertionChecker(),
                    baseline="no_retrieval", corpus_ids=set(demo_mod.CORPUS))
    return _finish(res, args.out)


def cmd_score(args) -> int:
    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit("no questions loaded")
    arms = [ReplayArm(n, load_answers(p, n), controlled=n in (args.controlled or [])) for n, p in _arm_specs(args.answers).items()]
    checker = judge = None
    if args.checker == "lexical":
        checker = LexicalAssertionChecker()
    elif args.checker == "llm":
        checker = LLMAssertionChecker(_aoai(args).chat)
    if args.judge == "llm":
        judge = LLMPairwiseJudge(_aoai(args).chat)
    corpus = None
    if args.corpus_ids:
        corpus = {l.strip() for l in Path(args.corpus_ids).read_text(encoding="utf-8").splitlines() if l.strip()}
    res = run_arena(questions, arms, assertion_checker=checker, pairwise_judge=judge,
                    criteria=args.criteria.split(",") if args.criteria else None,
                    baseline=args.baseline, corpus_ids=corpus, workers=args.workers)
    return _finish(res, args.out)


def cmd_export(args) -> int:
    questions = load_questions(args.questions)
    answers = {n: load_answers(p, n) for n, p in _arm_specs(args.answers).items()}
    out = export_autoe(questions, answers, args.out, base=args.base)
    print(f"wrote AutoE inputs + settings.yaml to {out}")
    return 0


def cmd_power(args) -> int:
    for n in args.n:
        print(f"{n:>4} questions: need >= {stats.min_wins_for_significance(n)} wins (no ties) for p < 0.05")
    for w in args.win_rate:
        print(f"true win rate {w:.2f}: ~{stats.questions_needed(w)} questions for 80% power (alpha 0.05)")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    arena = sub.add_parser("arena", help="retrieval A/B arena (GraphRAG vs vector RAG vs live search)")
    asub = arena.add_subparsers(dest="arena_cmd", required=True)

    p = asub.add_parser("demo", help="offline demo on a fictional kinase corpus")
    p.add_argument("--out", default="sciprobe-out/arena-demo")
    p.set_defaults(fn=cmd_demo)

    p = asub.add_parser("score", help="score recorded answers from any pipeline")
    p.add_argument("--questions", nargs="+", required=True, help="question/assertion files (BenchmarkQED JSON or JSONL)")
    p.add_argument("--answers", action="append", required=True, metavar="ARM=PATH")
    p.add_argument("--controlled", action="append", help="arm names that share one generator (retrieval-only difference)")
    p.add_argument("--baseline", help="arm used as the no-retrieval baseline for lift")
    p.add_argument("--checker", choices=["lexical", "llm", "none"], default="lexical")
    p.add_argument("--judge", choices=["assertions", "llm", "none"], default="assertions",
                   help="pairwise judge: assertion counts (default, no LLM) or counterbalanced LLM")
    p.add_argument("--criteria", help="comma list for the LLM judge (default: BenchmarkQED's four)")
    p.add_argument("--corpus-ids", help="file with one corpus doc id / DOI per line (flags out-of-corpus citations)")
    p.add_argument("--aoai-endpoint")
    p.add_argument("--aoai-deployment")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="sciprobe-out/arena")
    p.set_defaults(fn=cmd_score)

    p = asub.add_parser("export-autoe", help="write answers in BenchmarkQED AutoE layout + settings.yaml")
    p.add_argument("--questions", nargs="+", required=True)
    p.add_argument("--answers", action="append", required=True, metavar="ARM=PATH")
    p.add_argument("--base", required=True)
    p.add_argument("--out", default="sciprobe-out/autoe")
    p.set_defaults(fn=cmd_export)

    p = asub.add_parser("power", help="how many questions do you need?")
    p.add_argument("--n", type=int, nargs="+", default=[10, 50, 100, 200])
    p.add_argument("--win-rate", type=float, nargs="+", default=[0.6, 0.65, 0.7])
    p.set_defaults(fn=cmd_power)
