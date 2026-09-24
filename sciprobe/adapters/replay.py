"""Record/replay: score transcripts produced elsewhere (e.g. by the Discovery
Toolbox's ``discovery_askAgent``, which logs to ~/.md-toolbox/conversations)."""
from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from pathlib import Path

from ..model import AgentReply

DEFAULT_TOOLBOX_LOG_DIR = Path.home() / ".md-toolbox" / "conversations"


def norm_prompt(p: str) -> str:
    return re.sub(r"\s+", " ", p or "").strip()


def _iter_files(sources: list[str | Path]):
    for s in sources:
        p = Path(s).expanduser()
        if p.is_dir():
            yield from sorted(p.glob("*.jsonl"))
        elif p.is_file():
            yield p


class ReplayAdapter:
    """Serve recorded replies keyed by (agent, prompt). Understands two formats:

    * Discovery Toolbox conversation logs (header + user/agent message lines)
    * flat records: {"agent": ..., "prompt": ..., "text": ..., "latency_ms": ...}

    Duplicate prompts (e.g. REPEAT probes) are served in recorded order.
    """

    name = "replay"

    def __init__(self, sources: list[str | Path] | None = None):
        self._store: dict[tuple[str, str], list[AgentReply]] = defaultdict(list)
        self._cursor: dict[tuple[str, str], int] = defaultdict(int)
        self._lock = threading.Lock()
        for f in _iter_files(sources or [DEFAULT_TOOLBOX_LOG_DIR]):
            self._load(f)

    def __len__(self) -> int:
        return sum(len(v) for v in self._store.values())

    def add(self, agent: str, prompt: str, reply: AgentReply) -> None:
        self._store[(agent.lower(), norm_prompt(prompt))].append(reply)

    def _load(self, path: Path) -> None:
        agent = None
        pending_prompt = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") == "header":
                agent = rec.get("agentName")
                continue
            if rec.get("kind") == "message":
                msg = rec.get("message", {})
                if msg.get("role") == "user":
                    pending_prompt = msg.get("text", "")
                elif msg.get("role") == "agent" and pending_prompt is not None:
                    who = msg.get("agentName") or agent
                    if who:
                        self.add(who, pending_prompt, AgentReply(
                            text=msg.get("text", ""), latency_ms=msg.get("latencyMs"),
                            meta={"source": str(path), "transport": msg.get("transport")}))
                    pending_prompt = None
                continue
            if "agent" in rec and "prompt" in rec:
                self.add(rec["agent"], rec["prompt"], AgentReply(
                    text=rec.get("text") or rec.get("response") or "", latency_ms=rec.get("latency_ms"),
                    error=rec.get("error"), meta={"source": str(path)}))

    def ask(self, agent: str, prompt: str) -> AgentReply:
        key = (agent.lower(), norm_prompt(prompt))
        with self._lock:
            replies = self._store.get(key)
            if not replies:
                return AgentReply(text="", error="no recorded reply for this (agent, prompt)")
            i = self._cursor[key]
            if i >= len(replies):
                return AgentReply(text="", error=f"only {len(replies)} recorded reply(ies) for this prompt")
            self._cursor[key] = i + 1
            return replies[i]


class RecordingAdapter:
    """Pass-through adapter that appends every exchange to a JSONL file so any
    live run can be re-scored later, bit-for-bit."""

    def __init__(self, inner, path: str | Path):
        self.inner = inner
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.name = f"recording({getattr(inner, 'name', 'inner')})"
        self._lock = threading.Lock()

    def ask(self, agent: str, prompt: str) -> AgentReply:
        rep = self.inner.ask(agent, prompt)
        rec = {"agent": agent, "prompt": prompt, "text": rep.text, "latency_ms": rep.latency_ms, "error": rep.error}
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rep

    def cleanup(self, mode: str = "all"):
        fn = getattr(self.inner, "cleanup", None)
        return fn(mode) if fn else None
