"""Adapters connect the harness to something that answers prompts."""
from __future__ import annotations

import time
from typing import Callable, Protocol

from ..model import AgentReply


class Adapter(Protocol):
    name: str

    def ask(self, agent: str, prompt: str) -> AgentReply: ...


class CallableAdapter:
    """Wrap any ``fn(agent, prompt) -> str`` (handy for notebooks and tests)."""

    def __init__(self, fn: Callable[[str, str], str], name: str = "callable"):
        self.fn = fn
        self.name = name

    def ask(self, agent: str, prompt: str) -> AgentReply:
        t0 = time.perf_counter()
        try:
            text = self.fn(agent, prompt)
        except Exception as e:  # noqa: BLE001 - surface any backend failure as a verdict
            return AgentReply(text="", error=f"{type(e).__name__}: {e}", latency_ms=(time.perf_counter() - t0) * 1e3)
        return AgentReply(text=text, latency_ms=(time.perf_counter() - t0) * 1e3)


from .discovery import DiscoveryAdapter  # noqa: E402
from .reference import FaultInjectingAdapter, RDKitReferenceAgent  # noqa: E402
from .replay import RecordingAdapter, ReplayAdapter  # noqa: E402

__all__ = [
    "Adapter",
    "CallableAdapter",
    "DiscoveryAdapter",
    "FaultInjectingAdapter",
    "RDKitReferenceAgent",
    "RecordingAdapter",
    "ReplayAdapter",
]
