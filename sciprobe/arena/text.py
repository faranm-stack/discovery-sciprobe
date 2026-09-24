"""Tiny text utilities shared by offline retrievers, citation and assertion checks."""
from __future__ import annotations

import re

_STOP = set(
    "a an and are as at be by for from has have how in into is it its of on or that the their this to was were "
    "what when which while with within does do did can could should would may might than then there these those "
    "also such via per not no yes more most less many much over under between across about after before".split()
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/.]*[A-Za-z0-9]|[A-Za-z0-9]")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in _STOP and (len(t) > 2 or any(ch.isdigit() for ch in t))]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]
