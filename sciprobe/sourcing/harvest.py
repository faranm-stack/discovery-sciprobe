"""Harvest listed evaluation prompts from a Discovery catalog (GitHub or a local
checkout such as a PR branch):

* starter-kits/*/kit.json  -> samplePrompts (+ expectedOutput rubric)
* agents/*/README.md       -> "Usage" tables (| Prompt | Input Files | Description |)
* agents/*/metadata.yaml, agent.yaml -> tags / description (capability profile)
* agents/*/tools/*/example-input-files/* -> known inputs shipped with the agent
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .. import net


@dataclass
class ListedPrompt:
    agent: str | None  # None for kit-level prompts that span agents
    source: str  # "kit" | "readme" | "pr" | "custom"
    source_path: str
    prompt: str
    input_files: list[str] = field(default_factory=list)
    expected: str = ""
    difficulty: str = ""
    candidate_agents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentProfile:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    example_inputs: list[str] = field(default_factory=list)
    listed: list[ListedPrompt] = field(default_factory=list)


# ------------------------------------------------------------------ sources

class GitHubCatalog:
    def __init__(self, owner: str = "microsoft", repo: str = "discovery", ref: str = "main"):
        self.owner, self.repo, self.ref = owner, repo, ref
        self._paths: list[str] | None = None

    @property
    def label(self) -> str:
        return f"github:{self.owner}/{self.repo}@{self.ref}"

    def paths(self) -> list[str]:
        if self._paths is None:
            tree = net.get_json(f"https://api.github.com/repos/{self.owner}/{self.repo}/git/trees/{self.ref}",
                                {"recursive": 1})
            self._paths = [t["path"] for t in tree.get("tree", []) if t.get("type") == "blob"]
        return self._paths

    def read(self, path: str) -> str:
        return net.get_text(f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.ref}/{path}")


class LocalCatalog:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def label(self) -> str:
        return f"local:{self.root}"

    def paths(self) -> list[str]:
        return [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file() and ".git" not in p.parts]

    def read(self, path: str) -> str:
        return (self.root / path).read_text(encoding="utf-8-sig", errors="replace")


def open_catalog(spec: str):
    """``github:owner/repo@ref`` (default microsoft/discovery@main) or a local path."""
    if spec.startswith("github:") or spec == "github":
        body = spec.partition(":")[2] or "microsoft/discovery"
        repo_part, _, ref = body.partition("@")
        owner, _, repo = repo_part.partition("/")
        return GitHubCatalog(owner or "microsoft", repo or "discovery", ref or "main")
    return LocalCatalog(spec)


# ------------------------------------------------------------------ parsers

def parse_kit(text: str, path: str) -> list[ListedPrompt]:
    kit = json.loads(text)
    agents = [r["ref"].split("/")[-1] for r in kit.get("agentRefs", []) if r.get("ref")]
    out = []
    for sp in kit.get("samplePrompts", []):
        if sp.get("prompt"):
            out.append(ListedPrompt(None, "kit", path, sp["prompt"].strip(), expected=sp.get("expectedOutput", ""),
                                    difficulty=sp.get("difficulty", ""), candidate_agents=agents))
    return out


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_readme_usage(text: str, agent: str, path: str) -> list[ListedPrompt]:
    """Every markdown table whose header has a 'Prompt' column."""
    out: list[ListedPrompt] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            header = [h.lower() for h in _cells(line)]
            pcol = next((j for j, h in enumerate(header) if h in ("prompt", "example prompt", "sample prompt", "query")), None)
            icol = next((j for j, h in enumerate(header) if "input" in h), None)
            dcol = next((j for j, h in enumerate(header) if h in ("description", "expected", "expected output", "output")), None)
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = _cells(lines[i])
                if pcol is not None and pcol < len(row) and row[pcol]:
                    prompt = row[pcol].strip().strip('"“”').strip()
                    files = []
                    if icol is not None and icol < len(row):
                        files = [f.strip(" `") for f in re.split(r"[,;]| and ", row[icol]) if f.strip(" `-—")]
                    desc = row[dcol] if dcol is not None and dcol < len(row) else ""
                    out.append(ListedPrompt(agent, "readme", path, prompt, files, desc))
                i += 1
            continue
        i += 1
    return out


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Enough YAML for catalog metadata: top-level scalars, '>'/'|' blocks, and '- item' lists."""
    out: dict[str, Any] = {}
    key = None
    block: list[str] | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        top = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", raw)
        if top:
            if key and block is not None:
                out[key] = " ".join(block).strip()
            key, val = top.group(1), top.group(2).strip()
            block = None
            if val in (">", "|", ">-", "|-"):
                block = []
            elif val:
                out[key] = val.strip("'\"")
            else:
                out[key] = []
            continue
        if key is None:
            continue
        if block is not None and raw.startswith(" "):
            block.append(raw.strip())
        elif isinstance(out.get(key), list) and raw.strip().startswith("- "):
            out[key].append(raw.strip()[2:].strip("'\""))
    if key and block is not None:
        out[key] = " ".join(block).strip()
    return out


# ------------------------------------------------------------------ harvest

def _stem(t: str) -> str:
    s = re.sub(r"(?:ations?|ings?|ers?|ed|es|s)$", "", t)
    return s if len(s) >= 3 else t


def route_prompt(prompt: str, candidates: list[AgentProfile]) -> list[str]:
    """Agents a kit-level prompt is really about: explicitly named ones first,
    otherwise the best term overlap with each agent's name/tags/description."""
    from ..arena.text import content_tokens

    low = prompt.lower()
    named = [c.name for c in candidates if c.name.lower() in low or c.name.replace("-", " ").lower() in low]
    if named:
        return named
    ptoks = set(content_tokens(prompt))
    pst = {_stem(t) for t in ptoks}
    best, best_score = [], 0.0
    for c in candidates:
        atoks = set(content_tokens(" ".join([c.name.replace("-", " "), " ".join(c.tags).replace("-", " "), c.description])))
        score = len(ptoks & atoks) + 0.5 * len(pst & {_stem(t) for t in atoks})
        if score > best_score:
            best, best_score = [c.name], score
        elif score == best_score and score > 0:
            best.append(c.name)
    return best[:2]


def harvest(catalog, agents: list[str] | None = None, include_readme: bool = True) -> dict[str, AgentProfile]:
    paths = catalog.paths()
    wanted = {a.lower() for a in agents} if agents else None
    names = sorted({p.split("/")[1] for p in paths if p.startswith("agents/") and p.count("/") >= 2})
    profiles: dict[str, AgentProfile] = {}
    for name in names:
        if wanted and name.lower() not in wanted:
            continue
        prof = AgentProfile(name)
        meta_path = f"agents/{name}/metadata.yaml"
        if meta_path in paths:
            meta = parse_simple_yaml(catalog.read(meta_path))
            prof.description = str(meta.get("description") or "")
            prof.tags = [t for t in meta.get("tags", []) if isinstance(t, str)] if isinstance(meta.get("tags"), list) else []
        prof.example_inputs = sorted(p for p in paths if p.startswith(f"agents/{name}/") and "/example-input-files/" in p
                                     and not p.lower().endswith(".md"))
        readme = f"agents/{name}/README.md"
        if include_readme and readme in paths:
            try:
                prof.listed += parse_readme_usage(catalog.read(readme), name, readme)
            except Exception:  # noqa: BLE001
                pass
        profiles[name] = prof
    all_names = set(names)
    for kp in (p for p in paths if p.startswith("starter-kits/") and p.endswith("kit.json")):
        try:
            kit_prompts = parse_kit(catalog.read(kp), kp)
        except Exception:  # noqa: BLE001
            continue
        for lp in kit_prompts:
            cands = [profiles[a] for a in lp.candidate_agents if a in profiles]
            if not cands and wanted:
                continue
            if not cands:
                cands = [AgentProfile(a) for a in lp.candidate_agents if a in all_names]
            for a in route_prompt(lp.prompt, cands):
                if a in profiles:
                    profiles[a].listed.append(ListedPrompt(a, lp.source, lp.source_path, lp.prompt, lp.input_files,
                                                           lp.expected, lp.difficulty, lp.candidate_agents))
    return profiles
