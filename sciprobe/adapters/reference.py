"""Offline agents: an RDKit-backed gold agent and a fault injector.

The gold agent proves the harness end to end without any cloud access. The
fault injector wraps any adapter and plants realistic scientific mistakes so we
can verify the harness actually catches them.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time

from ..model import AgentReply

_SMILES_RE = re.compile(r"SMILES\s+(\S+)")
_ANSWER_LINE_RE = re.compile(r"ANSWER\s*[:=]\s*(\{[^{}]*\})", re.IGNORECASE)

_KEYWORDS = [
    ("exact mass", "exact_mass"),
    ("average molecular weight", "mw"),
    ("molecular formula", "formula"),
    ("hydrogen-bond donors", "hbd"),
    ("hydrogen-bond acceptors", "hba"),
    ("polar surface area", "tpsa"),
    ("heavy", "heavy_atoms"),
    ("stereocenters", "stereocenters"),
    ("rotatable", "rotb"),
    ("logp", "logp"),
    ("rings", "rings"),
]


def _answer(value, unit: str = "", reason: str | None = None) -> str:
    obj = {"value": value, "unit": unit}
    if reason:
        obj["reason"] = reason
    return f"ANSWER: {json.dumps(obj)}"


class RDKitReferenceAgent:
    """Answers cheminformatics prompts by actually running RDKit."""

    name = "rdkit-reference"

    def ask(self, agent: str, prompt: str) -> AgentReply:
        from rdkit import Chem, RDLogger

        from ..packs.cheminformatics import props

        RDLogger.DisableLog("rdApp.*")
        t0 = time.perf_counter()
        m = _SMILES_RE.search(prompt)
        low = prompt.lower()
        key = next((k for kw, k in _KEYWORDS if kw in low), None)
        if not m or key is None:
            text = "I do not have a tool that can answer this question.\n" + _answer(None, reason="out of scope")
            return AgentReply(text=text, latency_ms=(time.perf_counter() - t0) * 1e3)
        smiles = m.group(1).rstrip(".,;")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            text = f"RDKit could not parse SMILES {smiles}: invalid SMILES.\n" + _answer(None, reason="invalid SMILES")
        else:
            spec = next(p for p in props() if p.key == key)
            val = spec.compute(mol)
            if isinstance(val, float):
                val = round(val, 4)
            text = f"Computed with RDKit: {val}\n" + _answer(val, spec.unit or "")
        return AgentReply(text=text, latency_ms=(time.perf_counter() - t0) * 1e3, meta={"agent": agent})


class FaultInjectingAdapter:
    """Wrap an adapter and plant scientific failure modes with given probabilities.

    faults keys:
      exact_mass_for_mw     report monoisotopic mass when average MW is asked
      prefix_slip           multiply the value by 1000 but keep the unit label
      input_order_noise     add +/-(1-3)% noise keyed on the exact prompt text, so
                            equivalent inputs (random SMILES, rotated XYZ) disagree
      hallucinate_invalid   replace a correct refusal with a confident number
      drop_answer           strip the ANSWER line, leaving prose only
    """

    def __init__(self, inner, faults: dict[str, float], seed: int = 0, name: str | None = None):
        self.inner = inner
        self.faults = faults
        self.seed = seed
        self.name = name or f"faulty({getattr(inner, 'name', 'inner')})"

    def _rng(self, prompt: str, tag: str) -> random.Random:
        h = hashlib.sha256(f"{self.seed}:{tag}:{prompt}".encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def _fires(self, prompt: str, fault: str) -> bool:
        p = self.faults.get(fault, 0.0)
        return p > 0 and self._rng(prompt, fault).random() < p

    def ask(self, agent: str, prompt: str) -> AgentReply:
        rep = self.inner.ask(agent, prompt)
        if rep.error or not rep.text:
            return rep
        m = _ANSWER_LINE_RE.search(rep.text)
        if not m:
            return rep
        obj = json.loads(m.group(1))
        value, unit = obj.get("value"), obj.get("unit", "")
        planted: list[str] = []

        if value is None and self._fires(prompt, "hallucinate_invalid"):
            value, unit = round(self._rng(prompt, "fake").uniform(50, 400), 3), "g/mol"
            planted.append("hallucinate_invalid")
        elif isinstance(value, (int, float)):
            # At most one value fault per answer, so each diagnosis is attributable.
            if "average molecular weight" in prompt.lower() and self._fires(prompt, "exact_mass_for_mw"):
                from rdkit import Chem
                from rdkit.Chem import Descriptors

                sm = _SMILES_RE.search(prompt)
                mol = Chem.MolFromSmiles(sm.group(1)) if sm else None
                if mol is not None:
                    value = round(Descriptors.ExactMolWt(mol), 4)
                    planted.append("exact_mass_for_mw")
            if not planted and self._fires(prompt, "prefix_slip"):
                value = value * 1000
                planted.append("prefix_slip")
            if not planted and isinstance(value, float) and self._fires(prompt, "input_order_noise"):
                r = self._rng(prompt, "noise")
                value = value * (1 + r.choice([-1, 1]) * r.uniform(0.01, 0.03))
                planted.append("input_order_noise")

        if self._fires(prompt, "drop_answer"):
            text = f"The result is approximately {value} {unit}." if value is not None else "Something went wrong."
            planted.append("drop_answer")
        else:
            new_obj = {**obj, "value": value, "unit": unit}
            if value is not None:
                new_obj.pop("reason", None)
            text = rep.text[: m.start()] + f"ANSWER: {json.dumps(new_obj)}" + rep.text[m.end():]
        return AgentReply(text=text, latency_ms=rep.latency_ms, meta={**rep.meta, "planted": planted})
