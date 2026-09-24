"""Metamorphic / invariance probes.

These need no ground truth: they send scientifically *equivalent* inputs to an
agent and check that its answers agree. A molecule does not change when its
SMILES string is re-ordered; a molecular energy does not change when the
molecule is rotated or its atoms are listed in a different order.
"""
from __future__ import annotations

import math
import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from . import units
from .extract import extract
from .model import ANSWER_INSTRUCTION, AgentReply, Status


class Relation(str, Enum):
    SMILES_RANDOMIZATION = "smiles_randomization"
    ENANTIOMER = "enantiomer"
    RIGID_MOTION = "rigid_motion"
    ATOM_PERMUTATION = "atom_permutation"
    UNIT_REFRAME = "unit_reframe"
    PARAPHRASE = "paraphrase"
    REPEAT = "repeat"


@dataclass
class InvarianceProbe:
    id: str
    domain: str
    agents: list[str]
    relation: Relation
    template: str  # uses {input}; UNIT_REFRAME also uses {unit}
    base_input: str
    unit: str | None = None  # unit all answers are compared in
    rel_tol: float = 0.0
    abs_tol: float = 0.0
    n_variants: int = 4
    extras: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Variant:
    label: str
    prompt: str
    input: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvarianceResult:
    probe_id: str
    agent: str
    domain: str
    relation: str
    status: Status
    score: float
    spread: float | None
    tolerance: float
    values: list[dict[str, Any]]
    diagnosis: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ---------------------------------------------------------------- generators

def randomized_smiles(smiles: str, n: int, seed: int = 0) -> list[str]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES {smiles!r}")
    canon = Chem.MolToSmiles(mol)
    seen = {smiles, canon}
    out: list[str] = []
    for attempt in range(8):
        for s in Chem.MolToRandomSmilesVect(mol, max(4 * n, 8), randomSeed=seed + 7919 * attempt):
            if s not in seen and Chem.MolToSmiles(Chem.MolFromSmiles(s)) == canon:
                seen.add(s)
                out.append(s)
                if len(out) >= n:
                    return out
    return out


def mirror_smiles(smiles: str) -> str:
    """Invert every tetrahedral centre (@ <-> @@): the enantiomer."""
    return re.sub(r"@@|@", lambda m: "@" if m.group(0) == "@@" else "@@", smiles)


def parse_xyz(block: str) -> list[tuple[str, float, float, float]]:
    atoms = []
    for line in block.strip().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    if not atoms:
        raise ValueError("no atoms in XYZ block")
    return atoms


def format_xyz(atoms: list[tuple[str, float, float, float]]) -> str:
    return "\n".join(f"{el:<2} {x:12.6f} {y:12.6f} {z:12.6f}" for el, x, y, z in atoms)


def _random_rotation(rng: random.Random) -> list[list[float]]:
    # Uniform random rotation from a random unit quaternion (Shoemake 1992).
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q = (
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    )
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def rigid_motion(block: str, rng: random.Random, max_shift: float = 5.0) -> str:
    atoms = parse_xyz(block)
    r = _random_rotation(rng)
    t = [rng.uniform(-max_shift, max_shift) for _ in range(3)]
    moved = []
    for el, x, y, z in atoms:
        v = (x, y, z)
        moved.append((el, *(sum(r[i][j] * v[j] for j in range(3)) + t[i] for i in range(3))))
    return format_xyz(moved)


def permute_atoms(block: str, rng: random.Random) -> str:
    atoms = parse_xyz(block)
    for _ in range(20):
        perm = atoms[:]
        rng.shuffle(perm)
        if perm != atoms or len(atoms) < 2:
            return format_xyz(perm)
    return format_xyz(atoms[::-1])


def _wrap(template: str, **kw: str) -> str:
    return f"{template.format(**kw).strip()}\n\n{ANSWER_INSTRUCTION}"


def variants(probe: InvarianceProbe, seed: int = 0) -> list[Variant]:
    rng = random.Random(f"{probe.id}:{seed}")
    rel, base, tpl = probe.relation, probe.base_input, probe.template
    n = max(2, probe.n_variants)
    out: list[Variant] = []

    if rel == Relation.SMILES_RANDOMIZATION:
        out.append(Variant("original", _wrap(tpl, input=base), base))
        for i, s in enumerate(randomized_smiles(base, n - 1, seed)):
            out.append(Variant(f"random_smiles_{i + 1}", _wrap(tpl, input=s), s))
    elif rel == Relation.ENANTIOMER:
        mirrored = mirror_smiles(base)
        if mirrored == base:
            raise ValueError(f"{probe.id}: base SMILES has no tetrahedral stereo to mirror")
        out.append(Variant("original", _wrap(tpl, input=base), base))
        out.append(Variant("enantiomer", _wrap(tpl, input=mirrored), mirrored))
    elif rel in (Relation.RIGID_MOTION, Relation.ATOM_PERMUTATION):
        canon = format_xyz(parse_xyz(base))
        out.append(Variant("original", _wrap(tpl, input=canon), canon))
        for i in range(n - 1):
            moved = rigid_motion(base, rng) if rel == Relation.RIGID_MOTION else permute_atoms(base, rng)
            out.append(Variant(f"{rel.value}_{i + 1}", _wrap(tpl, input=moved), moved))
    elif rel == Relation.UNIT_REFRAME:
        for u in probe.extras["units"]:
            out.append(Variant(f"asked_in_{u}", _wrap(tpl, input=base, unit=u), base, {"unit": u}))
    elif rel == Relation.PARAPHRASE:
        for i, t in enumerate(probe.extras["templates"]):
            out.append(Variant(f"paraphrase_{i}", _wrap(t, input=base), base))
    elif rel == Relation.REPEAT:
        for i in range(n):
            out.append(Variant(f"repeat_{i}", _wrap(tpl, input=base), base))
    else:  # pragma: no cover
        raise ValueError(f"unknown relation {rel}")
    return out


# ---------------------------------------------------------------- evaluation

def evaluate(probe: InvarianceProbe, agent: str, replies: list[tuple[Variant, AgentReply]]) -> InvarianceResult:
    dim = units.dimension(probe.unit) if probe.unit is not None and units.is_known(probe.unit) else None
    values: list[dict[str, Any]] = []
    numeric: list[float] = []
    refused = errors = 0
    for var, rep in replies:
        row: dict[str, Any] = {"label": var.label, "input": var.input, "value": None, "raw": None, "note": ""}
        if rep.error:
            errors += 1
            row["note"] = f"error: {rep.error}"
            values.append(row)
            continue
        ex = extract(rep.text, dim)
        row["raw"] = None if ex.value is None else f"{ex.value} {ex.unit or ''}".strip()
        if ex.refused:
            refused += 1
            row["note"] = f"refused: {ex.reason}"
        elif ex.value is not None:
            try:
                v = float(ex.value)
                src_unit = ex.unit or var.meta.get("unit") or probe.unit
                if probe.unit and src_unit:
                    v = units.convert(v, src_unit, probe.unit)
                row["value"] = v
                numeric.append(v)
            except (TypeError, ValueError, units.UnitError) as e:
                row["note"] = f"unusable: {e}"
        values.append(row)

    ref_mag = statistics.fmean(abs(v) for v in numeric) if numeric else 0.0
    tol = max(probe.abs_tol, probe.rel_tol * ref_mag)
    common = dict(probe_id=probe.id, agent=agent, domain=probe.domain, relation=probe.relation.value,
                  tolerance=tol, values=values, tags=list(probe.tags))

    if errors == len(replies):
        return InvarianceResult(status=Status.ERROR, score=0.0, spread=None, diagnosis="all variants errored", **common)
    if refused and numeric:
        bad = [r["label"] for r in values if r["note"].startswith("refused")]
        return InvarianceResult(status=Status.FAIL, score=0.0, spread=None,
                                diagnosis=f"answered some equivalent inputs but refused {bad}", **common)
    if len(numeric) < 2:
        return InvarianceResult(status=Status.NO_ANSWER, score=0.0, spread=None,
                                diagnosis=f"only {len(numeric)} comparable answer(s)", **common)

    spread = max(numeric) - min(numeric)
    if spread <= tol:
        return InvarianceResult(status=Status.PASS, score=1.0, spread=spread, **common)
    score = max(0.0, 1.0 - (spread - tol) / (4.0 * tol)) if tol > 0 else 0.0
    lo = min(values, key=lambda r: r["value"] if r["value"] is not None else math.inf)
    hi = max(values, key=lambda r: r["value"] if r["value"] is not None else -math.inf)
    diag = f"spread {spread:.4g} {probe.unit or ''} > tol {tol:.3g}; min at {lo['label']}, max at {hi['label']}"
    guess = units.diagnose_ratio(hi["value"], lo["value"]) if lo["value"] else None
    if guess:
        diag += f" ({guess})"
    return InvarianceResult(status=Status.FAIL, score=round(score, 4), spread=spread, diagnosis=diag, **common)
