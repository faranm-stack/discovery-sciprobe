"""Cheminformatics oracles. Ground truth is computed live with RDKit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..invariance import InvarianceProbe, Relation
from ..model import Confusion, Kind, OracleTask, Reference

AGENTS = ["mol-toolkit", "core-python-agent"]

MOLECULES: dict[str, str] = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "s_naproxen": "C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1",
    "s_nicotine": "CN1CCC[C@H]1c1cccnc1",
    "oseltamivir": "CCC(CC)O[C@@H]1C=C(C[C@@H]([C@H]1NC(C)=O)N)C(=O)OCC",
}

CORE = {
    ("aspirin", "mw"), ("aspirin", "exact_mass"), ("aspirin", "formula"), ("aspirin", "tpsa"),
    ("s_naproxen", "mw"), ("s_naproxen", "stereocenters"), ("s_naproxen", "hba"),
    ("oseltamivir", "formula"), ("oseltamivir", "stereocenters"), ("oseltamivir", "hbd"),
}


@dataclass(frozen=True)
class PropSpec:
    key: str
    question: str
    kind: Kind
    unit: str | None
    compute: Callable[[Any], Any]
    abs_tol: float = 0.0
    confusions: Callable[[Any], list[Confusion]] | None = None


def _mol(smiles: str):
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"invalid SMILES {smiles!r}")
    return m


def _props() -> list[PropSpec]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    def stereo_all(m):
        return len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))

    def stereo_assigned(m):
        return len(Chem.FindMolChiralCenters(m, includeUnassigned=False, useLegacyImplementation=False))

    def diff(label, fn):
        return lambda m: [Confusion(fn(m), label)]

    return [
        PropSpec("mw", "What is the average molecular weight (standard atomic weights, NOT the monoisotopic "
                 "mass) of the molecule with SMILES {smiles} ? Report it in g/mol.",
                 Kind.NUMERIC, "g/mol", Descriptors.MolWt, 0.05,
                 lambda m: [Confusion(Descriptors.ExactMolWt(m), "monoisotopic exact mass reported instead of average MW", "g/mol")]),
        PropSpec("exact_mass", "What is the monoisotopic exact mass of the molecule with SMILES {smiles} ? Report it in Da.",
                 Kind.NUMERIC, "Da", Descriptors.ExactMolWt, 0.005,
                 lambda m: [Confusion(Descriptors.MolWt(m), "average MW reported instead of monoisotopic mass", "Da")]),
        PropSpec("formula", "What is the molecular formula (Hill notation) of the molecule with SMILES {smiles} ?",
                 Kind.FORMULA, None, rdMolDescriptors.CalcMolFormula),
        PropSpec("hbd", "How many Lipinski hydrogen-bond donors (the NHOH count: number of N-H and O-H bonds) does "
                 "the molecule with SMILES {smiles} have?", Kind.EXACT, None, Lipinski.NHOHCount,
                 confusions=diff("counted donor atoms (NH2 = 1) instead of N-H/O-H bonds (Lipinski NHOH: NH2 = 2)",
                                 Lipinski.NumHDonors)),
        PropSpec("hba", "How many Lipinski hydrogen-bond acceptors (the NO count: number of N and O atoms) does the "
                 "molecule with SMILES {smiles} have?", Kind.EXACT, None, Lipinski.NOCount,
                 confusions=diff("RDKit NumHAcceptors definition used instead of Lipinski NO count", Lipinski.NumHAcceptors)),
        PropSpec("tpsa", "What is the topological polar surface area (Ertl TPSA, N and O contributions only) of the "
                 "molecule with SMILES {smiles} ? Report it in Å^2.", Kind.NUMERIC, "Å^2", rdMolDescriptors.CalcTPSA, 0.5),
        PropSpec("heavy_atoms", "How many heavy (non-hydrogen) atoms does the molecule with SMILES {smiles} have?",
                 Kind.EXACT, None, lambda m: m.GetNumHeavyAtoms(),
                 confusions=diff("hydrogens counted as heavy atoms", lambda m: Chem.AddHs(m).GetNumAtoms())),
        PropSpec("rings", "How many rings (smallest set of smallest rings) does the molecule with SMILES {smiles} have?",
                 Kind.EXACT, None, rdMolDescriptors.CalcNumRings),
        PropSpec("stereocenters", "How many tetrahedral stereocenters (assigned or unassigned) does the molecule "
                 "with SMILES {smiles} have?", Kind.EXACT, None, stereo_all,
                 confusions=diff("counted only stereocenters assigned in the SMILES", stereo_assigned)),
        PropSpec("rotb", "How many rotatable bonds (RDKit default definition) does the molecule with SMILES {smiles} have?",
                 Kind.NUMERIC, None, rdMolDescriptors.CalcNumRotatableBonds, 1.0),
        PropSpec("logp", "What is the Wildman-Crippen logP of the molecule with SMILES {smiles} ?",
                 Kind.NUMERIC, None, Crippen.MolLogP, 0.3),
    ]


_PROPS_CACHE: list[PropSpec] | None = None


def props() -> list[PropSpec]:
    global _PROPS_CACHE
    if _PROPS_CACHE is None:
        _PROPS_CACHE = _props()
    return _PROPS_CACHE


def tasks() -> list[OracleTask]:
    out: list[OracleTask] = []
    for name, smi in MOLECULES.items():
        m = _mol(smi)
        for p in props():
            val = p.compute(m)
            if p.kind == Kind.NUMERIC:
                ref = Reference(Kind.NUMERIC, float(val), p.unit, abs_tol=p.abs_tol, source="RDKit (computed live)")
            else:
                ref = Reference(p.kind, val, p.unit, source="RDKit (computed live)")
            confs = [c for c in (p.confusions(m) if p.confusions else []) if str(c.value) != str(val)]
            tags = ["rdkit-verifiable"] + (["core"] if (name, p.key) in CORE else [])
            out.append(OracleTask(
                id=f"chem.{p.key}.{name}", domain="cheminformatics", agents=AGENTS,
                question=p.question.format(smiles=smi), reference=ref, confusions=confs, tags=tags,
            ))
    return out


def probes() -> list[InvarianceProbe]:
    mw_q = "What is the average molecular weight (standard atomic weights) of the molecule with SMILES {input} ? Report it in g/mol."
    tpsa_q = "What is the Ertl topological polar surface area of the molecule with SMILES {input} ? Report it in Å^2."
    logp_q = "What is the Wildman-Crippen logP of the molecule with SMILES {input} ?"
    return [
        InvarianceProbe("chem.inv.mw.smiles.oseltamivir", "cheminformatics", AGENTS, Relation.SMILES_RANDOMIZATION,
                        mw_q, MOLECULES["oseltamivir"], "g/mol", abs_tol=0.05, n_variants=4, tags=["core"]),
        InvarianceProbe("chem.inv.tpsa.smiles.naproxen", "cheminformatics", AGENTS, Relation.SMILES_RANDOMIZATION,
                        tpsa_q, MOLECULES["s_naproxen"], "Å^2", abs_tol=0.5, n_variants=3),
        InvarianceProbe("chem.inv.logp.enantiomer.naproxen", "cheminformatics", AGENTS, Relation.ENANTIOMER,
                        logp_q, MOLECULES["s_naproxen"], None, abs_tol=0.01, tags=["core"]),
    ]
