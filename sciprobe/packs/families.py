"""Verifier families: one verification strategy per *kind* of scientific agent,
so the long tail of catalog agents with no listed questions still gets checked.

Family                Agents                                  How the answer is known
--------------------  --------------------------------------  -------------------------------------------
sequence              biopython, core-python-agent            exact computation (standard genetic code)
crystallography       spglib, pymatgen, ase                   symmetry (space groups) + first-principles density
kinetics              cantera, tellurium, copasi-basico       analytic solutions of the rate equations
metabolic modelling   cobrapy                                 published textbook FBA result
statistics            stat-agent                              exact arithmetic / exact binomial test
name -> structure     opsin, pubchem                          RDKit canonical-SMILES identity
functional groups     molecular-groups                        RDKit SMARTS matching
chem compute          openbabel, mordred                      RDKit recomputation
generative design     janus, tamgen                           constraint satisfaction on generated SMILES
retrosynthesis        aizynthfinder, retrochimera             precursor validity + known key disconnection
structure alignment   tmalign                                 self-alignment identity (TM-score 1, RMSD 0)
"""
from __future__ import annotations

import json
import math
import re
import statistics

from ..arena.stats import binom_two_sided
from ..model import Confusion, Kind, OracleTask, Reference

# ------------------------------------------------------------------ sequence
_BASES = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
_CODON = {a + b + c: _AA[16 * i + 4 * j + k] for i, a in enumerate(_BASES) for j, b in enumerate(_BASES) for k, c in enumerate(_BASES)}
DNA = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"


def translate(dna: str) -> str:
    prot = ""
    for i in range(0, len(dna) - 2, 3):
        aa = _CODON[dna[i:i + 3]]
        if aa == "*":
            break
        prot += aa
    return prot


def revcomp(dna: str) -> str:
    return dna[::-1].translate(str.maketrans("ACGT", "TGCA"))


def _sequence() -> list[OracleTask]:
    ag = ["biopython", "core-python-agent"]
    gc = 100 * (DNA.count("G") + DNA.count("C")) / len(DNA)
    src = "computed exactly (standard genetic code, NCBI table 1)"
    return [
        OracleTask("fam.seq.gc", "sequence", ag, f"What is the GC content (percent) of the DNA sequence {DNA} ?",
                   Reference(Kind.NUMERIC, gc, "", abs_tol=0.1, source=src), tags=["family", "core"]),
        OracleTask("fam.seq.revcomp", "sequence", ag, f"What is the reverse complement of the DNA sequence {DNA} ?",
                   Reference(Kind.EXACT, revcomp(DNA), source=src),
                   confusions=[Confusion(DNA[::-1], "reversed but not complemented"),
                               Confusion(DNA.translate(str.maketrans("ACGT", "TGCA")), "complemented but not reversed")],
                   tags=["family"]),
        OracleTask("fam.seq.translate", "sequence", ag,
                   f"Translate the DNA sequence {DNA} from the first base using the standard genetic code, stopping at "
                   f"the first stop codon. Give the protein sequence in one-letter codes.",
                   Reference(Kind.EXACT, translate(DNA), source=src), tags=["family", "core"]),
    ]


# ------------------------------------------------------------------ crystallography
_NA = 6.02214076e23


def _crystal() -> list[OracleTask]:
    ag = ["spglib", "pymatgen", "ase"]
    rho_si = 8 * 28.0855 / (_NA * (5.431e-8) ** 3)
    return [
        OracleTask("fam.xtal.nacl.spacegroup", "crystallography", ag,
                   "Rock-salt NaCl: face-centred cubic lattice a = 5.64 Å, Na at (0,0,0) and Cl at (0.5,0.5,0.5) in the "
                   "conventional cell. What is its space-group number?",
                   Reference(Kind.EXACT, 225, source="International Tables: Fm-3m (No. 225)"), tags=["family", "core"]),
        OracleTask("fam.xtal.si.spacegroup", "crystallography", ag,
                   "Diamond-cubic silicon, a = 5.431 Å (8 atoms in the conventional cell). What is its space-group number?",
                   Reference(Kind.EXACT, 227, source="International Tables: Fd-3m (No. 227)"),
                   confusions=[Confusion(225, "fcc Bravais lattice reported instead of the diamond structure")],
                   tags=["family", "core"]),
        OracleTask("fam.xtal.si.density", "crystallography", ["pymatgen", "ase"],
                   "Compute the density of diamond-cubic silicon with lattice constant a = 5.431 Å. Report it in g/cm^3.",
                   Reference(Kind.NUMERIC, rho_si, "g/cm^3", rel_tol=0.005, source="8 × 28.0855 u / (N_A a³)"),
                   tags=["family"]),
    ]


# ------------------------------------------------------------------ kinetics & metabolism
def _kinetics() -> list[OracleTask]:
    ag = ["cantera", "tellurium", "copasi-basico"]
    return [
        OracleTask("fam.kin.first_order", "kinetics", ag,
                   "Simulate the first-order reaction A -> B with k = 0.1 s^-1 and [A]0 = 1.0 mol/L. What is [A] at t = 10 s "
                   "(in mol/L)?",
                   Reference(Kind.NUMERIC, math.exp(-1), "", rel_tol=0.005, source="analytic: [A]0·e^(-kt)"),
                   confusions=[Confusion(1 - math.exp(-1), "reported [B] instead of [A]")], tags=["family", "core"]),
        OracleTask("fam.kin.equilibrium", "kinetics", ag,
                   "Simulate the reversible reaction A <-> B with kf = 2 s^-1, kr = 1 s^-1, [A]0 = 1.0 mol/L, [B]0 = 0, "
                   "until equilibrium. What is the equilibrium [A] in mol/L?",
                   Reference(Kind.NUMERIC, 1 / 3, "", rel_tol=0.01, source="analytic: K = kf/kr = 2 → [A]eq = 1/3"),
                   confusions=[Confusion(2 / 3, "reported [B]eq instead of [A]eq")], tags=["family"]),
        OracleTask("fam.fba.ecoli_core", "metabolic-modelling", ["cobrapy"],
                   "Using the E. coli core metabolic model (e_coli_core, BiGG) with its default aerobic glucose medium, "
                   "what is the maximal biomass (growth) rate from flux balance analysis, in 1/h?",
                   Reference(Kind.RANGE, unit="", lo=0.86, hi=0.89,
                             source="Orth, Fleming & Palsson, EcoSal Plus (2010); BiGG e_coli_core: 0.874 h^-1"),
                   tags=["family", "core"]),
    ]


# ------------------------------------------------------------------ statistics
_X = [1, 2, 3, 4, 5]
_Y = [2, 4, 5, 4, 5]


def _pearson(x, y) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def _stats() -> list[OracleTask]:
    ag = ["stat-agent"]
    return [
        OracleTask("fam.stat.pearson", "statistics", ag,
                   f"Compute the Pearson correlation coefficient between x = {_X} and y = {_Y}.",
                   Reference(Kind.NUMERIC, _pearson(_X, _Y), "", abs_tol=0.001, source="exact arithmetic"),
                   tags=["family", "core"]),
        OracleTask("fam.stat.binom", "statistics", ag,
                   "A coin lands heads 9 times in 10 flips. What is the exact two-sided binomial test p-value against p = 0.5?",
                   Reference(Kind.NUMERIC, binom_two_sided(9, 10), "", abs_tol=0.0005, source="exact binomial"),
                   confusions=[Confusion(binom_two_sided(9, 10) / 2, "one-sided p-value reported")], tags=["family"]),
        OracleTask("fam.stat.sd", "statistics", ag,
                   "What is the sample standard deviation (n-1 denominator) of [2, 4, 4, 4, 5, 5, 7, 9]?",
                   Reference(Kind.NUMERIC, statistics.stdev([2, 4, 4, 4, 5, 5, 7, 9]), "", abs_tol=0.001, source="exact"),
                   confusions=[Confusion(statistics.pstdev([2, 4, 4, 4, 5, 5, 7, 9]), "population SD (n denominator) reported")],
                   tags=["family"]),
    ]


# ------------------------------------------------------------------ chemistry families
NAMES = {
    "2-acetyloxybenzoic acid": "CC(=O)Oc1ccccc1C(=O)O",
    "N-(4-hydroxyphenyl)acetamide": "CC(=O)Nc1ccc(O)cc1",
    "1,3,7-trimethylpurine-2,6-dione": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "(2S)-2-(6-methoxynaphthalen-2-yl)propanoic acid": "C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1",
}
GROUPS = {"carboxylic acid": "[CX3](=O)[OX2H1]", "ester": "[#6][CX3](=O)[OX2][#6]", "amide": "[NX3][CX3](=O)[#6]",
          "phenol": "[OX2H][c]"}
GROUP_MOLS = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O", "paracetamol": "CC(=O)Nc1ccc(O)cc1"}


def _chem() -> list[OracleTask]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    out: list[OracleTask] = []
    for i, (name, smi) in enumerate(NAMES.items()):
        out.append(OracleTask(f"fam.name2struct.{i}", "name-to-structure", ["opsin", "pubchem"],
                              f"Convert the IUPAC name '{name}' to a SMILES string (include stereochemistry if specified).",
                              Reference(Kind.SMILES, smi, source="RDKit canonical identity"),
                              tags=["family"] + (["core"] if "(2S)" in name or i == 0 else [])))
    for mname, smi in GROUP_MOLS.items():
        m = Chem.MolFromSmiles(smi)
        for g, smarts in GROUPS.items():
            has = m.HasSubstructMatch(Chem.MolFromSmarts(smarts))
            out.append(OracleTask(f"fam.groups.{mname}.{g.replace(' ', '_')}", "functional-groups", ["molecular-groups"],
                                  f"Does the molecule with SMILES {smi} contain a {g} group? Answer true or false.",
                                  Reference(Kind.EXACT, ["true", "yes"] if has else ["false", "no"],
                                            source=f"RDKit SMARTS {smarts}"), tags=["family"]))
    for mname, smi in [("aspirin", "CC(=O)Oc1ccccc1C(=O)O"), ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")]:
        m = Chem.MolFromSmiles(smi)
        out.append(OracleTask(f"fam.chem.mw.{mname}", "cheminformatics", ["openbabel", "mordred"],
                              f"What is the average molecular weight of the molecule with SMILES {smi} ? Report it in g/mol.",
                              Reference(Kind.NUMERIC, Descriptors.MolWt(m), "g/mol", abs_tol=0.05, source="RDKit"),
                              confusions=[Confusion(Descriptors.ExactMolWt(m), "monoisotopic mass reported", "g/mol")],
                              tags=["family", "core"]))
        out.append(OracleTask(f"fam.chem.formula.{mname}", "cheminformatics", ["openbabel", "mordred"],
                              f"What is the molecular formula of the molecule with SMILES {smi} ?",
                              Reference(Kind.FORMULA, rdMolDescriptors.CalcMolFormula(m), source="RDKit"), tags=["family"]))
    return out


# ------------------------------------------------------------------ generative & retrosynthesis (custom checkers)
def _as_smiles_list(value, text: str) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed]
        except json.JSONDecodeError:
            pass
        return [t for t in re.split(r"[\s,;]+", value) if t]
    return []


def check_generation(value, text, n=5, lo=150.0, hi=300.0, smarts="[CX3](=O)[OX2H1]"):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    smis = _as_smiles_list(value, text)
    patt = Chem.MolFromSmarts(smarts)
    valid, ok, canon = 0, 0, set()
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        valid += 1
        c = Chem.MolToSmiles(m)
        if c in canon:
            continue
        canon.add(c)
        if lo <= Descriptors.MolWt(m) <= hi and m.HasSubstructMatch(patt):
            ok += 1
    score = min(ok, n) / n
    diag = f"{len(smis)} given, {valid} valid, {len(canon)} unique, {ok} satisfy MW {lo:g}-{hi:g} + carboxylic acid"
    return ok >= n - 1 and len(smis) >= n, score, diag


def check_retro_aspirin(value, text):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    smis = _as_smiles_list(value, text)
    mols = [Chem.MolFromSmiles(s) for s in smis]
    if not smis:
        return False, 0.0, "no precursors given"
    if any(m is None for m in mols):
        return False, 0.0, f"invalid precursor SMILES in {smis}"
    salicylic = Chem.MolToSmiles(Chem.MolFromSmiles("O=C(O)c1ccccc1O"))
    has_key = any(Chem.MolToSmiles(m) == salicylic for m in mols)
    heavy = sum(m.GetNumHeavyAtoms() for m in mols)
    balanced = heavy >= 13  # aspirin has 13 heavy atoms
    score = 0.5 * has_key + 0.5 * balanced
    diag = ("" if has_key else "salicylic acid (the standard disconnection) not among precursors; ") + \
           ("" if balanced else f"precursors hold only {heavy} heavy atoms (< 13 in aspirin)")
    return has_key and balanced, score, diag.strip("; ")


def _generative() -> list[OracleTask]:
    return [
        OracleTask("fam.gen.acids", "generative-design", ["janus", "tamgen"],
                   "Propose 5 distinct, chemically valid molecules, each with an average molecular weight between 150 and "
                   "300 g/mol and containing a carboxylic acid group. Give the value as a JSON list of SMILES strings.",
                   Reference(Kind.CUSTOM, "≥4 of 5 valid, unique, MW 150-300, contain COOH"), checker=check_generation,
                   tags=["family", "core"]),
        OracleTask("fam.retro.aspirin", "retrosynthesis", ["aizynthfinder", "retrochimera"],
                   "Propose a single-step retrosynthesis for aspirin (SMILES CC(=O)Oc1ccccc1C(=O)O). Give the value as a "
                   "JSON list of precursor SMILES.",
                   Reference(Kind.CUSTOM, "valid precursors incl. salicylic acid, atoms accounted"), checker=check_retro_aspirin,
                   tags=["family", "core"]),
    ]


# ------------------------------------------------------------------ structure alignment
def _alignment() -> list[OracleTask]:
    ag = ["tmalign"]
    return [
        OracleTask("fam.align.self_tm", "structure-alignment", ag,
                   "Align PDB 1UBQ chain A against itself with TM-align. What is the TM-score (normalised by chain A)?",
                   Reference(Kind.NUMERIC, 1.0, "", abs_tol=0.001, source="self-alignment identity"), tags=["family", "core"]),
        OracleTask("fam.align.self_rmsd", "structure-alignment", ag,
                   "Align PDB 1UBQ chain A against itself with TM-align. What is the RMSD of the aligned residues, in Å?",
                   Reference(Kind.NUMERIC, 0.0, "Å", abs_tol=0.01, source="self-alignment identity"), tags=["family"]),
    ]


def tasks() -> list[OracleTask]:
    return _sequence() + _crystal() + _kinetics() + _stats() + _chem() + _generative() + _alignment()


def probes() -> list:
    return []
