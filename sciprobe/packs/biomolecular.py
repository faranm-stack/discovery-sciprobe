"""Biomolecular oracles: structure databases, structure prediction, MD, compound databases."""
from __future__ import annotations

from ..invariance import InvarianceProbe, Relation
from ..model import Confusion, Kind, OracleTask, Reference

PDB = ["pdb-search", "pdb-insights"]
FOLD = ["alphafold", "boltztwo"]
MD = ["gromacs", "openmm"]

UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def tasks() -> list[OracleTask]:
    return [
        OracleTask(
            id="bio.pdb.1ubq.resolution", domain="structural-biology", agents=PDB, tags=["core"],
            question="What is the X-ray diffraction resolution of PDB entry 1UBQ? Report it in Å.",
            reference=Reference(Kind.NUMERIC, 1.8, "Å", abs_tol=0.01,
                                source="RCSB PDB 1UBQ (Vijay-Kumar, Bugg & Cook, J. Mol. Biol. 194, 531, 1987)"),
        ),
        OracleTask(
            id="bio.pdb.1ubq.residues", domain="structural-biology", agents=PDB, tags=["core"],
            question="How many amino-acid residues are in the deposited polymer sequence of PDB entry 1UBQ chain A?",
            reference=Reference(Kind.EXACT, 76, source="RCSB PDB 1UBQ"),
        ),
        OracleTask(
            id="bio.fold.ubiquitin.plddt", domain="structure-prediction", agents=FOLD, tags=["expensive"],
            question=f"Predict the structure of this protein and report the mean pLDDT over all residues:\n{UBIQUITIN}",
            reference=Reference(Kind.RANGE, unit="", lo=80.0, hi=100.0,
                                source="Ubiquitin is a compact, well-ordered domain; AF2-class models give mean pLDDT > 80"),
            confusions=[Confusion(0.9, "pLDDT reported on a 0-1 scale instead of 0-100")],
        ),
        OracleTask(
            id="md.water.tip3p.density", domain="molecular-dynamics", agents=MD, tags=["expensive"],
            question="Run an NPT molecular dynamics simulation of a box of at least 500 TIP3P water molecules at 298 K "
                     "and 1 bar (PME electrostatics, >= 1 ns production after equilibration). Report the average density in g/cm^3.",
            reference=Reference(Kind.RANGE, unit="g/cm^3", lo=0.96, hi=1.02,
                                source="TIP3P literature 0.98-1.00 g/cm^3 (Jorgensen 1983; Mark & Nilsson 2001); experiment 0.997"),
        ),
        OracleTask(
            id="db.pubchem.aspirin.cid", domain="compound-databases", agents=["pubchem"], tags=["core"],
            question="What is the PubChem CID of acetylsalicylic acid (aspirin)?",
            reference=Reference(Kind.EXACT, 2244, source="PubChem"),
            confusions=[Confusion(338, "salicylic acid CID returned instead of aspirin")],
        ),
        OracleTask(
            id="db.pubchem.caffeine.cid", domain="compound-databases", agents=["pubchem"],
            question="What is the PubChem CID of caffeine?",
            reference=Reference(Kind.EXACT, 2519, source="PubChem"),
        ),
        OracleTask(
            id="db.chembl.aspirin.id", domain="compound-databases", agents=["chembl"], tags=["core"],
            question="What is the ChEMBL molecule ID of aspirin?",
            reference=Reference(Kind.EXACT, "CHEMBL25", source="ChEMBL"),
        ),
    ]


def probes() -> list[InvarianceProbe]:
    return [
        InvarianceProbe("bio.inv.pdb.resolution.repeat", "structural-biology", ["pdb-search"], Relation.REPEAT,
                        "What is the X-ray diffraction resolution of PDB entry {input}? Report it in Å.", "1UBQ", "Å",
                        abs_tol=0.001, n_variants=3, notes="A database lookup must be deterministic."),
    ]
