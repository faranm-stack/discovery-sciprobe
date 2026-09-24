"""Quantum chemistry oracles (Psi4-class agents) with literature references."""
from __future__ import annotations

from ..invariance import InvarianceProbe, Relation
from ..model import Confusion, Kind, OracleTask, Reference

AGENTS = ["psi4", "pyscf"]

# O; H 1 0.96; H 1 0.96 2 104.5  -> Cartesian (angstrom)
WATER_XYZ = """O 0.000000 0.000000 0.000000
H 0.960000 0.000000 0.000000
H -0.240364 0.929421 0.000000"""


def tasks() -> list[OracleTask]:
    return [
        OracleTask(
            id="qc.h2.hf_sto3g.energy", domain="quantum-chemistry", agents=AGENTS, tags=["core"],
            question="Compute the RHF/STO-3G total energy of the H2 molecule at a bond length of exactly 1.4 bohr "
                     "(0.740848 Å). Report the total energy in hartree.",
            reference=Reference(Kind.NUMERIC, -1.11676, "hartree", abs_tol=3e-4,
                                source="Szabo & Ostlund, Modern Quantum Chemistry (1996), §3.5.2: E0 = -1.1167 Eh"),
            confusions=[Confusion(-1.8310, "electronic energy reported without nuclear repulsion", "hartree")],
        ),
        OracleTask(
            id="qc.h2.fci_sto3g.energy", domain="quantum-chemistry", agents=AGENTS,
            question="Compute the full-CI (FCI) / STO-3G total energy of H2 at a bond length of 0.7414 Å. "
                     "Report the total energy in hartree.",
            reference=Reference(Kind.NUMERIC, -1.13727, "hartree", abs_tol=3e-4,
                                source="Exact diagonalisation in STO-3G; e.g. OpenFermion H2_sto-3g_singlet_0.7414 (FCI -1.137270)"),
            confusions=[Confusion(-1.11668, "Hartree-Fock energy reported instead of FCI (correlation missing)", "hartree")],
        ),
        OracleTask(
            id="qc.h2.hf_sto3g.bond_length", domain="quantum-chemistry", agents=AGENTS,
            question="Optimise the geometry of H2 at the RHF/STO-3G level. What is the equilibrium H-H bond length in Å?",
            reference=Reference(Kind.NUMERIC, 0.712, "Å", abs_tol=0.004,
                                source="Szabo & Ostlund (1996): RHF/STO-3G R_e = 1.346 bohr = 0.712 Å"),
            confusions=[Confusion(0.7414, "experimental bond length quoted instead of computed RHF/STO-3G value", "Å")],
        ),
        OracleTask(
            id="qc.h2o.scf_ccpvdz.energy", domain="quantum-chemistry", agents=AGENTS, tags=["core"],
            question="Compute the RHF/cc-pVDZ total energy of water with the Z-matrix geometry "
                     "'O; H 1 0.96; H 1 0.96 2 104.5' (distances in Å, angle in degrees). Report the total energy in hartree.",
            reference=Reference(Kind.NUMERIC, -76.02663, "hartree", abs_tol=5e-4,
                                source="Psi4 manual quick-start / test scf1: -76.0266327341 Eh"),
        ),
    ]


def probes() -> list[InvarianceProbe]:
    tpl = ("Compute the RHF/cc-pVDZ total energy (hartree) of the water molecule with these Cartesian "
           "coordinates in Å (charge 0, singlet):\n{input}")
    return [
        InvarianceProbe("qc.inv.h2o.rigid_motion", "quantum-chemistry", AGENTS, Relation.RIGID_MOTION, tpl,
                        WATER_XYZ, "hartree", abs_tol=2e-5, n_variants=3, tags=["core"],
                        notes="Energy is invariant to rotation and translation."),
        InvarianceProbe("qc.inv.h2o.atom_permutation", "quantum-chemistry", AGENTS, Relation.ATOM_PERMUTATION, tpl,
                        WATER_XYZ, "hartree", abs_tol=2e-5, n_variants=3,
                        notes="Energy is invariant to the order atoms are listed in."),
        InvarianceProbe("qc.inv.h2.unit_reframe", "quantum-chemistry", AGENTS, Relation.UNIT_REFRAME,
                        "Compute the RHF/STO-3G total energy of H2 at a bond length of {input}. Report it in {unit}.",
                        "0.740848 Å", "hartree", abs_tol=2e-4,
                        extras={"units": ["hartree", "eV", "kcal/mol", "kJ/mol"]},
                        notes="Same physical quantity requested in four units must agree after conversion."),
    ]
