"""Trained-model predictors: climate GWP oracles + invariance for ML property models.

ML predictors often have no single "true" answer, but they MUST be invariant to
how an identical molecule is written. That is checkable without ground truth.
"""
from __future__ import annotations

from ..invariance import InvarianceProbe, Relation
from ..model import Confusion, Kind, OracleTask, Reference

HFC_134A = "FCC(F)(F)F"
HFO_1234YF = "C=C(F)C(F)(F)F"
S_NAPROXEN = "C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def tasks() -> list[OracleTask]:
    return [
        OracleTask(
            id="ml.gwp.hfc134a", domain="climate", agents=["gwp-predictor"], tags=["core"],
            question=f"Predict the 100-year global warming potential (GWP100, CO2 = 1) of the molecule with SMILES {HFC_134A} (HFC-134a).",
            reference=Reference(Kind.RANGE, unit="", lo=1100, hi=1800,
                                source="IPCC AR4 1430, AR5 1300, AR6 1530"),
        ),
        OracleTask(
            id="ml.gwp.hfo1234yf", domain="climate", agents=["gwp-predictor"], tags=["core"],
            question=f"Predict the 100-year global warming potential (GWP100, CO2 = 1) of the molecule with SMILES {HFO_1234YF} (HFO-1234yf).",
            reference=Reference(Kind.RANGE, unit="", lo=0, hi=10, source="IPCC AR5 <1, AR6 0.501"),
            confusions=[Confusion(1430, "returned HFC-134a's GWP (the refrigerant it replaces)")],
        ),
    ]


def probes() -> list[InvarianceProbe]:
    gwp_q = "Predict the 100-year global warming potential (GWP100, CO2 = 1) of the molecule with SMILES {input}."
    tox_q = ("Using your toxicity model's primary endpoint, what is the predicted probability (0-1) that the "
             "molecule with SMILES {input} is toxic?")
    sol_q = "Using your trained property model, predict the aqueous solubility (logS, log10 mol/L) of the molecule with SMILES {input}."
    return [
        InvarianceProbe("ml.inv.gwp.smiles.hfc134a", "climate", ["gwp-predictor"], Relation.SMILES_RANDOMIZATION,
                        gwp_q, HFC_134A, "", rel_tol=0.01, n_variants=3, tags=["core"]),
        InvarianceProbe("ml.inv.tox.smiles.caffeine", "toxicology", ["toxpred"], Relation.SMILES_RANDOMIZATION,
                        tox_q, CAFFEINE, "", abs_tol=0.02, n_variants=4, tags=["core"]),
        InvarianceProbe("ml.inv.tox.enantiomer.naproxen", "toxicology", ["toxpred"], Relation.ENANTIOMER,
                        tox_q, S_NAPROXEN, "", abs_tol=0.02,
                        notes="2-D descriptor/graph models cannot distinguish enantiomers; answers should match."),
        InvarianceProbe("ml.inv.sol.smiles.caffeine", "molecular-properties", ["chemprop", "chemberta"],
                        Relation.SMILES_RANDOMIZATION, sol_q, CAFFEINE, "", abs_tol=0.05, n_variants=4, tags=["core"]),
        InvarianceProbe("ml.inv.sol.repeat.caffeine", "molecular-properties", ["chemprop", "chemberta"], Relation.REPEAT,
                        sol_q, CAFFEINE, "", abs_tol=0.01, n_variants=3, notes="Trained models should be deterministic."),
    ]
