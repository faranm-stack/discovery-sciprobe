"""Engineering physics oracles: SPICE circuits (analytic) and CFD (Ghia 1982)."""
from __future__ import annotations

import math

from ..invariance import InvarianceProbe, Relation
from ..model import Confusion, Kind, OracleTask, Reference

SPICE = ["xyce"]
CFD = ["openfoam"]

_RC_STEP = ("Using a SPICE transient simulation: an ideal 5 V step source (0 V before t = 0, 1 ns rise time) drives "
            "R = 1 kΩ in series with C = 1 µF to ground; the capacitor starts uncharged. ")
_GHIA = "Ghia, Ghia & Shin, J. Comput. Phys. 48, 387-411 (1982), Table I, Re = 100"
_CAVITY = ("Run a 2-D steady incompressible lid-driven cavity simulation at Reynolds number 100: unit square "
           "cavity, lid velocity U = 1 m/s, kinematic viscosity 0.01 m^2/s, uniform mesh of at least 64x64 cells. ")


def tasks() -> list[OracleTask]:
    v_tau = 5 * (1 - math.exp(-1))
    v_5tau = 5 * (1 - math.exp(-5))
    f0 = 1 / (2 * math.pi * math.sqrt(10e-3 * 100e-9))
    return [
        OracleTask(
            id="ckt.rc.v_at_tau", domain="circuits", agents=SPICE, tags=["core"],
            question=_RC_STEP + "What is the capacitor voltage at t = 1 ms? Report it in V.",
            reference=Reference(Kind.NUMERIC, v_tau, "V", rel_tol=0.01, source="Analytic: V(1-e^{-t/RC}), RC = 1 ms"),
            confusions=[Confusion(5 * math.exp(-1), "resistor voltage (or discharge curve) reported instead of capacitor voltage", "V")],
        ),
        OracleTask(
            id="ckt.rc.v_at_5tau", domain="circuits", agents=SPICE,
            question=_RC_STEP + "What is the capacitor voltage at t = 5 ms? Report it in V.",
            reference=Reference(Kind.NUMERIC, v_5tau, "V", rel_tol=0.005, source="Analytic: V(1-e^{-5})"),
        ),
        OracleTask(
            id="ckt.rlc.resonance", domain="circuits", agents=SPICE, tags=["core"],
            question="A series RLC circuit has R = 10 Ω, L = 10 mH and C = 100 nF. Using an AC sweep, find the resonant "
                     "frequency (peak current). Report it in Hz.",
            reference=Reference(Kind.NUMERIC, f0, "Hz", rel_tol=0.01, source="Analytic: f0 = 1/(2π√(LC))"),
            confusions=[Confusion(2 * math.pi * f0, "angular frequency (rad/s) reported as Hz", "Hz")],
        ),
        OracleTask(
            id="ckt.divider.dc", domain="circuits", agents=SPICE,
            question="DC operating point: a 12 V source feeds R1 = 3 kΩ in series with R2 = 1 kΩ to ground. "
                     "What is the voltage across R2? Report it in V.",
            reference=Reference(Kind.NUMERIC, 3.0, "V", abs_tol=0.01, source="Analytic voltage divider"),
            confusions=[Confusion(9.0, "voltage across R1 reported instead of R2", "V")],
        ),
        OracleTask(
            id="cfd.cavity_re100.u_center", domain="cfd", agents=CFD, tags=["expensive"],
            question=_CAVITY + "Report the normalised horizontal velocity u/U at the cavity centre (x = 0.5, y = 0.5).",
            reference=Reference(Kind.NUMERIC, -0.20581, "", abs_tol=0.015, source=_GHIA),
        ),
        OracleTask(
            id="cfd.cavity_re100.u_min", domain="cfd", agents=CFD, tags=["expensive"],
            question=_CAVITY + "Along the vertical centreline x = 0.5, report the minimum normalised horizontal velocity u/U.",
            reference=Reference(Kind.NUMERIC, -0.21090, "", abs_tol=0.015, source=_GHIA + " (at y = 0.4531)"),
        ),
    ]


def probes() -> list[InvarianceProbe]:
    return [
        InvarianceProbe("ckt.inv.rc.paraphrase", "circuits", SPICE, Relation.PARAPHRASE, "", "", "V", rel_tol=0.01,
                        extras={"templates": [
                            _RC_STEP + "What is the capacitor voltage at t = 1 ms? Report it in V.",
                            "Simulate in SPICE: 1 µF capacitor (initially 0 V) charged through a 1000 Ω resistor from an "
                            "ideal 0-to-5 V step at t = 0 (1 ns rise). Give the capacitor voltage 1000 µs after the step, in V.",
                            "RC charging circuit, R = 1e3 ohm, C = 1e-6 F, 5 V step input at t = 0, capacitor uncharged. "
                            "Run a transient analysis and report V(C) at t = 0.001 s in volts.",
                        ]}, tags=["core"]),
    ]
