"""Task-pack registry."""
from __future__ import annotations

from importlib import import_module

from ..invariance import InvarianceProbe
from ..model import OracleTask

PACKS = ["cheminformatics", "quantum", "engineering", "biomolecular", "models", "controls", "families"]
OPTIONAL_PACKS = ["live"]  # network: answer keys fetched from PubChem/RCSB/UniProt/ChEMBL/ClinicalTrials.gov


def load(packs: list[str] | None = None, live_n: int = 3, seed: int = 0) -> tuple[list[OracleTask], list[InvarianceProbe]]:
    tasks: list[OracleTask] = []
    probes: list[InvarianceProbe] = []
    for name in packs or PACKS:
        if name == "live":
            from ..sourcing import live

            tasks.extend(live.tasks(live_n, seed))
            continue
        if name not in PACKS:
            raise KeyError(f"unknown pack {name!r}; choose from {PACKS + OPTIONAL_PACKS}")
        mod = import_module(f"{__name__}.{name}")
        tasks.extend(mod.tasks())
        probes.extend(mod.probes())
    return tasks, probes
