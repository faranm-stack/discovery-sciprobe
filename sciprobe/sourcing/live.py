"""Live oracles: fresh, verifiable questions whose answer keys are fetched from
authoritative sources at build time (PubChem, RCSB PDB, UniProt, ChEMBL,
ClinicalTrials.gov). Sampling is seeded, so a run is reproducible, but the set
can be re-drawn at will, so agents cannot be tuned to a fixed benchmark.

Pools only need to contain *valid identifiers*: every value that is graded is
fetched live and cited (URL + retrieval date); IDs that fail lookup are skipped.
"""
from __future__ import annotations

import random
from datetime import date
from typing import Callable

from .. import net
from ..model import Kind, OracleTask, Reference
from ..verify import authorities as auth

DRUGS = ["aspirin", "ibuprofen", "caffeine", "acetaminophen", "imatinib", "metformin", "atorvastatin", "sildenafil",
         "warfarin", "omeprazole", "diazepam", "osimertinib", "gefitinib", "erlotinib", "dasatinib", "nilotinib",
         "lapatinib", "sorafenib", "ruxolitinib", "tofacitinib"]
PDB_IDS = ["1UBQ", "1HSG", "4HHB", "1CRN", "2LYZ", "3PTB", "1IEP", "6LU7", "1M17", "2HYY", "3ERT", "1STP", "2RH1", "5KIR"]
UNIPROT = ["P61626", "P69905", "P00533", "P04637", "P38398", "P01308", "P00519", "P29597", "P24941", "P06239"]
CHEMBL_IDS = ["CHEMBL25", "CHEMBL941", "CHEMBL1421", "CHEMBL939", "CHEMBL553", "CHEMBL1789941", "CHEMBL221959",
              "CHEMBL1642", "CHEMBL192", "CHEMBL1431"]
TRIAL_CONDITIONS = ["kinase inhibitor", "acute myeloid leukemia", "non-small cell lung cancer"]


def _src(name: str, url: str) -> str:
    return f"{name} live lookup {date.today().isoformat()}: {url}"


def _safe(items, fn: Callable, n: int, rng: random.Random):
    pool = list(items)
    rng.shuffle(pool)
    out = []
    for x in pool:
        if len(out) >= n:
            break
        try:
            out.append((x, fn(x)))
        except net.NotFound:
            continue
    return out


def pubchem_tasks(n: int, rng: random.Random) -> list[OracleTask]:
    tasks = []
    for name, rec in _safe(DRUGS, auth.pubchem_by_name, n, rng):
        tasks.append(OracleTask(f"live.pubchem.cid.{name}", "compound-databases", ["pubchem"],
                                f"What is the PubChem CID of {name}?",
                                Reference(Kind.EXACT, rec["cid"], source=_src("PubChem", rec["url"])), tags=["live"]))
        tasks.append(OracleTask(f"live.pubchem.mw.{name}", "compound-databases", ["pubchem", "mol-toolkit"],
                                f"What is the molecular weight of {name} (average, standard atomic weights)? Report it in g/mol.",
                                Reference(Kind.NUMERIC, rec["mw"], "g/mol", abs_tol=0.06, source=_src("PubChem", rec["url"])),
                                tags=["live"]))
    return tasks


def rcsb_tasks(n: int, rng: random.Random) -> list[OracleTask]:
    tasks = []
    for pid, rec in _safe(PDB_IDS, auth.rcsb_entry, n, rng):
        if rec["resolution"]:
            tasks.append(OracleTask(f"live.pdb.resolution.{pid}", "structural-biology", ["pdb-search", "pdb-insights"],
                                    f"What is the experimental resolution of PDB entry {pid}? Report it in Å.",
                                    Reference(Kind.NUMERIC, float(rec["resolution"]), "Å", abs_tol=0.01,
                                              source=_src("RCSB PDB", rec["url"])), tags=["live"]))
        if rec["deposited_residues"]:
            tasks.append(OracleTask(f"live.pdb.residues.{pid}", "structural-biology", ["pdb-insights", "pdb-search"],
                                    f"How many polymer residues (monomers) are deposited in PDB entry {pid}, summed over all chains?",
                                    Reference(Kind.EXACT, int(rec["deposited_residues"]), source=_src("RCSB PDB", rec["url"])),
                                    tags=["live"]))
    return tasks


def uniprot_tasks(n: int, rng: random.Random) -> list[OracleTask]:
    return [
        OracleTask(f"live.uniprot.length.{acc}", "proteins", ["online-researcher", "esm-embed", "alphafold"],
                   f"How many amino acids long is the canonical sequence of UniProt entry {acc}?",
                   Reference(Kind.EXACT, int(rec["length"]), source=_src("UniProt", rec["url"])), tags=["live"])
        for acc, rec in _safe(UNIPROT, auth.uniprot_entry, n, rng) if rec.get("length")
    ]


def chembl_tasks(n: int, rng: random.Random) -> list[OracleTask]:
    tasks = []
    for cid, rec in _safe(CHEMBL_IDS, auth.chembl_molecule, n, rng):
        if rec["max_phase"] is not None:
            tasks.append(OracleTask(f"live.chembl.phase.{cid}", "compound-databases", ["chembl"],
                                    f"What is the maximum clinical development phase recorded in ChEMBL for {cid}?",
                                    Reference(Kind.EXACT, int(rec["max_phase"]), source=_src("ChEMBL", rec["url"])),
                                    tags=["live"]))
        if rec["mw"]:
            tasks.append(OracleTask(f"live.chembl.mw.{cid}", "compound-databases", ["chembl"],
                                    f"What is the full molecular weight of ChEMBL compound {cid}? Report it in g/mol.",
                                    Reference(Kind.NUMERIC, rec["mw"], "g/mol", abs_tol=0.06, source=_src("ChEMBL", rec["url"])),
                                    tags=["live"]))
    return tasks


def _status_variants(s: str) -> list[str]:
    s = s or ""
    words = s.replace("_", " ").lower()
    return list(dict.fromkeys([s, words, words.replace(" not ", ", not "), words.title()]))


def ctgov_tasks(n: int, rng: random.Random) -> list[OracleTask]:
    cond = rng.choice(TRIAL_CONDITIONS)
    studies = auth.ctgov_search(cond, n=max(10, 3 * n))
    rng.shuffle(studies)
    return [
        OracleTask(f"live.ctgov.status.{s['id']}", "clinical-trials", ["clinical-trials"],
                   f"What is the current overall recruitment status of clinical trial {s['id']} on ClinicalTrials.gov?",
                   Reference(Kind.EXACT, _status_variants(s["status"]),
                             source=_src("ClinicalTrials.gov", f"https://clinicaltrials.gov/study/{s['id']}")),
                   tags=["live"])
        for s in studies[:n] if s.get("status")
    ]


GENERATORS: dict[str, Callable[[int, random.Random], list[OracleTask]]] = {
    "pubchem": pubchem_tasks, "rcsb": rcsb_tasks, "uniprot": uniprot_tasks, "chembl": chembl_tasks, "ctgov": ctgov_tasks,
}


def tasks(n: int = 3, seed: int = 0, sources: list[str] | None = None) -> list[OracleTask]:
    out: list[OracleTask] = []
    for name in sources or list(GENERATORS):
        rng = random.Random(f"{seed}:{name}")
        try:
            out.extend(GENERATORS[name](n, rng))
        except Exception as e:  # noqa: BLE001 - one unreachable authority must not sink the others
            import sys

            print(f"live oracles: {name} skipped ({e})", file=sys.stderr)
    return out


def probes() -> list:
    return []
