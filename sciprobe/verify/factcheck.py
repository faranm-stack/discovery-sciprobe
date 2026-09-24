"""Fact-check free-text agent answers against authoritative sources.

This is the "no answer key" path: for open-ended prompts (starter-kit sample
prompts, README usage examples, PR prompts) we can still verify every checkable
claim an agent makes:

* identifiers it cites exist (PubChem CID, ChEMBL, PDB, UniProt, NCT, PMID, DOI)
  and are on-topic for the question;
* SMILES it writes are chemically valid;
* molecular formulas / molecular weights it states for a molecule match what
  RDKit recomputes from that molecule's structure.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .. import net
from ..arena.text import content_tokens
from . import authorities as auth

_PATTERNS: dict[str, re.Pattern] = {
    "nct": re.compile(r"\b(NCT\d{8})\b"),
    "chembl": re.compile(r"\b(CHEMBL\d{1,9})\b", re.IGNORECASE),
    "pmid": re.compile(r"(?:\bPMID[:\s#]*|pubmed\.ncbi\.nlm\.nih\.gov/)(\d{5,9})\b", re.IGNORECASE),
    "doi": re.compile(r"\b(10\.\d{4,9}/[^\s\]\)\[,;\"'<>]+)"),
    "cid": re.compile(r"(?:\b(?:PubChem\s+)?CID[:\s#]*|pubchem\.ncbi\.nlm\.nih\.gov/compound/)(\d{1,10})\b", re.IGNORECASE),
    "pdb": re.compile(r"\bPDB(?:\s*(?:ID|entry|code))?[\s:#(]*([1-9][A-Za-z0-9]{3})\b|rcsb\.org/structure/([1-9][A-Za-z0-9]{3})", re.IGNORECASE),
    "uniprot": re.compile(r"(?:\bUniProt(?:KB)?(?:\s*(?:ID|accession))?[\s:#(]*|uniprot\.org/uniprotkb/)"
                          r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b", re.IGNORECASE),
}
_BACKTICK = re.compile(r"`([^`\s]{3,200})`")
_SMILES_LABEL = re.compile(r"SMILES[:\s]+([A-Za-z0-9@+\-\[\]\(\)=#$/\\%.]{3,200})")
_SMILES_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$/\\%.]+$")
_FORMULA = re.compile(r"(?<![A-Za-z0-9])(C\d{0,3}H\d{0,3}(?:(?:Br|Cl|[NOSPFIB])\d{0,3})*)(?![a-z0-9])")
_MW = re.compile(r"(\d{2,4}(?:\.\d{1,4})?)\s*(?:g\s*/\s*mol|g·mol|Da\b|daltons?\b)", re.IGNORECASE)

_STATUS_BAD = {"NOT_FOUND", "INVALID", "MISMATCH"}


@dataclass
class ClaimCheck:
    kind: str
    value: str
    status: str  # VERIFIED | NOT_FOUND | INVALID | MISMATCH | OFF_TOPIC | UNCHECKED
    detail: str = ""
    label: str | None = None
    url: str | None = None


@dataclass
class FactCheckReport:
    checks: list[ClaimCheck] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    @property
    def n_bad(self) -> int:
        return sum(c.status in _STATUS_BAD for c in self.checks)

    @property
    def score(self) -> float | None:
        """Fraction of checkable claims that held up (None if nothing checkable)."""
        judged = [c for c in self.checks if c.status != "UNCHECKED"]
        return None if not judged else sum(c.status in ("VERIFIED", "OFF_TOPIC") for c in judged) / len(judged)

    def to_dict(self) -> dict[str, Any]:
        return {"counts": self.counts, "score": self.score, "checks": [asdict(c) for c in self.checks]}


# ------------------------------------------------------------------ extraction

def extract_ids(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for kind, pat in _PATTERNS.items():
        vals = []
        for m in pat.finditer(text or ""):
            v = next(g for g in m.groups() if g)
            v = v.rstrip(".") if kind == "doi" else v.upper() if kind in ("nct", "chembl", "pdb", "uniprot") else v
            if v not in vals:
                vals.append(v)
        if vals:
            out[kind] = vals
    return out


def _looks_like_smiles(tok: str) -> bool:
    if not _SMILES_CHARS.match(tok) or tok.isdigit() or tok.isalpha() and tok.upper() == tok and len(tok) < 4:
        return False
    # Hill formulas (C9H8O4) use SMILES characters but bare H with counts never appears in SMILES
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", tok) and re.search(r"H\d*(?:[A-Z]|$)", tok) and re.search(r"\d", tok):
        return False
    return bool(re.search(r"[=#()\[\]@]|[cnos]\d|\d", tok)) and bool(re.search(r"[CNOScnos]", tok))


def extract_smiles(text: str) -> list[str]:
    cands = [m.group(1) for m in _BACKTICK.finditer(text or "")] + [m.group(1) for m in _SMILES_LABEL.finditer(text or "")]
    out = []
    for c in cands:
        c = c.rstrip(".,;")
        if _looks_like_smiles(c) and not _PATTERNS["nct"].fullmatch(c) and c not in out:
            out.append(c)
    return out


# ------------------------------------------------------------------ verification

def _relevant(label: str | None, prompt: str | None) -> bool:
    if not label or not prompt:
        return True
    return bool(set(content_tokens(label)) & set(content_tokens(prompt)))


def _lookup(kind: str) -> Callable[[str], dict[str, Any]]:
    return {"nct": auth.ctgov_study, "chembl": auth.chembl_molecule, "doi": auth.crossref_work,
            "cid": auth.pubchem_by_cid, "pdb": auth.rcsb_entry, "uniprot": auth.uniprot_entry}[kind]


def _label(kind: str, rec: dict[str, Any]) -> str | None:
    if kind == "uniprot":
        return " ".join(x for x in (rec.get("name"), rec.get("gene"), rec.get("organism")) if x)
    return rec.get("title") or rec.get("name")


def _check_id(kind: str, value: str, prompt: str | None, topic_kinds: set[str]) -> ClaimCheck:
    try:
        rec = _lookup(kind)(value)
    except net.NotFound:
        return ClaimCheck(kind, value, "NOT_FOUND", "identifier does not exist at the authority")
    except Exception as e:  # noqa: BLE001
        return ClaimCheck(kind, value, "UNCHECKED", f"lookup failed: {e}"[:200])
    label = _label(kind, rec)
    status = "VERIFIED" if kind not in topic_kinds or _relevant(label, prompt) else "OFF_TOPIC"
    return ClaimCheck(kind, value, status, "" if status == "VERIFIED" else "exists but shares no terms with the question",
                      label, rec.get("url"))


def _chem_checks(text: str, prompt: str | None, mw_tol: float) -> list[ClaimCheck]:
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        return []
    RDLogger.DisableLog("rdApp.*")
    from ..scoring import parse_formula

    checks: list[ClaimCheck] = []
    mols = {}
    for s in extract_smiles(text):
        m = Chem.MolFromSmiles(s)
        if m is None:
            if prompt and s in prompt:
                continue  # the agent is quoting the user's (possibly deliberately invalid) input, not claiming it
            checks.append(ClaimCheck("smiles", s, "INVALID", "RDKit cannot parse this SMILES"))
        else:
            mols[Chem.MolToSmiles(m)] = m
    subject = None
    if len(mols) == 1:
        subject = next(iter(mols.values()))
    elif not mols and prompt:
        prompt_mols = {Chem.MolToSmiles(m): m for s in extract_smiles(prompt) + _SMILES_LABEL.findall(prompt)
                       if (m := Chem.MolFromSmiles(s.rstrip(".,;?"))) is not None}
        if len(prompt_mols) == 1:
            subject = next(iter(prompt_mols.values()))
    if subject is None:
        return checks
    true_formula = rdMolDescriptors.CalcMolFormula(subject)
    for f in dict.fromkeys(_FORMULA.findall(text)):
        try:
            ok = parse_formula(f) == parse_formula(true_formula)
        except ValueError:
            continue
        checks.append(ClaimCheck("formula", f, "VERIFIED" if ok else "MISMATCH",
                                 "" if ok else f"RDKit formula for the molecule is {true_formula}", true_formula))
    avg, mono = Descriptors.MolWt(subject), Descriptors.ExactMolWt(subject)
    for v in dict.fromkeys(float(x) for x in _MW.findall(text)):
        if abs(v - avg) <= mw_tol or abs(v - mono) <= mw_tol:
            which = "average MW" if abs(v - avg) <= mw_tol else "monoisotopic mass"
            checks.append(ClaimCheck("mw", f"{v:g}", "VERIFIED", f"matches {which}", f"{avg:.3f} / {mono:.4f}"))
        else:
            checks.append(ClaimCheck("mw", f"{v:g}", "MISMATCH",
                                     f"RDKit: average {avg:.3f}, monoisotopic {mono:.4f}", f"{avg:.3f}"))
    return checks


def fact_check(text: str, prompt: str | None = None, max_per_kind: int = 25, mw_tol: float = 0.1,
               topic_kinds: tuple[str, ...] = ("nct", "pmid", "doi"), workers: int = 4) -> FactCheckReport:
    """Verify every checkable claim in ``text``. ``prompt`` enables on-topic checks
    for literature/trial identifiers."""
    ids = extract_ids(text)
    jobs = [(k, v) for k, vs in ids.items() if k != "pmid" for v in vs[:max_per_kind]]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        checks = list(pool.map(lambda kv: _check_id(kv[0], kv[1], prompt, set(topic_kinds)), jobs))
    pmids = ids.get("pmid", [])[:max_per_kind]
    if pmids:
        try:
            found = auth.pubmed_summaries(pmids)
            for p in pmids:
                if p not in found:
                    checks.append(ClaimCheck("pmid", p, "NOT_FOUND", "PMID does not exist"))
                else:
                    rel = "pmid" not in topic_kinds or _relevant(found[p]["title"], prompt)
                    checks.append(ClaimCheck("pmid", p, "VERIFIED" if rel else "OFF_TOPIC",
                                             "" if rel else "exists but shares no terms with the question",
                                             found[p]["title"], found[p]["url"]))
        except Exception as e:  # noqa: BLE001
            checks += [ClaimCheck("pmid", p, "UNCHECKED", f"lookup failed: {e}"[:200]) for p in pmids]
    checks += _chem_checks(text, prompt, mw_tol)
    return FactCheckReport(checks)
