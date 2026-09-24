"""Clients for authoritative public sources. Each returns a small normalised
dict and raises ``net.NotFound`` when the identifier does not exist."""
from __future__ import annotations

import urllib.parse
from typing import Any

from .. import net

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
RCSB = "https://data.rcsb.org/rest/v1/core/entry"
UNIPROT = "https://rest.uniprot.org/uniprotkb"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CROSSREF = "https://api.crossref.org/works"

_PC_PROPS = "MolecularWeight,MolecularFormula,IsomericSMILES,InChIKey,IUPACName"


def _pc_props(r: dict[str, Any]) -> dict[str, Any]:
    return {"cid": int(r["CID"]), "mw": float(r["MolecularWeight"]), "formula": r.get("MolecularFormula"),
            "smiles": r.get("IsomericSMILES") or r.get("SMILES"), "inchikey": r.get("InChIKey"),
            "name": r.get("IUPACName"), "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{r['CID']}"}


def pubchem_by_name(name: str) -> dict[str, Any]:
    d = net.get_json(f"{PUBCHEM}/compound/name/{urllib.parse.quote(name)}/property/{_PC_PROPS}/JSON")
    return _pc_props(d["PropertyTable"]["Properties"][0])


def pubchem_by_cid(cid: int | str) -> dict[str, Any]:
    d = net.get_json(f"{PUBCHEM}/compound/cid/{int(cid)}/property/{_PC_PROPS}/JSON")
    return _pc_props(d["PropertyTable"]["Properties"][0])


def rcsb_entry(pdb_id: str) -> dict[str, Any]:
    e = net.get_json(f"{RCSB}/{pdb_id.upper()}")
    info = e.get("rcsb_entry_info") or {}
    res = info.get("resolution_combined") or [None]
    return {"id": pdb_id.upper(), "title": (e.get("struct") or {}).get("title"),
            "method": ((e.get("exptl") or [{}])[0]).get("method"), "resolution": res[0],
            "deposited_residues": info.get("deposited_polymer_monomer_count"),
            "polymer_entities": info.get("polymer_entity_count"),
            "url": f"https://www.rcsb.org/structure/{pdb_id.upper()}"}


def uniprot_entry(acc: str) -> dict[str, Any]:
    u = net.get_json(f"{UNIPROT}/{acc}.json", {"fields": "length,gene_primary,organism_name,protein_name"})
    genes = u.get("genes") or [{}]
    desc = (u.get("proteinDescription") or {}).get("recommendedName") or {}
    return {"accession": u.get("primaryAccession", acc), "length": (u.get("sequence") or {}).get("length"),
            "gene": ((genes[0].get("geneName") or {}).get("value")),
            "organism": (u.get("organism") or {}).get("scientificName"),
            "name": (desc.get("fullName") or {}).get("value"),
            "url": f"https://www.uniprot.org/uniprotkb/{acc}"}


def chembl_molecule(chembl_id: str) -> dict[str, Any]:
    c = net.get_json(f"{CHEMBL}/molecule/{chembl_id.upper()}.json")
    props = c.get("molecule_properties") or {}
    phase = c.get("max_phase")
    return {"id": c.get("molecule_chembl_id", chembl_id.upper()), "name": c.get("pref_name"),
            "max_phase": float(phase) if phase not in (None, "") else None,
            "mw": float(props["full_mwt"]) if props.get("full_mwt") else None, "formula": props.get("full_molformula"),
            "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id.upper()}/"}


def ctgov_study(nct: str) -> dict[str, Any]:
    s = net.get_json(f"{CTGOV}/{nct.upper()}")
    p = s.get("protocolSection") or {}
    return {"id": nct.upper(), "title": (p.get("identificationModule") or {}).get("briefTitle"),
            "status": (p.get("statusModule") or {}).get("overallStatus"),
            "phases": (p.get("designModule") or {}).get("phases") or [],
            "conditions": (p.get("conditionsModule") or {}).get("conditions") or [],
            "url": f"https://clinicaltrials.gov/study/{nct.upper()}"}


def ctgov_search(condition: str, n: int = 10) -> list[dict[str, Any]]:
    d = net.get_json(CTGOV, {"query.cond": condition, "pageSize": n,
                             "fields": "NCTId,BriefTitle,OverallStatus,Phase,Condition"})
    out = []
    for s in d.get("studies") or []:
        p = s.get("protocolSection") or {}
        nct = (p.get("identificationModule") or {}).get("nctId")
        if nct:
            out.append({"id": nct, "title": (p.get("identificationModule") or {}).get("briefTitle"),
                        "status": (p.get("statusModule") or {}).get("overallStatus"),
                        "phases": (p.get("designModule") or {}).get("phases") or []})
    return out


def pubmed_summaries(pmids: list[str]) -> dict[str, dict[str, Any]]:
    if not pmids:
        return {}
    d = net.get_json(f"{EUTILS}/esummary.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
    res = d.get("result") or {}
    out = {}
    for pid in pmids:
        r = res.get(pid)
        if r and not r.get("error"):
            out[pid] = {"id": pid, "title": r.get("title"), "journal": r.get("fulljournalname"),
                        "year": (r.get("pubdate") or "")[:4], "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"}
    return out


def crossref_work(doi: str) -> dict[str, Any]:
    d = net.get_json(f"{CROSSREF}/{urllib.parse.quote(doi, safe='/')}")
    m = d.get("message") or {}
    return {"id": doi, "title": (m.get("title") or [None])[0], "journal": (m.get("container-title") or [None])[0],
            "url": f"https://doi.org/{doi}"}
