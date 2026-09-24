import json
import random

import pytest

from sciprobe import net, packs, runner
from sciprobe.adapters import CallableAdapter
from sciprobe.model import AgentReply, Kind, OracleTask, Reference, Status
from sciprobe.packs import families
from sciprobe.scoring import score_task
from sciprobe.sourcing import live
from sciprobe.sourcing.bank import OpenItem, evaluate_open, load_custom_questions, rubric_elements
from sciprobe.sourcing.harvest import (
    AgentProfile,
    LocalCatalog,
    harvest,
    parse_kit,
    parse_readme_usage,
    parse_simple_yaml,
    route_prompt,
)
from sciprobe.verify import extract_ids, extract_smiles, fact_check
from sciprobe.verify.provenance import provenance


# ------------------------------------------------------------------ fake authorities
FAKE = {
    "pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin": {"PropertyTable": {"Properties": [
        {"CID": 2244, "MolecularWeight": "180.16", "MolecularFormula": "C9H8O4", "IsomericSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}]}},
    "pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244": {"PropertyTable": {"Properties": [
        {"CID": 2244, "MolecularWeight": "180.16", "MolecularFormula": "C9H8O4", "IUPACName": "2-acetyloxybenzoic acid"}]}},
    "data.rcsb.org/rest/v1/core/entry/1UBQ": {"struct": {"title": "UBIQUITIN"}, "exptl": [{"method": "X-RAY DIFFRACTION"}],
                                               "rcsb_entry_info": {"resolution_combined": [1.8], "deposited_polymer_monomer_count": 76}},
    "rest.uniprot.org/uniprotkb/P00519.json": {"primaryAccession": "P00519", "sequence": {"length": 1130},
                                                "genes": [{"geneName": {"value": "ABL1"}}], "organism": {"scientificName": "Homo sapiens"},
                                                "proteinDescription": {"recommendedName": {"fullName": {"value": "Tyrosine-protein kinase ABL1"}}}},
    "www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL941.json": {"molecule_chembl_id": "CHEMBL941", "pref_name": "IMATINIB",
                                                             "max_phase": "4.0", "molecule_properties": {"full_mwt": "493.62", "full_molformula": "C29H31N7O"}},
    "clinicaltrials.gov/api/v2/studies/NCT00000001": {"protocolSection": {"identificationModule": {"briefTitle": "Imatinib in CML"},
                                                                            "statusModule": {"overallStatus": "COMPLETED"}}},
    "clinicaltrials.gov/api/v2/studies?": {"studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT00000001", "briefTitle": "Imatinib in CML"},
                                                                            "statusModule": {"overallStatus": "ACTIVE_NOT_RECRUITING"}}}]},
    "eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi": {"result": {"uids": ["11111111"], "11111111": {"title": "Imatinib resistance in CML"}}},
}


def fake_transport(url, headers, body, method):
    for key, val in FAKE.items():
        if key in url:
            return 200, val
    return 404, "not found"


@pytest.fixture
def fake_net():
    net.set_transport(fake_transport)
    old = net.MIN_INTERVAL_S
    net.MIN_INTERVAL_S = 0
    yield
    net.MIN_INTERVAL_S = old
    net.set_transport(None)


# ------------------------------------------------------------------ fact-checking
def test_extract_ids_and_smiles():
    text = ("See NCT00000001, CHEMBL941, PMID: 11111111, doi 10.1000/xyz123. PDB 1UBQ and rcsb.org/structure/4HHB, "
            "UniProt P00519, PubChem CID 2244. Structure `CC(=O)Oc1ccccc1C(=O)O`.")
    ids = extract_ids(text)
    assert ids["nct"] == ["NCT00000001"] and ids["chembl"] == ["CHEMBL941"] and ids["pmid"] == ["11111111"]
    assert ids["pdb"] == ["1UBQ", "4HHB"] and ids["uniprot"] == ["P00519"] and ids["cid"] == ["2244"]
    assert ids["doi"] == ["10.1000/xyz123"]
    assert extract_smiles(text) == ["CC(=O)Oc1ccccc1C(=O)O"]
    assert extract_smiles("formula `C9H8O4`, `C16H28N2O4`, ring `C1CCCCC1`") == ["C1CCCCC1"]
    assert "pdb" not in extract_ids("We ran 1000 steps in 2024.")


def test_fact_check_flags_fabrications(fake_net):
    text = ("Imatinib (CHEMBL941) was tested in NCT00000001 and NCT99999999; see PMID 11111111 and PMID 22222222. "
            "Structure PDB 1UBQ, fake PDB 9ZZZ. SMILES `C1CC(`.")
    rep = fact_check(text, prompt="imatinib trials in CML")
    by = {(c.kind, c.value): c.status for c in rep.checks}
    assert by[("chembl", "CHEMBL941")] == "VERIFIED"
    assert by[("nct", "NCT00000001")] == "VERIFIED"
    assert by[("nct", "NCT99999999")] == "NOT_FOUND"
    assert by[("pmid", "11111111")] == "VERIFIED" and by[("pmid", "22222222")] == "NOT_FOUND"
    assert by[("pdb", "9ZZZ")] == "NOT_FOUND"
    assert by[("smiles", "C1CC(")] == "INVALID"
    assert rep.n_bad == 4 and 0 < rep.score < 1


def test_fact_check_off_topic(fake_net):
    rep = fact_check("Relevant trial: NCT00000001", prompt="semaglutide obesity outcomes")
    assert rep.checks[0].status == "OFF_TOPIC"


def test_fact_check_recomputes_formula_and_mw():
    # the real live failure: mol-toolkit on (S)-naproxen
    text = ("The SMILES `C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1` is valid; its formula is **C18H16O3**. "
            "Total = **280.323 g/mol**")
    rep = fact_check(text)
    by = {(c.kind, c.value): c for c in rep.checks}
    assert by[("formula", "C18H16O3")].status == "MISMATCH" and "C14H14O3" in by[("formula", "C18H16O3")].detail
    assert by[("mw", "280.323")].status == "MISMATCH"
    good = fact_check("Aspirin `CC(=O)Oc1ccccc1C(=O)O` is C9H8O4 with MW 180.16 g/mol (exact 180.0423 Da).")
    assert {c.status for c in good.checks} == {"VERIFIED"}


def test_fact_check_uses_prompt_molecule_when_reply_has_none():
    rep = fact_check("Its molecular weight is 250.1 g/mol.", prompt="MW of SMILES CC(=O)Oc1ccccc1C(=O)O ?")
    assert rep.checks and rep.checks[0].status == "MISMATCH"


def test_fact_check_ignores_echoed_invalid_input():
    reply = "The SMILES `C1CCCC` is invalid because ring 1 is never closed."
    assert fact_check(reply, prompt="MW of SMILES C1CCCC ?").checks == []
    assert fact_check(reply).checks[0].status == "INVALID"


# ------------------------------------------------------------------ provenance
def test_provenance_labels():
    assert provenance(AgentReply("x", meta={"output_types": ["reasoning", "function_call", "message"]})) == "tool-called"
    assert provenance(AgentReply("x", meta={"output_types": ["message"]})) == "no-tool-call"
    assert provenance(AgentReply("Plan: run one Python script with psi4_utils ...")) == "claims-tool-use"
    assert provenance(AgentReply("V = 5(1-e^-1) = 3.16 V")) == "no-evidence"
    assert provenance(AgentReply("", error="boom")) == "n/a"


# ------------------------------------------------------------------ harvesting
README = """# Psi4
## Usage
| Prompt | Input Files | Description |
|--------|-------------|-------------|
| "Calculate the energy of water at the HF/cc-pVDZ level" | water.xyz | Single-point HF energy |
| "What is the HOMO-LUMO gap of benzene?" | benzene.xyz, extra.xyz | Orbital analysis |

| Parameter | Description |
|---|---|
| model | chat model |
"""

KIT = {"agentRefs": [{"ref": "agents/alphafold", "role": "primary"}, {"ref": "agents/autodock"}, {"ref": "agents/chembl"},
                     {"ref": "agents/pubchem"}, {"ref": "agents/pubmed"}],
       "samplePrompts": [
           {"prompt": "Dock ibuprofen against COX-2 (PDB 5KIR) and report poses.", "expectedOutput": "Ranked poses, docking scores, and key residues."},
           {"prompt": "Search ChEMBL and PubChem for analogs of imatinib.", "expectedOutput": "Table of analogs"},
           {"prompt": "Predict the 3D structure of human lysozyme.", "expectedOutput": "Predicted structure"}]}


def test_parse_readme_usage():
    rows = parse_readme_usage(README, "psi4", "agents/psi4/README.md")
    assert [r.prompt for r in rows] == ["Calculate the energy of water at the HF/cc-pVDZ level", "What is the HOMO-LUMO gap of benzene?"]
    assert rows[1].input_files == ["benzene.xyz", "extra.xyz"] and rows[0].expected == "Single-point HF energy"


def test_parse_simple_yaml():
    y = parse_simple_yaml("name: mol-toolkit\ndescription: >\n  Agent dedicated\n  to chemistry.\ntags:\n  - cheminformatics\n  - rdkit\n")
    assert y["name"] == "mol-toolkit" and y["description"] == "Agent dedicated to chemistry." and y["tags"] == ["cheminformatics", "rdkit"]


def test_route_prompt():
    cands = [AgentProfile("alphafold", "protein structure prediction", ["structural-biology"]),
             AgentProfile("autodock", "molecular docking of ligands", ["molecule-docking"]),
             AgentProfile("chembl", "", []), AgentProfile("pubchem", "", [])]
    assert route_prompt("Search ChEMBL and PubChem for analogs", cands) == ["chembl", "pubchem"]
    assert route_prompt("Dock ibuprofen against COX-2", cands) == ["autodock"]
    assert route_prompt("Predict the 3D structure of lysozyme", cands) == ["alphafold"]


def test_harvest_local_catalog(tmp_path):
    for name, meta in [("psi4", "name: psi4\ndescription: quantum chemistry\ntags:\n  - quantum-chemistry\n"),
                       ("autodock", "name: autodock\ndescription: molecular docking\ntags:\n  - molecule-docking\n"),
                       ("alphafold", "name: alphafold\ndescription: protein structure prediction\n"),
                       ("chembl", "name: chembl\n"), ("pubchem", "name: pubchem\n"), ("pubmed", "name: pubmed\n")]:
        d = tmp_path / "agents" / name
        (d / "tools" / name / "example-input-files").mkdir(parents=True)
        (d / "metadata.yaml").write_text(meta)
    (tmp_path / "agents/psi4/README.md").write_text(README)
    (tmp_path / "agents/psi4/tools/psi4/example-input-files/water.xyz").write_text("3\n\nO 0 0 0")
    (tmp_path / "starter-kits/life").mkdir(parents=True)
    (tmp_path / "starter-kits/life/kit.json").write_text(json.dumps(KIT))
    prof = harvest(LocalCatalog(tmp_path))
    assert prof["psi4"].tags == ["quantum-chemistry"] and len(prof["psi4"].listed) == 2
    assert prof["psi4"].example_inputs == ["agents/psi4/tools/psi4/example-input-files/water.xyz"]
    assert [lp.prompt[:4] for lp in prof["autodock"].listed] == ["Dock"]
    assert len(prof["chembl"].listed) == 1 and len(prof["pubchem"].listed) == 1
    assert prof["alphafold"].listed[0].expected == "Predicted structure"


def test_kit_parse():
    lps = parse_kit(json.dumps(KIT), "k.json")
    assert len(lps) == 3 and lps[0].candidate_agents[0] == "alphafold"


# ------------------------------------------------------------------ open items
def test_rubric_elements():
    assert rubric_elements("Ranked poses, docking scores, and key interacting residues.") == [
        "Ranked poses", "docking scores", "key interacting residues"]


def test_evaluate_open_offline_rubric():
    item = OpenItem("o1", "autodock", "Dock ibuprofen", "kit", expected="Ranked poses, docking scores, and key residues.")
    good = evaluate_open(item, AgentReply("Ranked poses: 1..3. Docking scores -7.1 kcal/mol. Key residues Arg120, Tyr355."), offline=True)
    weak = evaluate_open(item, AgentReply("Done."), offline=True)
    assert good["status"] == "OK" and good["rubric_coverage"] == 1.0
    assert weak["status"] == "WEAK" and "rubric missing" in weak["diagnosis"]
    assert evaluate_open(item, AgentReply("", error="HTTP 500"))["status"] == "ERROR"


def test_evaluate_open_flags_fabricated_ids(fake_net):
    item = OpenItem("o2", "pubchem", "Find imatinib analogs", "kit")
    res = evaluate_open(item, AgentReply("Analog CHEMBL941 and CHEMBL99999999."))
    assert res["status"] == "FLAGGED" and "NOT_FOUND" in res["diagnosis"]


def test_custom_questions_and_runner_open(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(json.dumps({"agent": "mol-toolkit", "prompt": "What is aspirin?", "expected": "structure, uses"}) + "\n")
    items = load_custom_questions(f)
    assert items[0].agent == "mol-toolkit" and items[0].source == "custom"
    res = runner.run([], [], CallableAdapter(lambda a, p: "Aspirin structure `CC(=O)Oc1ccccc1C(=O)O`; uses: analgesic."),
                     open_items=items, open_offline=True)
    assert res.open and res.open[0]["status"] in ("OK", "WEAK")


# ------------------------------------------------------------------ live oracles
def test_live_generators_with_fake_authorities(fake_net):
    rng = random.Random(0)
    old = (live.DRUGS, live.PDB_IDS, live.CHEMBL_IDS, live.UNIPROT)
    live.DRUGS, live.PDB_IDS, live.CHEMBL_IDS, live.UNIPROT = ["aspirin", "notadrug"], ["1UBQ", "0000"], ["CHEMBL941"], ["P00519"]
    try:
        ts = live.pubchem_tasks(2, rng) + live.rcsb_tasks(2, rng) + live.chembl_tasks(2, rng) + live.uniprot_tasks(1, rng) + live.ctgov_tasks(1, rng)
    finally:
        live.DRUGS, live.PDB_IDS, live.CHEMBL_IDS, live.UNIPROT = old
    by = {t.id: t for t in ts}
    assert by["live.pubchem.cid.aspirin"].reference.value == 2244
    assert by["live.pdb.resolution.1UBQ"].reference.value == 1.8 and by["live.pdb.residues.1UBQ"].reference.value == 76
    assert by["live.chembl.phase.CHEMBL941"].reference.value == 4
    assert by["live.uniprot.length.P00519"].reference.value == 1130
    status = by["live.ctgov.status.NCT00000001"].reference.value
    assert "active, not recruiting" in status
    assert all("live lookup" in t.reference.source for t in ts)
    v = score_task(by["live.ctgov.status.NCT00000001"], AgentReply('ANSWER: {"value": "Active, not recruiting"}'), "clinical-trials")
    assert v.status == Status.PASS


# ------------------------------------------------------------------ families pack
def test_family_references_are_right():
    assert families.translate(families.DNA) == "MAIVMGR"
    assert families.revcomp("AAGC") == "GCTT"
    t = {x.id: x for x in families.tasks()}
    assert t["fam.seq.gc"].reference.value == pytest.approx(100 * sum(b in "GC" for b in families.DNA) / len(families.DNA))
    assert t["fam.stat.pearson"].reference.value == pytest.approx(6 / 60 ** 0.5)
    assert t["fam.stat.binom"].reference.value == pytest.approx(22 / 1024)
    assert t["fam.xtal.si.density"].reference.value == pytest.approx(2.329, abs=0.002)
    assert t["fam.groups.aspirin.ester"].reference.value == ["true", "yes"]
    assert t["fam.groups.aspirin.amide"].reference.value == ["false", "no"]
    assert t["fam.groups.paracetamol.phenol"].reference.value == ["true", "yes"]


def test_smiles_kind_accepts_any_spelling_and_checks_stereo():
    from sciprobe.invariance import randomized_smiles

    s_nap = "C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1"
    respelled = randomized_smiles(s_nap, 1, seed=3)[0]
    task = OracleTask("t", "d", ["a"], "q", Reference(Kind.SMILES, s_nap))
    ok = score_task(task, AgentReply(f'ANSWER: {{"value": "{respelled}"}}'), "opsin")
    flat = score_task(task, AgentReply('ANSWER: {"value": "CC(C(=O)O)c1ccc2cc(OC)ccc2c1"}'), "opsin")
    other = score_task(task, AgentReply('ANSWER: {"value": "CCO"}'), "opsin")
    assert ok.status == Status.PASS
    assert flat.status == Status.FAIL and "stereochemistry" in flat.diagnosis
    assert other.diagnosis == "different molecule"


def test_generation_checker():
    t = {x.id: x for x in families.tasks()}["fam.gen.acids"]
    good = '["OC(=O)c1ccccc1O", "OC(=O)c1ccc(Cl)cc1", "OC(=O)CCc1ccccc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "OC(=O)c1ccc2ccccc2c1"]'
    bad = '["CCO", "OC(=O)CCc1ccccc1", "OC(=O)CCc1ccccc1", "C1CC(", "OC(=O)C"]'
    assert score_task(t, AgentReply(f'ANSWER: {{"value": {good}}}'), "janus").status == Status.PASS
    v = score_task(t, AgentReply(f'ANSWER: {{"value": {bad}}}'), "janus")
    assert v.status == Status.FAIL and "1 satisfy" in v.diagnosis


def test_retro_checker():
    t = {x.id: x for x in families.tasks()}["fam.retro.aspirin"]
    ok = score_task(t, AgentReply('ANSWER: {"value": ["O=C(O)c1ccccc1O", "CC(=O)OC(C)=O"]}'), "aizynthfinder")
    wrong = score_task(t, AgentReply('ANSWER: {"value": ["c1ccccc1", "CC(=O)O"]}'), "aizynthfinder")
    assert ok.status == Status.PASS and wrong.status == Status.FAIL and "salicylic" in wrong.diagnosis


def test_packs_include_families_and_new_agents():
    tasks, probes = packs.load()
    agents = {a for t in tasks for a in t.agents}
    for a in ("biopython", "spglib", "cantera", "cobrapy", "stat-agent", "opsin", "molecular-groups", "janus",
              "aizynthfinder", "tmalign", "pyscf", "openbabel"):
        assert a in agents, a
