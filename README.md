# sciprobe

Scientific evaluation for AI agents (built for Microsoft Discovery). It asks agents real science questions and checks their answers against ground truth, authoritative databases, and physics. LLM judges are used only as a last resort.

## Where questions come from (in priority order)

| # | Source | Graded by |
|---|---|---|
| 1 | **Listed prompts**: starter-kit `samplePrompts`, agent README "Usage" tables, a PR branch (`--catalog <path>`), or your own file (`--questions q.jsonl`) | fact-checking against authorities, rubric coverage (`expectedOutput`), provenance |
| 2 | **Built-in oracle packs**: curated tasks with cited answers | unit-aware tolerance, plus a diagnosis of *why* a wrong answer is wrong |
| 3 | **Verifier families**: one strategy per kind of agent, for everything that isn't listed | exact computation, symmetry, analytic solutions, RDKit, self-consistency |
| 4 | **Live oracles** (`--live N`): fresh questions whose answers are fetched from PubChem / RCSB / UniProt / ChEMBL / ClinicalTrials.gov at run time | same as 2; re-drawn every run, so agents can't be tuned to it |
| 5 | **Invariance probes**: equivalent inputs (re-ordered SMILES, rotated geometry, enantiomers, unit reframing, paraphrase, repeats) | answers must agree; no ground truth needed |
| 6 | **Negative controls**: invalid or impossible inputs | the agent must refuse, not invent a number |
| 7 | **Retrieval arena**: GraphRAG vs vector RAG vs live search | BenchmarkQED-style counterbalanced pairwise judging, assertions, citation checks, paired statistics |

Run `sciprobe bank --catalog github` for the coverage map: per agent, what can be asked and where someone still needs to step in.

## Verification strategies

* **Oracles:** analytic physics (RC circuit, RLC resonance), literature benchmarks (Ghia 1982 cavity flow, Szabo & Ostlund H₂, the Psi4 test-suite water energy), RDKit recomputation.
* **Fact-check (`sciprobe factcheck`):**
  * every CID, ChEMBL ID, PDB ID, UniProt accession, NCT number, PMID or DOI in an answer is looked up; each must exist and be on-topic;
  * SMILES must parse;
  * formulas and MWs stated for a molecule must match RDKit.
* **Families:** sequence (genetic code), crystallography (space groups, density), kinetics (analytic), FBA (e_coli_core), statistics (exact), name→structure (canonical SMILES), functional groups (SMARTS), generative design (constraint satisfaction), retrosynthesis (key disconnection + atom accounting), structure alignment (self-alignment identity).
* **Provenance:** did the agent actually call its tool, or answer from the language model alone?

## Quick start

```powershell
python -m pip install -e .[test,net]              # net = truststore: use the OS cert store behind TLS-inspecting proxies
python -m sciprobe bank --catalog github --live 2  # coverage map + gaps
python -m sciprobe run --adapter reference --tags core             # offline self-test (RDKit gold agent)
python -m sciprobe run --adapter faulty --tags core                # proves planted mistakes are caught
python -m sciprobe factcheck --replay                              # fact-check Discovery Toolbox transcripts
python -m sciprobe arena demo                                      # retrieval A/B on a fictional corpus
python -m sciprobe arena power                                     # how many questions do you need?
```

Live against Discovery (a fresh investigation per run, deleted afterwards by default):

```powershell
python -m sciprobe run --adapter discovery --project <project> --workspace-arm-id <id> `
  --catalog github --tags core --live 2 --record out\transcripts.jsonl --fail-under 70
```

`--cleanup all|conversations|none` (default `all`) deletes only what sciprobe created. `--record` keeps transcripts locally, and `--adapter replay` re-scores them later.

## Retrieval arena (Bookshelf / GraphRAG vs vector RAG)

Hold the corpus, questions and answer model constant; vary only retrieval. Arms: `RetrieveThenAnswer` (one shared generator plus any retriever: `AzureAISearchRetriever`, `HttpRetriever`, `TfidfRetriever`, `NullRetriever` for the no-retrieval baseline), and `CallableArm` / `DiscoveryAgentArm` for end-to-end systems. It reads BenchmarkQED AutoQ question/assertion files and answer files, writes an AutoE input layout plus `settings.yaml` (`arena export-autoe`), and treats the *question* as the unit of analysis: sign test, Wilson and bootstrap confidence intervals, Holm correction, judge order-consistency, and warnings for confounds.
