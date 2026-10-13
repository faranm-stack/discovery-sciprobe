# Discovery probes: portable handoff

Packet: `2026-10-13.2` | Canonical home: Discovery CatalogProbe

This packet is copied to both private repositories so either project has a
starting point. Update the canonical copy and deliberately synchronize the
other; neither Copilot product automatically synchronizes chats or files.

## Purpose and current truth

- **CatalogProbe:** functional evidence for catalog agents and starter kits.
  Source: [repository](https://github.com/faranm-stack/discovery-catalogprobe).
- **SciProbe:** scientific answer grading using reference data, tolerances and
  consistency checks. Source:
  [repository](https://github.com/faranm-stack/discovery-sciprobe).
- Both are repositories/CLIs today, not installed commercial extensions.
  The eventual shared extension and Discovery app target are planned.
- Discovery **services/Studio** is the pilot target. Successful services use
  does not prove local Discovery app compatibility.
- Starter kits have equal priority to agents. Work from a manual runbook,
  automate repeatable steps, document gaps, and compare pinned catalog and
  runtime versions. Do not equate schema validity with successful loading.

## Source snapshot and evidence limits

Inspected repository heads before this documentation update:

| Source | Revision | What it establishes |
| --- | --- | --- |
| CatalogProbe | `edadb48e085474ba3f34920d2799cdd34b72b628` | CLI, static checks, planning, response checks, JSON reports and baseline comparison |
| SciProbe | `574461bf006737eaa74ffbc3a2c59f980a266d81` | Question packs, scoring, authority lookup and replay/direct adapters |
| Life-sciences kit | `kit.json` blob `01d22073034488fc4d5ed5179be66a51a216dd8e` | 20-agent kit; sample `sp-4` concerns PubMed literature and ongoing trials |
| PubMed agent | `agent.yaml` blob `91e46bc59220e68a0859439d9fafcc6e1af47311` | Catalog instruction snapshot, not deployed tool-image identity |

CatalogProbe's [runner](https://github.com/faranm-stack/discovery-catalogprobe/blob/edadb48e085474ba3f34920d2799cdd34b72b628/dft/runner.py)
loads the client supplied through `--client-dir` from the PR #100 evaluation
utility. It does not currently deploy agents. Toolbox deployment is a separate
operator step; a replaceable deploy adapter is a design goal, not implemented
proof. A Toolbox invocation is not a CatalogProbe CLI run.

Historical evidence: 23 CatalogProbe unit tests and a static CI run passed;
the baseline contains three bad keyword-argument call sites in two agents.
A baseline-green run does not mean the catalog has no failures. These results
were not rerun in this session. Earlier SciProbe experiments are separate
evidence; inconsistent historical answer totals must be reconciled from raw
records before quoting an exact total.

## Five-pattern assessment

| Pattern | Current evidence / smallest gap |
| --- | --- |
| Tool use | Clients, replay and graders exist; deployment and proof collection are not one integrated runner |
| Reflection | Deterministic checks exist; no validated score-audit loop or need for another model judge is established |
| Planning | Test plans exist; the pilot still needs verified runtime inputs and an execution receipt |
| Workers | CatalogProbe has bounded async cases, not demonstrated multi-agent orchestration; keep one case |
| Memory | Results/baselines exist; this packet adds cross-host continuity, not automatic synchronization |

## 2026-10-13 updated live pilot outcome

- Intended target: the existing `PRValidation` project, not `RepoValidation`
  and not a newly created project.
- The earlier preflight-only attempt did not invoke an agent. The user later
  renewed permission for one bounded live pilot. These are separate attempts.
- Toolbox deployed and verified `pubmed` in PRValidation, using the existing
  registered PubMed tool and existing `gpt-5-mini` model. No tool image build
  or new storage container was part of the deployment.
- Read-only ARM checks confirmed PubMed registration Succeeded, declared GPU
  count zero, and a provisioned Standard_D4s_v6 CPU pool. The image is a mutable
  `pubmed:latest` tag; its digest/startup were not captured.
- One request was rejected before agent execution because our investigation
  name was too long. Discovery requires 3-24 letters/digits/hyphens. After
  correcting only that input, one REST invocation returned an answer.
- The answer reports PubMed-ExecutepythonCode success, a 45.90-second compute
  run, and final_results.json saved/shared. Those are agent-reported claims:
  the exported Toolbox log contains messages, not the raw tool-result record.
  Job success, actual compute-call count and file persistence are not yet
  independently verified.
- Both returned citation titles, PMIDs, years and DOIs were independently
  matched against official PubMed ESummary metadata: PMID
  [42865775](https://pubmed.ncbi.nlm.nih.gov/42865775/) and
  [42864707](https://pubmed.ncbi.nlm.nih.gov/42864707/).
  This was a direct metadata check, not a SciProbe grading run.
- The Toolbox round-trip was 470,278 ms. Overall setup exceeded the requested
  20-minute ceiling. Billed usage remains Unknown. No Log Analytics query,
  CatalogProbe CLI run, full-kit test or Discovery app test was performed.
- Azure CLI project GET returned Unauthorized while Toolbox access worked.
  Cause unconfirmed; no permission change or identity-root-cause claim.
- Raw transcript, metadata receipt and a readable report are retained in the
  user's approved local evidence folder, not published to these repositories.

## Pilot scope and next action

One CPU-only **literature subset** of
[life-sciences sample sp-4](https://github.com/microsoft/discovery/blob/main/starter-kits/life-sciences/kit.json),
using PubMed. The complete sample also asks for ongoing clinical trials; that
part is explicitly excluded from this first case. The rest of the kit,
including GPU agents, remains untested.

**Do not rerun first.** Inspect the existing Studio conversation in PRValidation,
investigation `cp-bio-1013-1440`, for its actual tool-result card and output JSON.
Capture job/correlation ID, terminal state, image identity and output contents.
Then update only the evidence gaps above. The local receipt holds the returned
conversation ID and resource URI.

Prompt-scope summary (the original Toolbox transcript retains the full submitted text):

> CPU-only literature smoke test for the life-sciences kit. Use your existing
> PubMed tool to find at most two publications about GLP-1 receptor agonists
> and cardiovascular outcomes. Return each exact title, PMID, publication year,
> DOI if present, and one sentence about the study's reported finding. Use no
> more than one compute execution and existing approved storage only. Do not
> install packages, request GPU, create infrastructure, inspect unrelated data,
> or retry a failed tool. On failure, stop and return the exact error and run
> ID. Do not answer from memory or claim current clinical-trial status. This is
> a literature-only subset of sample sp-4, not medical advice or a full kit test.

## Short meeting script

"We have two complementary test harnesses: CatalogProbe checks whether an agent
uses its tools and completes a task; SciProbe checks the scientific answer.
The offline checks and graders exist. For the next demo we're selecting the
CPU PubMed part of the life-sciences starter kit, not deploying all 20 agents.
We will show the real Studio conversation, tool outcome and citation checks."

"We have now loaded the biology agent and received a real Studio answer.
Both returned citation identities match PubMed. The agent reports its CPU tool
ran and saved a result, but we still need the actual job receipt and file to
independently confirm those parts. This is partial starter-kit coverage, not
a whole-kit pass. We will inspect that existing evidence before spending tokens
on another run. App support and extension packaging remain future work."

## Routing, persistence and resumption

- Use the [Microsoft 365 starter](./M365-STARTER.txt) for one narrow draft or
  source-finding task. Verify work-account access; private GitHub links may not
  be readable. Attach an approved snapshot if necessary.
- Keep raw traces in approved access-controlled storage, not a public repo.
  These private repos contain the method and sanitized handoff, not credentials
  or an implied permission to redistribute internal data.
- Record measured usage only when exposed by the provider. The two Copilot
  products have different licensing/metering; Markdown is not a spend limiter.
- Resume from this packet, not the entire chat. Preserve failures, unknowns,
  limits, current permissions and the one next action.
