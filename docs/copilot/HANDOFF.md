# Discovery probes: portable handoff

Packet: `2026-10-13.1` | Canonical home: Discovery CatalogProbe

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

## 2026-10-13 preflight outcome

- Intended target: the existing `PRValidation` project, not `RepoValidation`
  and not a newly created project.
- Live Toolbox data-plane inventory read that project successfully but found
  no matching `pubmed` agent. The shared PubMed tool exists in the selected
  workspace. Azure CLI also reported the intended lab tenant/subscription.
- This does NOT verify the tool image, CPU binding, storage access, current
  model deployment, or Log Analytics permissions. A separate filtered inventory
  query returned an empty result; it does not overturn the successful live
  project read.
- Read-only setup exceeded the requested 20-minute ceiling. Deployment and
  model invocation were therefore NOT started. No compute job, new resource,
  Studio conversation, scientific result, or demo pass was created.
- No live job was started by this session. Do not interpret this as a statement
  about unrelated jobs. No logs were queried and no billed usage was measured.

## Next bounded demonstration

One CPU-only **literature subset** of
[life-sciences sample sp-4](https://github.com/microsoft/discovery/blob/main/starter-kits/life-sciences/kit.json),
using PubMed. The complete sample also asks for ongoing clinical trials; that
part is explicitly excluded from this first case. The rest of the kit,
including GPU agents, remains untested.

1. Verify current project, PubMed tool definition/image, CPU node pool, model,
   existing output storage and absence of a conflicting deployment.
2. With fresh scoped approval, deploy only PubMed using the existing tool and
   resources. Record the actual catalog and deployed image identities.
3. Submit ONE named investigation/conversation in Studio through Toolbox REST.
   No transport failover, new packages, GPU, repeated calls or permission fixes.
4. Capture returned IDs, paired tool calls/results, terminal state and actual
   outputs. Check up to two PMIDs/titles against PubMed. Do not count answer
   prose as proof of tool success. Label replay/CLI/manual grading accurately.
5. Save a human-readable report and exact returned Studio link, if available.
   A guessed deep link is not evidence. If blocked, preserve the error and stop.

Prepared prompt (not executed):

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

"Today's access checks found the project and existing PubMed tool, but the
agent is not deployed there and no live test completed within the timebox.
The next action is one approved load-and-invoke test, with no GPU or new
infrastructure. We now keep the workflow and checkpoint in both repos so
Microsoft 365 Copilot can help with planning and reporting without restarting
the investigation. App support and combined extension packaging remain future
work."

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
