---
name: probe-harness
description: "Use for Discovery SciProbe or CatalogProbe harness work, CPU biology starter-kit demos, loop engineering, token budgets, and handoffs between GitHub Copilot and Microsoft 365 Copilot. Reuse evidence, choose one bounded next task, and save a portable checkpoint."
---

# Probe harness: bounded work and portable context

Read [the handoff](../../../docs/copilot/HANDOFF.md) first. It is a dated snapshot,
not fresh runtime proof or authorization. In Microsoft 365 Copilot, attach the
handoff and [starter](../../../docs/copilot/M365-STARTER.txt) explicitly; do not
assume a repository link installs this skill or grants access.

## Choose the cheapest adequate route

- Existing code/CI: counts, schemas, numerical grading, artifact checks, diffs.
- Approved Microsoft 365 Copilot work chat: requirements, cited internal
  context, open-source comparison, ADO text and meeting-script drafts.
- Code-capable assistant/approved runner: precise edits, deployment, invocation,
  source-linked failures and actual evidence collection.
- Confirm the receiving surface's access, tools and licensing. Do not assume
  Microsoft 365 work is free, unmetered, or able to run Azure commands.

## Apply patterns only where they earn their cost

1. **Tools:** narrow typed calls; explicit failures. Record the actual transport
   and distinguish deployment, tool-call, tool-result and artifact evidence.
2. **Reflection:** deterministic assertions first. Preserve the original result.
   No retry-until-green. Design-review loops default to at most three passes;
   they do not authorize three live executions.
3. **Planning:** show scope, acceptance criteria and a stop condition before
   acting. Start with one case and one worker. A partial kit is not a kit pass.
4. **Workers:** use only for substantial independent work. Async test cases are
   not separate reasoning agents. Do not launch a standard worker roster.
5. **Memory:** keep the handoff and current question in context; retrieve only
   relevant source/evidence. Keep full raw proof in approved restricted storage.

## Smoke-test envelope

- Agree on target, case, allowed resource use and elapsed-time ceiling.
- Default pilot: one CPU-only invocation, concurrency one, no new infrastructure,
  no GPU, no LLM judge, no permission changes and no automatic retry.
- Check authentication and resource identity before mutation. Stop on denied
  access, missing prerequisites, exhausted time, or uncertain permissions.
- Markdown and prompts cannot enforce a dollar cap or stop server-side compute.
  Use provider/runner controls; if a call times out, record its ID and unknown
  state. Do not submit a duplicate or claim it was cancelled.
- A correct-looking answer or tool call alone does not establish a completed
  tool execution. Missing evidence is Unknown, not Pass.

## Handoff after every useful result or blocker

Update the handoff with source revisions, accepted scope, evidence links,
actual outcomes, open jobs, limits and exactly one next action. Separate
implemented, executed, planned and blocked. Never transfer credentials.

Resume by confirming the packet revision, known fact, evidence access and
next action. Reuse matching evidence; revalidate only changed dependencies.
Runtime scores do not approve PRs, clinical use, or product release.
