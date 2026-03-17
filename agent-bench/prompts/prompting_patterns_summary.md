# Company Prompting Patterns -- Summary for Thesis Methodology

Based on 143 exported agent conversation logs from Kimara.ai production workflows (Jan--Mar 2026).

## What the company does

The company runs a multi-agent pipeline where a human orchestrator breaks feature work into discrete tasks (usually 3--10 per feature) and dispatches each one to a separate Claude Code session. Each session gets a single structured prompt and works on its own. Three roles keep showing up: implementer, spec-compliance reviewer, and code-quality reviewer. Occasionally there are exploration agents that just read and summarize code before anyone writes a plan.

## How their prompts are structured

The prompts follow a pretty consistent five-part layout:

1. A role sentence at the top ("You are implementing Task N...", "You are a spec compliance reviewer...")
2. A context block with project background, ticket references, what happened in earlier tasks, and the goal of the current change
3. The actual task specification, often with exact file paths, line ranges, and inline code showing what the change should look like
4. Execution steps as a numbered list (read, edit, test, commit)
5. An output/report format telling the agent what to return when it's done

## How specific the instructions get

Very specific. Prompts regularly include complete function implementations pasted inline, exact file paths with line numbers like `src/server/api/routers/teams.ts:46-137`, verbatim commit messages, shell commands to run, and expected test output. Reviewer prompts include checklist templates where the agent fills in pass/fail per requirement. The company almost never leaves implementation decisions open-ended. When they do, they give a short list of acceptable approaches.

## How they package context

They reference existing code by file path and line range rather than pasting it. The agent reads files itself via tool calls. Reference implementations are pointed to by path ("Study `BackendFlavorSelector.tsx:20-365` for the styled card pattern"). Plan documents live in `docs/plans/` and get cited by path. A repo-level `AGENTS.md` file lays out project-wide conventions, and agents read it on startup.

## Guardrails they use

A few things keep coming up:

- "Before You Begin" blocks that ask the agent to raise questions before doing anything
- Edit-scope restrictions ("only edit files under agent-bench/")
- Warnings on reviewer prompts not to trust the implementer's self-report ("CRITICAL: Do Not Trust theport" -- verify independently)
- Self-review checklists at the end of implementer prompts
- Explicit prohibitions against modifying eval harness files, restructuring the repo, or adding packages
- Prescribed commit messages; agents run lint and type-checks before committing

## What they expect back

Implementer agents return a structured report: what was implemented, which files changed, test results, self-review notes, and any concerns. Reviewer agents return a compliance checklist (pass/fail per requirement), a verdict (APPROVED or NEEDS FIXES), and an issues list sorted by severity. In practice the agents write files directly and commit to git; the human checks the diff.

## What this means for the benchmark

The production prompt regime in this thesis replicates the company's average implementer prompt: a role preamble, the task specification with file paths and source context, execution steps including running tests, constraints on what the agent can and cannot touch, a self-review checklist, and a structured output format. This is not a minimal or idealized prompt -- it's the kind of prompt shaped by months of production use with multi-agent orchestration.

The key difference from the company's actual workflow is that our benchmark agents cannot use tool calls to read files or run commands interactively. Instead, source files and test commands are provided upfront in the prompt, and the agent returns modified file contents in its response. The architecture implementation handles the rest.
