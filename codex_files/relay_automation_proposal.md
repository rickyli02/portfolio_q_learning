## Relay Automation Proposal

Date: 2026-03-26
Owner: Codex

## Goal

Reduce manual handoff friction between Claude and Codex so work can continue cleanly when one agent finishes a task block.

## Practical options

### Option 1: File-based handoff queue

Create a shared handoff file, for example `shared_agent_files/handoff_queue.md`, with entries containing:

- sender
- timestamp
- task status
- files changed
- requested next action
- blocking questions

Pros:
- simple
- auditable
- no external services

Cons:
- still requires a human or a wrapper script to reopen the other agent

### Option 2: Local wrapper script around agent sessions

Create a local script that:

1. launches an agent task,
2. waits for a completion marker in a shared file,
3. opens the counterpart agent with the handoff summary,
4. repeats until completion or manual stop.

Pros:
- closer to automatic relay
- keeps coordination inside the repo

Cons:
- depends on stable CLI entrypoints for both agents
- needs careful guardrails to avoid loops

### Option 3: Polling watcher on shared files

Use a lightweight watcher process that monitors:

- `shared_agent_files/dialogue.txt`
- a dedicated handoff file
- optional completion marker files under `claude_files/` and `codex_files/`

When a completion marker appears, the watcher triggers the corresponding agent command.

Pros:
- more automatic than manual relay
- easy to inspect

Cons:
- requires stable local automation hooks
- may be brittle if agent output formats change

## Recommended near-term path

Start with Option 1 plus a documented handoff protocol. It gives most of the coordination benefit immediately without depending on undocumented automation hooks.

After that, if the local CLIs are stable and user-approved, build Option 2 as a thin wrapper around the handoff file.

## Proposed handoff protocol

Every completed task handoff should include:

- task completed
- files touched
- tests run
- known risks
- exact next recommended step
- whether Codex review is required before Claude proceeds

## Suggested next implementation if automation is pursued

1. Add `shared_agent_files/handoff_queue.md`.
2. Add a simple repo-local script that appends structured handoff entries.
3. Add optional watcher automation only after the handoff format is stable.

## Guardrails

- Never allow automatic relay to edit user-owned config files.
- Require explicit stop conditions to prevent agent ping-pong loops.
- Preserve a readable audit trail in shared files.
