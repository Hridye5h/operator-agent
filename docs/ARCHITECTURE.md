# Architecture

## Shape

```
                    ┌──────────────┐
      goal  ───────▶│  Agent loop  │◀──────── observations
                    │ (agent/core) │
                    └──────┬───────┘
                           │ one action per turn
                    ┌──────▼───────┐
                    │   Planner    │   Gemini, or a deterministic fallback
                    │  (agent/llm) │
                    └──────┬───────┘
                           │ Decision{action, args, confidence}
                    ┌──────▼───────┐
                    │ ToolRegistry │   name → schema → callable
                    │(agent/tools) │
                    └──────┬───────┘
          ┌────────────────┼────────────────┬──────────────┐
          ▼                ▼                ▼              ▼
    files.py         browser.py        browser.py      human.py
    documents        write via         verify via      escalation
                     Playwright        read-back
          │                │                │              │
          └────────────────┴────────┬───────┴──────────────┘
                                    ▼
                               Trace (append-only)
                      trace.json · report.md · screenshots
```

## The loop

`Agent.run()` holds three things: the goal, an observation buffer, and a working
state dict. Each turn it builds a prompt from those plus the tool catalogue,
asks the planner for exactly one action, executes it, records the result, and
repeats until the goal is verified or the step budget runs out.

```
for step in range(max_steps):
    decision = planner.decide(SYSTEM_PROMPT, context(goal, observations, state))

    if decision.action == "finish":
        if not verified:                 # ← the hard gate
            observations.append("SYSTEM: Refused to finish: not verified.")
            continue
        return RunResult("completed", verified=True, evidence=evidence)

    if decision.confidence < floor:      # ← uncertainty routes to a human
        decision = escalate(decision)

    result, ms = tools.invoke(decision.action, bind(decision.args, state))
    trace.add("observation", result.brief(), ok=result.ok, retriable=result.retriable)
    observations.append(f"{decision.action}(...) -> {result.brief()}")

    if result.ok:
        absorb(decision.action, result, state)   # extracted values become authoritative
        if decision.action == verification_tool:
            verified, evidence = True, result.data
```

Four properties follow from this structure:

**Verification cannot be skipped.** The `finish` branch is gated on a flag that
only the verification tool can set, and only by succeeding. A planner that tries
to finish early is refused and told why; the refusal enters the observation
buffer, so it can adapt.

**Failure is an observation.** `ToolRegistry.invoke` catches everything,
including a tool crashing, and converts it to a `ToolResult`. The loop never
sees an exception, so a transient browser timeout is something the agent reasons
about rather than something that kills the run.

**Values are bound, not retyped.** When the planner says
`submit_payable(use_extracted=true)`, `_bind()` fills the arguments from working
state. The planner never re-types a figure it read three steps ago, which
removes a whole class of transcription error. It also normalises: an extracted
`73219.0` becomes `"73219.00"` for a form that rejects anything else.

**Uncertainty is routed, not suppressed.** A decision below the confidence floor
is rewritten into an `ask_human` call before it executes.

## Components

| Module | Responsibility |
|---|---|
| `agent/core.py` | The loop, verification gate, state binding, escalation routing. Knows nothing about invoices. |
| `agent/llm.py` | `Planner` protocol, `GeminiPlanner`, `ScriptedPlanner`, tolerant JSON extraction from model output. |
| `agent/tools.py` | `ToolRegistry`, `Tool`, `ToolResult`. Schemas, argument filtering, failure capture. |
| `agent/trace.py` | Append-only run record; flushes after every step so a crashed run is still auditable. |
| `tools/files.py` | Document discovery and field extraction; reports ambiguity instead of resolving it. |
| `tools/browser.py` | Playwright session, form submission, error classification, screenshots, read-back verification. |
| `tools/human.py` | Escalation channel: interactive, preset, or unavailable. |

## Why the registry matters for generalization

A tool is a name, a JSON schema and a callable. The planner picks from
`registry.describe()`, which is generated — so a new capability is registered,
not wired into the loop:

```python
reg.tool("read_invoice", "Extract fields from one invoice...",
         {"type": "object", "properties": {...}})(read_invoice)
```

To point this at a different task — reconciling a statement, filing an expense,
updating a CRM record — you register different tools and change the goal string.
`agent/core.py` does not change. The one task-shaped assumption left in the core
is the default name of the verification tool, which is configuration
(`AgentConfig.verification_tool`), not control flow.

## Trust boundaries

The system is explicit about what it trusts:

| Source | Trusted for | Not trusted for |
|---|---|---|
| Extracted document fields | the values to enter | resolving its own contradictions |
| The planner (LLM) | what to do next | what a number on a page is |
| A success page | that a submission was accepted | that the record persisted correctly |
| A human answer | resolving an ambiguity | — |

The third row is the important one. Verification is a separate read from the
system of record, compared field by field against intent. A submission that
returned a success page but stored the wrong amount fails verification, and the
run does not report success.

## Trace as evidence

Every step appends a typed record (`goal`, `plan`, `tool_call`, `observation`,
`decision`, `escalation`, `verification`, `complete`, `error`) with timings and
artifact references, flushed to disk immediately. This serves three purposes:
debugging during development, evidence returned to the user at the end, and the
audit trail a deployment of this kind would need — who decided what, on what
observation, and what was actually written.
