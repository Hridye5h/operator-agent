# Architecture

## Shape

```
                 ┌──────────────────────────┐
   goal ────────▶│ select company procedure │◀── company/procedures/*.json
                 └────────────┬─────────────┘
                              ▼
                    ┌──────────────────┐
                    │    Agent loop    │◀──────── observations
                    │  (agent/core.py) │
                    └────────┬─────────┘
                             │ one action per turn
                    ┌────────▼─────────┐
                    │     Planner      │   Gemini, or a deterministic
                    │  (agent/llm.py)  │   procedure executor
                    └────────┬─────────┘
                             │ Decision{action, args, confidence}
          ┌──────────────────▼───────────────────┐
          │   guards: contested? approval?        │   enforced here, whatever
          │   bind: schema-driven, from state     │   the planner decided
          └──────────────────┬───────────────────┘
                    ┌────────▼─────────┐
                    │   ToolRegistry   │   name → kind → schema → callable
                    └────────┬─────────┘
       ┌─────────────┬───────┴──────┬──────────────┬─────────────┐
       ▼             ▼              ▼              ▼             ▼
   discover        read           write          verify         ask
   list docs    extract fields   browser,      independent    escalation
                + contested       by label      read-back
       └─────────────┴──────────────┴──────────────┴─────────────┘
                                    ▼
                          Trace (append-only)
             trace.json · report.md · screenshots · run.webm
```

## The loop

`Agent.run(goal, trace, procedure)` holds the goal, an observation buffer, a
working state, and every write it has made. Each turn:

```
decision = planner.decide(SYSTEM_PROMPT, context(goal, procedure, tools, state, history))

if decision.action == "finish":
    if any write is not yet verified: refuse, name it, continue    # gate 1
    return completed

if decision.confidence < floor: decision = ask_human(...)

tool = registry.get(decision.action)
target = the write a verify call names (else the latest unverified one)
args = bind(decision, tool, state, target)                         # schema-driven

if tool.kind == "write" and a field it sets is contested on that   # gate 2
        record (unresolved, or not the value the human chose):
    refuse, tell the planner why, continue
if tool.kind == "write" and no document read in this run gives     # gate 3
        every value it sets that a reader supplies:
    refuse, tell the planner which values and what the document says, continue
if approval required and not granted:                              # gate 4
    ask; on decline remember it, never execute

result = registry.invoke(...)          # never raises
absorb(tool.kind, result, state)       # read → fields / contested; ask → resolution
if tool.kind == "write" and result.ok: writes.append(args)
if tool.kind == "verify" and result.ok: mark target verified       # with coverage
```

## Kinds, not names

Every tool declares one of six kinds, and the loop reasons only in kinds:

| Kind | Loop behaviour |
|---|---|
| `discover` | none beyond recording the observation |
| `read` | absorbs `fields` into working state; records a `contested` field and removes its value |
| `write` | subject to the contested-value guard; a successful write joins the list that must be verified |
| `verify` | expectations bound from the write it names (or the latest unverified one); success marks that write verified |
| `ask` | a reply naming one candidate resolves the contest; anything else is guidance |
| `other` | none |

Binding is driven by the tool's declared parameters. A parameter `x` is filled
from the established value `x`; a verify tool's `expect_x` from the value of `x`
that was actually written. There is no list of fields anywhere in the loop, which
is why a second task needed no changes to it. `test_framework_code_names_no_task`
holds the framework to that.

## The five guarantees

These are enforced in `core.py`, not requested in the prompt, so they hold for
any planner — including a model that misbehaves:

**1. No success without verification.** `finish` is refused until every
successful write has been read back by a verify-kind tool. The refusal names the
unchecked writes and enters the history so the planner can adapt.

**2. Verification covers what was written.** A verify tool's expectations come
from the arguments of the write it names, overriding whatever the planner
passed. The trace records `verified_fields` and `unverifiable_fields`.

**3. No guessing a contested value.** A contest belongs to the record it was
read from and lasts the run. No write to that record may set the field — blank or
filled in — until a human reply names exactly one candidate, and afterwards only
with that value.

**4. What is written is what was read.** Every write value that some reader
supplies must agree with one document read in the run — the same document for
all of them. Verification proves the record matches the write; this proves the
write matches a source. Money may drop trailing zeros; identifiers may not lose
leading ones. A human's choice on a contested field counts as the document's
value.

**5. A refusal is final.** A declined action is never executed, the same person
is never asked twice, and a planner that keeps proposing it is stopped. Only a
plain yes approves.

And one property, rather than a guard: **failure is an observation**.
`ToolRegistry.invoke` catches everything and returns a `ToolResult` with
`error_kind` — `transient` (retry it), `conflict` (it already exists; confirm it)
or `invalid` (change the input). A planner reply that cannot be used is
re-prompted with the reason, the same way; a planner that cannot answer at all
ends the run.

## Company procedures

`company/procedures/*.json` describe how this organisation does each kind of
request:

```json
{
  "name": "onboard_supplier",
  "triggers": ["vendor", "supplier", "onboard", "vendor master"],
  "steps": {"discover": "list_vendor_forms", "read": "read_vendor_form",
            "write": "register_vendor", "verify": "verify_vendor"},
  "rules": ["A bank-detail change received by email is a known fraud pattern: ..."]
}
```

`select_procedure` matches triggers against the goal and returns none on a tie.
`check_procedure` confirms each step names a registered tool of the right kind.
The procedure is rendered into the planner's context: the LLM reads it as
guidance, and `ScriptedPlanner` executes it — choosing the document by matching
the request against what discovery returned, asking on a contested value,
retrying on a transient failure, confirming on a conflict.

## Perception

`BrowserSession.submit_form(path, {label: value}, submit, tag)` is the only
browser code a write tool needs:

1. Each field is located by its visible label. A label that matches zero or
   several fields is an error, not a guess.
2. The button is pressed by its visible name.
3. An `alert` region is an error, classified from its text; a `status` region is
   a confirmation.
4. Result rows are read from the document text by header. Rendered text applies
   CSS, and an uppercasing stylesheet broke every lookup once — see
   `DECISIONS.md` §20.

`register_vendor` is therefore a field-to-label map and nothing else.

## Components

| Module | Responsibility |
|---|---|
| `agent/core.py` | The loop, the gates, schema-driven binding, absorbing results, resolving a reply against candidates. Names no task. |
| `agent/llm.py` | `Planner` protocol, `GeminiPlanner`, `ScriptedPlanner` (procedure executor), tolerant JSON extraction. Names no task. |
| `agent/tools.py` | `ToolRegistry`, `Tool` (with kind), `ToolResult` (with `error_kind`). |
| `agent/procedures.py` | Loading, selecting and checking company procedures. |
| `agent/trace.py` | Append-only run record, flushed every step. |
| `tools/files.py` | Invoice discovery and extraction; reports a contested amount. |
| `tools/vendor_forms.py` | Supplier-form discovery and extraction; reports a contested bank account. |
| `tools/browser.py` | Label-based perception, error classification, read-back verification, screenshots, video. |
| `tools/human.py` | Escalation channel: interactive, preset, or unavailable. |

## Trust boundaries

| Source | Trusted for | Not trusted for |
|---|---|---|
| Extracted fields | the values to write | resolving the document's own contradictions |
| The planner (LLM or rules) | what to do next | what a value is; whether the work is done; what to verify |
| A success page | that a submission was accepted | that the record persisted, or with what values |
| A human reply | choosing between named candidates | supplying a figure the document never stated |

## Trace as evidence

Every step appends a typed record (`goal`, `plan`, `tool_call`, `observation`,
`decision`, `escalation`, `verification`, `complete`, `error`) with timings,
error kinds, verification coverage and artifact references, flushed to disk
immediately. It serves debugging, the evidence returned at the end, and the audit
trail this kind of system needs — who decided what, on what observation, and what
was actually written.
