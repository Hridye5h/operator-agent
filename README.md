# Operator — an autonomous task agent

Takes a natural-language business request, works out what to do, operates a real
browser against a company system that was never designed for an agent, recovers
when things fail, asks a human when it genuinely cannot decide, and **verifies
the outcome by reading it back** before it reports success.

Built for the CentrAlign AI engineering problem statement.

```
Goal → Understand → Plan → Execute → Observe → Adapt → Verify → Complete
```

The demonstrated task is invoice-to-ledger entry:

> *"Find the latest invoice from Northwind Traders, extract the amount and due
> date, record it as a payable in the internal AP system, and confirm it was
> stored correctly."*

---

## Quick start

```bash
pip install -r requirements.txt
python -m playwright install chromium
python mockenv/invoices/generate_invoices.py   # build the document corpus
python run_agent.py --scenario happy
```

No API key is required — see [Running without an API key](#running-without-an-api-key).

### The three scenarios

| Command | What it demonstrates |
|---|---|
| `python run_agent.py --scenario happy` | End-to-end autonomy, including recovery from a transient system outage |
| `python run_agent.py --scenario ambiguous` | The document contradicts itself; the agent escalates instead of guessing |
| `python run_agent.py --scenario no_human` | Same ambiguity, nobody available — the agent **stops** rather than picking |

Useful flags: `--headed` (watch the browser), `--interactive` (answer escalations
yourself), `--goal "..."` (any task), `--planner gemini|scripted`.

---

## What a run looks like

```
[03] THINK Need to see what documents exist.
[04]  ->   list_invoices()
[05]  <-   list_invoices: 6 document(s): ...
[06] THINK Read the most recent Northwind invoice.
[07]  ->   read_invoice(vendor='Northwind', latest=True)
[08]  <-   read_invoice: NW-2026-0671, due 2026-07-22, amount 73219.0
[09] THINK Enter the payable into the internal system.
[10]  ->   submit_payable(vendor='Northwind Traders', ... amount='73219.00' ...)
[11]  <-   submit_payable: ERROR (retriable): SYS-503 accounting service unavailable
[12] THINK Previous submission failed transiently; retrying.
[13]  ->   submit_payable(...)
[14]  <-   submit_payable: submitted OK -> record #1
[15] THINK Confirm the record actually landed and matches.
[16]  ->   verify_payable(invoice_number='NW-2026-0671', expect_amount='73219.00', ...)
[17]  <-   verify_payable: VERIFIED record #1: 73219.00, due 2026-07-22
[19] DONE  Outcome verified.
```

Nothing in that sequence is scripted. The failure at step 11 is real, the
classification at step 12 is made from the error the system returned, and step
17 is a second, independent read of the database.

Every run writes to `runs/<run-id>/`:

- `trace.json` — every step, tool call, argument, observation and timing
- `report.md` — the same run as a readable table
- `artifacts/*.png` — browser screenshots at each submission

---

## Why it is built this way

**The loop is task-agnostic.** `agent/core.py` knows about goals, observations,
tools and verification. It contains no mention of invoices. Task knowledge lives
in the tools and in the goal string, which is what lets the same loop drive a
different task. Adding a capability means registering a tool, not editing the
agent.

**It writes through the browser, reads through the API.** The mock system
exposes a JSON API, and the agent deliberately does not use it to write. Entry
goes through Playwright filling the same HTML form a person would, because
operating software that was never built for an agent is the actual problem. The
API is used only for verification — standing in for the independent read-back a
real deployment would do.

**Verification is enforced by the loop, not by the planner's good intentions.**
A run physically cannot return `completed` without having passed a verification
step that re-read the record from the system of record and compared it field by
field. If the planner tries to finish early it is refused and told why:

```python
if decision.action == "finish":
    if self.config.require_verification and not verified:
        observations.append("SYSTEM: Refused to finish: the outcome has not been verified.")
        continue
```

A success page is not proof. Something that looked submitted but did not persist
is caught here.

**Failure is data, not an exception.** Tools never raise into the loop; they
return a `ToolResult` carrying `ok`, an error message, and `retriable`. The
agent uses that flag to decide between retrying unchanged and changing approach
— retrying a validation rejection unchanged would just fail again.

**Extraction is deterministic; planning is not.** A language model is the right
tool for deciding what to do next. It is the wrong tool for reading a number off
a page, where a parser that fails loudly beats a model that quietly rounds a
figure. So the LLM plans, and extraction is explicit parsing that reports gaps
rather than inventing values.

**Escalation is a first-class outcome.** When a document states two
contradictory totals, the extractor does not pick one — it reports an ambiguity
and the agent asks. If nobody answers, the run ends as `escalated` with nothing
written. Doing nothing is the correct result there.

---

## Running without an API key

With `GEMINI_API_KEY` set, planning is done by Gemini. Without one, the system
falls back to `ScriptedPlanner` and says so:

```
[01] PLAN planner downgraded: no GEMINI_API_KEY configured — using deterministic planner
```

The fallback is **not** a recording of a successful run. It reasons over the same
observation buffer and the same tool registry, and takes every branch at runtime
from what it has actually seen — including noticing the ambiguity, choosing to
escalate, and distinguishing a retriable failure from a permanent one. It reasons
with rules instead of a model.

It exists so that the loop, the recovery paths and the verification step are all
testable and demonstrable without a network call. To use the real planner:

```bash
cp .env.example .env    # add GEMINI_API_KEY
python run_agent.py --scenario happy --planner gemini
```

---

## The environment

`mockenv/` is a stand-in company, built to misbehave in the three ways real
systems do:

| Failure mode | Trigger | Correct agent response |
|---|---|---|
| `SYS-503` transient outage | first submission of each process | retry unchanged |
| `VAL-003` strict amount format | `INR 73,219.00` instead of `73219.00` | reform the value |
| `VAL-006` duplicate guard | invoice already recorded | stop, do not double-enter |

These are pinned by tests in `tests/test_internal_app.py` so the agent's
recovery paths are exercised against something that genuinely fails, rather than
asserted against a mock that always succeeds.

The six invoices use three different layouts with different field labels
(`Invoice Number` / `Reference` / `Document`), so extraction cannot be one regex.
One of them is internally contradictory on purpose.

---

## Tests

```bash
python -m pytest tests/ -q      # 29 tests, ~4s
```

| File | Covers |
|---|---|
| `test_agent_loop.py` | Loop invariants: verification cannot be skipped, failed verification is not success, retriable vs permanent, tool crashes survive, escalation stops the run, low confidence reroutes to a human |
| `test_extraction.py` | All six invoices against ground truth, the columnar-layout regression, ambiguity reported not resolved |
| `test_internal_app.py` | The environment's three failure modes behave as specified |

The loop tests use stub tools, so they assert the agent's guarantees directly
and run in milliseconds.

---

## Layout

```
agent/            core.py (the loop) · llm.py (planners) · tools.py (registry) · trace.py
tools/            files.py (documents) · browser.py (Playwright) · human.py (escalation)
mockenv/          internal_app/ (FastAPI + SQLite + HTML forms) · invoices/ (generated PDFs)
runs/<run-id>/    trace.json · report.md · artifacts/*.png
docs/             ARCHITECTURE.md · DECISIONS.md · LIMITATIONS.md
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design,
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the trade-offs and what was
rejected, and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for what this does
not do and what would come next.
