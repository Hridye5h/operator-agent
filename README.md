# Operator — an autonomous task agent

Takes a natural-language business request, works out what to do, operates a real
browser against a company system that was never designed for an agent, recovers
when things fail, asks a human when it genuinely cannot decide, and **verifies
the outcome by reading it back** before it reports success.

**[▶ Live run viewer](https://hridye5h.github.io/operator-agent/)** — recorded runs,
step by step, planned both by a live model (Gemini) and by the deterministic
planner, with the screenshots and video of the agent's own browser.

```
Goal → Understand → Plan → Execute → Observe → Adapt → Verify → Complete
```

One agent, two different jobs in two different systems — chosen from the request
alone:

> *"Find the latest invoice from Northwind Traders, extract the amount and due
> date, record it as a payable in the internal AP system, and confirm it was
> stored correctly."* → **Accounts Payable**

> *"Onboard Orbit Logistics as a new supplier: register them in the Vendor Master
> from their registration form and confirm the record."* → **Vendor Master**

The second task was added without changing the agent loop or the planner — a new
procedure file, new tools and a new form. A test enforces that the framework code
names no task at all (see [Generalization](#generalization)).

---

## Quick start

```bash
pip install -r requirements.txt
python -m playwright install chromium
python mockenv/invoices/generate_invoices.py        # invoice corpus
python mockenv/vendors/generate_vendor_forms.py     # supplier registration corpus
python run_agent.py --scenario happy
```

No API key is required for any scenario except `batch` — see
[Running without an API key](#running-without-an-api-key).

### Scenarios

| Command | What it demonstrates |
|---|---|
| **Accounts Payable** | |
| `--scenario happy` | End-to-end autonomy, including recovery from a transient system outage |
| `--scenario ambiguous` | The invoice contradicts itself; the agent escalates instead of guessing |
| `--scenario no_human` | Same ambiguity, nobody available — the agent **stops** rather than picking |
| `--scenario high_value` | Payable over the approval limit; the agent gets sign-off before writing |
| `--scenario declined` | Approval refused — nothing is written, and the human is not asked twice |
| `--scenario duplicate` | Runs the task twice; the second pass confirms rather than double-entering |
| `--scenario batch` | Every Northwind invoice in one run — a shape no procedure describes, so it needs the LLM planner. The run cannot finish until each record written has been read back |
| **Vendor Master** — same agent, different procedure | |
| `--scenario supplier` | Registers a supplier from their signed form and verifies all six fields |
| `--scenario supplier_fraud_check` | A bank-account change arrived by email — the classic payment-fraud pattern. The agent refuses to pick and asks; a person confirms the signed-form account |
| `--scenario supplier_no_human` | Same fraud pattern, nobody to ask — no vendor is created with either account |

Useful flags: `--headed` (watch the browser), `--record-video` (save the browser
session as `run.webm`), `--interactive` (answer escalations yourself),
`--goal "..."` (any request), `--planner gemini|scripted`, `--approval-limit 50000`
(0 disables the approval policy).

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
- `artifacts/run.webm` — the whole browser session, with `--record-video`

---

## Why it is built this way

**The loop reasons about kinds and schemas, never names.** Every tool declares a
*kind* — `discover`, `read`, `write`, `verify`, `ask` — and a parameter schema.
A reader returns `fields`; a writer declares the fields it accepts; the loop
binds one to the other by name. It has no fixed list of fields and no tool names,
so a new task is new tools plus a goal. `tests/test_generalization.py` fails the
build if a task name ever appears in the framework code.

**Five guarantees live in the loop, not in the prompt.** Prompts are advice; a
weaker or misbehaving planner must not be able to get around these:

| Guarantee | What it stops |
|---|---|
| A run cannot finish until every record it wrote has been read back by a verify tool | reporting success on the strength of a success page, or after checking one record of a batch |
| Verification compares every field that was actually written | a planner verifying less than it wrote |
| A contested value cannot be written until a human resolves it — and then only as the value they chose | a planner — or a model — guessing between two figures, or reading another document to make the question go away |
| What is written must match a document read in the run, field for field | a mis-copied figure being written, then verified against itself |
| A declined action is never executed or re-asked | interrogating a person until they give in |

**Company procedures are data.** How the organisation does each kind of request —
which tool plays each role, and the house rules — lives in
`company/procedures/*.json`. The agent selects one from the goal and shows it to
whichever planner runs. An LLM reads it as guidance; the deterministic planner
executes it.

**The browser finds fields the way a person does.** Fields are located by their
visible label, buttons by their visible name, and outcomes by ARIA roles (`alert`
for errors, `status` for confirmations). Each field is read back after typing, so
a form that truncates or reformats a value is caught before anything is
submitted. No element ids, so the same code
operates any form a person could fill in.

**It writes through the browser, reads through the API.** The mock system
exposes a JSON API, and the agent deliberately does not use it to write. Entry
goes through Playwright filling the same HTML form a person would, because
operating software that was never built for an agent is the actual problem. The
API is used only for verification — standing in for the independent read-back a
real deployment would do.

**Verification is enforced by the loop, not by the planner's good intentions.**
A run physically cannot return `completed` until every record it wrote has been
re-read from the system of record and compared field by field — each one against
the values of its own write. If the planner tries to finish early it is refused
and told which writes are still unchecked:

```python
if decision.action == "finish":
    pending = [w for w in writes if not w["verified"]]
    if self.config.require_verification and (not verified or pending):
        observations.append(f"SYSTEM: {nudge}")    # names each unverified write
        continue
```

A success page is not proof. Something that looked submitted but did not persist
is caught here.

**Failure is data, not an exception.** Tools never raise into the loop; they
return a `ToolResult` carrying `ok`, an error message, and `retriable`. The
agent uses that flag to decide between retrying unchanged and changing approach
— retrying a validation rejection unchanged would just fail again. A write that
crashes partway is reported as *outcome unknown*, not retriable: it may have
landed, so it counts as unverified until the record has been read back.

**Extraction is deterministic; planning is not.** A language model is the right
tool for deciding what to do next. It is the wrong tool for reading a number off
a page, where a parser that fails loudly beats a model that quietly rounds a
figure. So the LLM plans, and extraction is explicit parsing that reports gaps
rather than inventing values.

**Escalation is a first-class outcome.** When a document contradicts itself —
two totals on an invoice, or a supplier's bank account changed by email — the
reader reports the field as contested and the agent asks. A reply only resolves
it if it names one of the candidates; "see the email from 15 March" is guidance,
not an amount. If nobody answers, the run ends as `escalated` with nothing
written. Doing nothing is the correct result there.

**Approval is policy, enforced by the loop.** A tool can declare
`requires_approval`, and a configurable policy can gate any individual call —
the demo requires human sign-off for payables over ₹100,000. The loop consults
the policy; the policy decides the rule, so no money logic lives in the agent. A
refusal is remembered: a declined action is never executed, never re-escalated
to the same person, and a planner that keeps proposing it is stopped rather than
allowed to spin. Only a plain yes approves — "you decide" or "yes, but check
with finance first" is a refusal. A policy that throws fails closed.

---

## Running without an API key

With `GEMINI_API_KEY` set, planning is done by Gemini. Without one, the system
falls back to `ScriptedPlanner` and says so:

```
[01] PLAN planner downgraded: no GEMINI_API_KEY configured — using deterministic planner
```

The fallback is **not** a recording of a successful run. It executes the company
procedure the agent selected, over the same observation buffer and tool registry,
and takes every branch at runtime from what it has actually seen — choosing the
document by matching the request against what discovery returned, noticing a
contested value, escalating, and telling a retriable failure from a conflict from
a rejection. It reasons with rules instead of a model, contains no task-specific
names, and only covers documented procedures; a request with no matching
procedure needs the LLM planner.

The Gemini path is covered without a key by stubbing the HTTP client: the tests
exercise the real planner against fenced, prose-wrapped, truncated and
safety-blocked replies, and `test_integration_llm.py` drives a complete run from
model output through the real browser to the verification read-back. A reply the
agent cannot parse is re-prompted with the parse error rather than ending the
run, because a model will occasionally answer with prose.

It exists so that the loop, the recovery paths and the verification step are all
testable and demonstrable without a network call. To use the real planner:

```bash
cp .env.example .env                  # add GEMINI_API_KEY (aistudio.google.com/apikey)
python scripts/check_gemini.py        # preflight: key, available models, one test call
python run_agent.py --scenario happy --planner gemini
```

`.env` is loaded at startup but never overrides a variable already set in the
shell. If `GEMINI_MODEL` is not set, the planner asks the API which models the key
can use and picks the first known-good one, so a retired model name does not
break a run. Every request carries a 60-second deadline. Per-minute rate limits
and transient server errors are waited out, using the server's own retry delay
when it sends one. Quotas and load are per model: when the model was chosen
automatically, a used-up daily quota, an overloaded model or an expired deadline
moves the planner to the next model the key can use, and the trace records the
switch. A model set with `GEMINI_MODEL` is never swapped.

`python scripts/record_llm_demos.py` records model-driven runs, with video, into
the run viewer. Runs are kept as they happened — a run that escalates or fails is
recorded, not retried until it looks good.

---

## Generalization

The second task — onboarding a supplier into the Vendor Master — was added with:

| Added | Changed |
|---|---|
| `company/procedures/onboard_supplier.json` | `agent/core.py` — **nothing** |
| `tools/vendor_forms.py` (reader for a new document type) | `agent/llm.py` — **nothing** |
| `register_vendor` / `verify_vendor` (a field-to-label map each) | `tools/browser.py` perception — **nothing** |
| a Vendor Master form in the mock environment | |

The framework is held to that by a test rather than a promise.
`test_framework_code_names_no_task` scans the loop, the planner, the registry and
the procedure loader for any tool, field or vendor name, and fails if one
appears.

**Beyond one shape:** both procedures share the discover → read → write →
verify shape, and the deterministic planner executes only that. The `batch`
scenario is a different shape — several records in one run — and needs the LLM
planner; the loop holds the run open until every record written has been read
back, each against its own values. A task such as reconciling two systems would
run the same way.

---

## The environment

`mockenv/` is a stand-in company with two systems — Accounts Payable and the
Vendor Master — built to misbehave in the ways real systems do:

| Failure mode | Trigger | Correct agent response |
|---|---|---|
| `SYS-503` transient outage | first submission of each process | retry unchanged |
| strict formats | `INR 73,219.00`; a malformed GSTIN, PAN or IFSC | reform the value or ask |
| `VND-004` consistency check | PAN does not match the one inside the GSTIN | do not retry unchanged |
| duplicate guards | invoice number or GSTIN already on file | confirm, do not double-enter |

These are pinned by tests in `tests/test_internal_app.py` so the agent's
recovery paths are exercised against something that genuinely fails, rather than
asserted against a mock that always succeeds.

The six invoices use three layouts with different field labels (`Invoice Number`
/ `Reference` / `Document`), so extraction cannot be one regex; one states two
different totals. Of the three supplier forms, one carries a bank-account change
received by email, and one has an account number with leading zeros, which must
survive extraction, the form, the database and verification as a string.

---

## Tests

```bash
python -m pytest tests/ -q                # 222 tests, a few seconds
python -m pytest tests/ -m integration    # + 8 runs against the real app and browser (~2 min)
```

| File | Covers |
|---|---|
| `test_agent_loop.py` | Loop invariants: verification cannot be skipped, every write in a batch is read back against its own values, failed verification is not success, retriable vs permanent, tool crashes survive, escalation stops the run, low confidence reroutes to a human, declined actions are never executed or re-asked, a broken policy fails closed |
| `test_generalization.py` | A misbehaving planner cannot write a contested value or verify less than it wrote; one deterministic planner executes both procedures; overlapping labels and negations in a human reply are not misread; the framework names no task |
| `test_extraction.py` | All six invoices against ground truth, the columnar-layout regression, ambiguity reported not resolved |
| `test_planner.py` | Parsing what a model actually sends (fenced, prose-wrapped, truncated, blocked), the Gemini request shape, how the agent re-prompts an unusable reply, rate-limit waits and the per-model quota fallback |
| `test_internal_app.py` | The environment's failure modes behave as specified |
| `test_trace.py` | The run record survives what a model writes: characters the console cannot print, multi-line steps, a run killed mid-write |
| `test_integration_browser.py` | Label-based form filling and result reading against the real app; leading zeros through the whole stack; the fraud pattern with and without a person to ask (marked `integration`) |
| `test_integration_llm.py` | A full run driven by LLM-shaped replies against the real browser, server and verification read-back (marked `integration`) |

The loop tests use stub tools, so they assert the agent's guarantees directly
and run in milliseconds.

---

## Layout

```
agent/              core.py (the loop) · llm.py (planners) · tools.py (registry, kinds)
                    procedures.py (selecting and checking procedures) · trace.py
company/procedures/ record_vendor_invoice.json · onboard_supplier.json
tools/              files.py (invoices) · vendor_forms.py (supplier forms)
                    browser.py (label-based perception) · human.py (escalation)
mockenv/            internal_app/ (FastAPI + SQLite + HTML forms, two systems)
                    invoices/ · vendors/ (generated PDFs)
runs/<run-id>/      trace.json · report.md · artifacts/*.png · artifacts/run.webm
docs/               ARCHITECTURE.md · DECISIONS.md · LIMITATIONS.md · COMPONENTS.md
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design,
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the trade-offs and what was
rejected, [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for what this does not do,
the assumptions it makes and what would come next, and
[`docs/COMPONENTS.md`](docs/COMPONENTS.md) for every model, library and external
service used.
