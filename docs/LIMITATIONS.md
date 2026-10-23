# Limitations, assumptions, and what comes next

## What this does not do

**It operates one application.** The browser tool knows how to fill this form.
Pointing it at an arbitrary website would require a general perception layer —
reading the DOM or a screenshot, locating fields by intent rather than by `#id`.
That is the single biggest gap between this and the stated target, and the work
is real: element grounding, scrolling, dynamic content, iframes, auth.

**No desktop or OS-level control.** Browser and files only. No native
applications.

**Extraction is pattern-based.** Three layouts are handled. A fourth with
different labels needs a new pattern. It reports gaps rather than guessing, so
failures are loud — but they are still failures.

**Planning is one step at a time.** There is no plan artifact that persists
across turns and gets revised; the agent re-decides from the observation buffer
each turn. That is robust to surprise and weak at multi-stage tasks where an
early choice constrains a later one.

**No parallelism.** Strictly sequential. Ten invoices means ten serial runs.

**Memory is per-run.** Nothing is retained between runs — no record of how this
company formats dates, which vendor amends invoices, or which forms are flaky.
Each run starts cold.

**The step budget is a blunt instrument.** `max_steps` caps a run. A genuinely
long task would hit the ceiling and report `exhausted` rather than progressively
summarising and continuing.

---

## Assumptions

- The documents are text-based PDFs. No OCR; scans would need a vision model.
- The internal system is reachable at `127.0.0.1:8099`, single-tenant, no auth.
  Real credentials, SSO and session expiry are out of scope.
- Invoice numbers are unique per vendor and usable as the verification key.
- `issued` date determines "latest". Ties are not resolved.
- Amounts are a single currency; no FX handling.
- The human answering an escalation is authorised to make that call. There is no
  approval hierarchy.
- `GEMINI_API_KEY` is optional; the deterministic planner covers its absence.

---

## Known rough edges

- `ScriptedPlanner` matches on substrings of the history buffer. It is readable
  and testable, but it is a rule engine — a genuinely novel situation falls
  through to `finish`, where the verification gate catches it.
- Screenshots are full-page PNGs with no redaction. Real deployment would need
  masking before anything is persisted.
- The SQLite ledger is wiped per run by default (`--scenario` resets it), so
  duplicate-detection across runs is only exercised in tests.
- Browser startup dominates runtime (~2–3s of a ~7s run).

---

## What I would build next, in order

**1. A perception layer for arbitrary forms.** Replace hardcoded selectors with:
read the accessibility tree, ask the model to map intent to elements, act, and
confirm the field holds what was typed. This is what turns "operates this app"
into "operates software". Everything below depends on it.

**2. Persistent company memory.** A store of what the agent learned — vendor
quirks, form conventions, which fields that flaky system rejects — keyed by
company and surfaced into the planning context. Today every run re-discovers
that the amount field hates commas.

**3. A revisable plan.** Keep a plan as a first-class artifact with steps that
can be marked done, failed or re-planned. One-step-at-a-time is robust but has
no notion of progress, which is what makes long tasks hard.

**4. Richer approval routing.** Approval itself is implemented — tools can
declare `requires_approval`, a configurable policy gates individual calls on
value, refusals are remembered and never re-asked, and a broken policy fails
closed. What is missing is *who* approves: today the question goes to whoever is
at the terminal. Real use needs approvals routed to a named role, with
delegation, timeouts and an audit record of who authorised what.

**5. An evaluation harness.** A scenario suite scoring autonomy, verification
rate, escalation precision and recovery rate, run on every change. Without it,
"more reliable" is an opinion. This is the piece I would want most before
trusting it with anything real.

**6. Parallel execution with a shared ledger.** Fan out across documents with
idempotency on invoice number, so a retry after a partial failure cannot
double-enter.
