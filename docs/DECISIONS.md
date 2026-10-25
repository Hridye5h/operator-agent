# Design decisions

Each entry: the choice, the alternative, and why.

---

### 1. Write through the browser, read through the API

The mock system has a perfectly good JSON API. The agent does not use it to
write.

**Rejected:** `POST /api/payables`. It would be faster, more reliable, and would
prove nothing — the actual problem is operating software that was never designed
for an agent. An agent that only works where an API already exists does not
address the brief.

So writes go through Playwright filling the same HTML form a person would, with
all the fragility that implies. The API is used only for verification, standing
in for the independent read-back a real deployment would do against a database
or a reporting endpoint.

---

### 2. Verification is enforced by the loop, not requested in the prompt

The system prompt tells the planner to verify before finishing. Prompts are
advice, not guarantees, so the loop also refuses:

```python
if decision.action == "finish" and not verified:
    observations.append("SYSTEM: Refused to finish: the outcome has not been verified.")
    continue
```

**Rejected:** trusting the instruction. A model that forgets, or decides the
success page was enough, would produce a confidently wrong "done". Making it a
control-flow invariant means no planner — including a worse future one — can
report success on an unverified outcome. `test_cannot_finish_without_verifying`
pins this.

---

### 3. Verification re-reads from the system of record

After submitting, the agent queries the database for that invoice number and
compares the stored amount and due date against what it intended to write.

**Rejected:** treating the success page as confirmation. The page is rendered
from the request, not from what was persisted. A write that was accepted and then
rolled back, truncated, or silently coerced would still render a success banner.
`test_failed_verification_does_not_count_as_success` covers the mismatch case.

---

### 4. Deterministic extraction, model-driven planning

The LLM decides what to do next. It does not read numbers off pages.

**Rejected:** LLM extraction. Pulling a total off an invoice is exactly where a
model's failure mode is worst — it will return something plausible and
well-formatted rather than admitting it is unsure, and a quietly rounded figure
in a ledger is far more damaging than a loud parse failure. Explicit parsing
either finds the field or reports it in `missing_fields`.

The cost is generality: a fourth invoice layout needs a new label pattern. That
is the right trade for the demonstrated task, and the honest upgrade path is a
model fallback *when deterministic extraction reports a gap* — the gap is already
detected and reported, so the hook exists.

---

### 5. Ambiguity is surfaced, not resolved

`globex_GX-90155.pdf` states a printed total of 118,590.00 and an amended total
of 96,400.00. The extractor returns neither. It returns both, flagged.

**Rejected:** heuristics like "prefer the amended figure" or "prefer the larger".
Both are guesses dressed as logic, and either could put the wrong number in a
ledger. The agent escalates with both candidates, and if nobody answers, the run
ends with nothing written.

This is why `--scenario no_human` exists as a demonstrated outcome rather than an
error: refusing to act is sometimes the correct result, and a system that cannot
do it is not safe to run unattended.

---

### 6. Retriable vs permanent failure are distinguished at the tool boundary

`ToolResult.retriable` is set by the tool that understands the error — the
browser tool classifies `SYS-5xx` as transient and `VAL-xxx` as a rejection of
the input.

**Rejected:** a blanket retry policy. Retrying a validation rejection unchanged
just fails again and burns the step budget; not retrying a transient outage
fails a task that would have worked. The distinction has to be made where the
error semantics are known, then passed to the planner as a hint.

---

### 7. Extracted values bind into later calls instead of being retyped

`submit_payable(use_extracted=true)` is resolved from working state by `_bind()`.

**Rejected:** having the planner copy values into its arguments. Every retype is
a chance to drop a digit, and a model that has to echo `73219.0` through three
turns will eventually echo it wrong. Binding also handles normalisation
centrally — the form rejects `73219.0` and wants `73219.00`.

---

### 8. An explicit loop rather than an agent framework

The loop is ~150 lines in `agent/core.py`.

**Rejected:** LangGraph, which I have used before (RepoSage) and would reach for
on a larger graph. Here the control flow *is* the submission: the verification
gate, the confidence reroute and the state binding are the interesting parts,
and burying them in framework callbacks would make them harder to read, harder
to test, and harder to defend. The loop is also what makes the stub-tool tests
possible — they drive it directly with a `SequencePlanner`.

---

### 9. A deterministic planner as a first-class fallback

Without an API key the system runs `ScriptedPlanner` and announces the downgrade.

**Rejected:** failing without a key, or shipping a recorded transcript. The
fallback reasons over the same observation buffer and takes every branch at
runtime — including spotting the ambiguity and classifying a retriable failure.
It reasons with rules instead of a model.

It exists so the loop, the recovery paths and the verification step are provable
without a network call, and so a reviewer with no key still sees the real system
rather than a video of it. It is not pretending to be an LLM; `planner=scripted`
is printed on every run.

---

### 10. The environment is built to misbehave

The mock system fails the first submission of every process, rejects formatted
amounts, and guards against duplicates — and those behaviours are pinned by
tests.

**Rejected:** an environment that always succeeds, with failure paths asserted
by unit tests alone. "Handles retries" is cheap to claim and easy to fake with a
`try/except` that never fires. Here the transient failure happens on every single
run, so the recovery is visible in the trace and in the demo rather than being
taken on trust.

---

### 11. Approval is enforced by the loop; the rule lives outside it

`Tool.requires_approval` gates a whole tool. `AgentConfig.approval_policy` gates
an individual call — the demo refuses to write a payable over ₹100,000 without
sign-off.

**Rejected:** putting the threshold in the agent, or in the planner's prompt.
Money rules change, and they are not the agent's business. The loop enforces
*that* a policy is consulted before every tool call; the policy decides *what*
the rule is. A policy that raises is treated as a refusal, so a broken policy
service cannot wave a payment through.

**Rejected also:** trusting the prompt to mention approval. The tool catalogue
advertises `[NEEDS APPROVAL]`, but an advertisement is not a control — before
this was wired up, a planner that ignored the label would have written anyway.

---

### 12. A refusal is remembered

The first implementation re-asked on every attempt. A planner that would not
take no for an answer produced **twelve** approval prompts in one run before the
step budget ran out.

That is a real failure even though nothing was written: a system that
interrogates a person until they give in has defeated the point of asking. Now a
decline is recorded against the action and its arguments, the same question is
answered from memory, and a planner that keeps proposing the refused action is
stopped with nothing written. `test_a_refusal_is_not_re_escalated_to_the_human`
pins the count at one.

---

### 13. An unusable model reply is a bad turn, not a dead run

A planner exception used to end the run immediately. That was fine while the
only planner was deterministic; a real model answers with prose, a truncated
object, or nothing at all often enough that it cannot be fatal.

The loop now re-prompts with the parse error attached — *"your previous reply
could not be used: ..."* — and only gives up after the retry budget. The same
principle the tools follow: failure is an observation.

**Rejected:** treating an empty response as a parse failure. An empty body
usually means the prompt was safety-blocked or the reply hit the token limit,
and re-prompting cannot fix either. `GeminiPlanner` inspects `prompt_feedback`
and `finish_reason` and reports which it was, so the trace says *"prompt
blocked: SAFETY"* rather than a misleading JSON error.

---

### 14. The LLM path is tested by injecting the client, not by mocking the planner

`GeminiPlanner` takes an optional `client`, so the tests exercise the real
planner — the request it builds, the config fields it sets, the response shapes
it must survive — with no key and no network.

**Rejected:** mocking `GeminiPlanner` itself. That would have tested the mock.
The defect this guards against is precisely a wrong SDK call or a response shape
nobody tried, and a mocked planner cannot catch either.

`test_integration_llm.py` closes the loop: model-shaped replies, including
fenced and prose-prefixed ones, drive the real browser against the real server
through to the real verification read-back.

---

### 15. A contested value cannot be written — enforced by the loop

"Escalates rather than guessing" used to be true because the deterministic
planner was written to ask. Nothing in the loop required it. A probe with a
planner that ignored the ambiguity and passed the printed total itself got the
guess **written and verified**: the unresolved-value guard only checked for a
*blank* amount, and verification compared fewer fields because there was no
extracted amount to expect.

Now a reader that finds a contradiction reports the field as `contested`, with
its candidates. While it is contested, the loop refuses any write that sets that
field — blank or filled in — and only a human reply naming one of the candidates
clears it. The guarantee no longer depends on which planner is running.

**Rejected:** strengthening the prompt. That is asking a model nicely not to do
the thing the guard exists to stop.

---

### 16. Verification expectations come from the write, not the planner

Verification compared what the planner chose to pass. A planner could therefore
verify less than it wrote — and in the probe above it did, checking the due date
but not the amount.

The loop now fills every `expect_<field>` of a verify tool from the arguments of
the write that actually happened, overriding anything the planner supplied, and
records in the trace which written fields were compared and which the verify
tool cannot check. Payables are now checked on all four written fields instead of
two.

**Rejected:** trusting `use_extracted`. It made the right thing easy, not the
wrong thing impossible.

---

### 17. Tools have kinds; the loop never reasons by name

The README used to say the loop "contains no mention of invoices". True of the
word; false in substance — `_bind` and `_absorb` hardcoded `vendor`,
`invoice_number`, `amount` and `due_date`, and the deterministic planner named
every invoice tool and all three vendors. A second task would have needed edits
to both.

Each tool now declares a kind (`discover`, `read`, `write`, `verify`, `ask`) and
a parameter schema. Binding matches a reader's `fields` to a writer's declared
parameters by name; verification is recognised by kind. A static test fails if a
task name appears in the framework code — and run against the earlier version it
finds eleven.

**Rejected:** a per-task adapter layer. It would have moved the coupling, not
removed it.

---

### 18. Company procedures are data

The brief describes an operator that uses *"company context to identify the
right sources, procedures, and permitted actions."* A procedure is a JSON file:
which tool plays each role, and the house rules for that kind of request. The
agent picks one by matching the goal and shows it to the planner.

That serves both planners. An LLM reads it as guidance. The deterministic planner
executes it, which is what let the same planner code run the second task with no
changes. If two procedures match equally, none is chosen — an ambiguous request
should not be forced into the wrong process.

**Rejected:** encoding procedures in the planner. That is how the first version
ended up naming every invoice tool.

---

### 19. Fields are found by their visible label

The browser tool used to fill `#vendor`, `#amount` and so on, and read the result
from `#r-id`. It operated one form.

It now finds fields by label (`get_by_label`), buttons by visible name, errors in
`role="alert"` and confirmations in `role="status"`, and reads result rows by
their header text. A new task needs a field-to-label map, not browser code, and
the same approach works on any form built to be usable by a person.

**Rejected:** a model reading screenshots for every field. Far more general, far
slower, and far less certain — and this step needs certainty. It is the right
fallback for forms without usable labels, which is listed under limitations.

---

### 20. Read the document, not the rendering

After the switch to labels, every scenario still passed — yet the write tools
were returning empty record IDs. The result page's stylesheet uppercases table
headers, and Playwright's `inner_text()` returns rendered text, so "Record ID"
came back as "RECORD ID" and every lookup missed. Only "GSTIN", already
uppercase, survived.

The scenarios passed because verification reads the record back independently —
which is exactly the case for not trusting the success page. The rows are now
read with `text_content()`, and a regression test asserts the parsed record; run
against the old code, it fails.

---

### 21. A reply must point at exactly one candidate

For the supplier task the candidates are "account on signed form" and "account
from email update". Matching any word of a label meant "use the account from the
email update" resolved to the **signed-form** account — the shared word "account"
matched the first label first. And "not 3048…, use 9180…" picked the first
number it saw.

Only words distinctive to one label now count, and a reply naming two candidates
resolves nothing. The cost is that some genuine replies need rephrasing. In a
fraud check, being asked again is the right failure.
