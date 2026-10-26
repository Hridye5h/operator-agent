# Evaluation — deterministic planner

2026-10-26 20:52 · 12 cases × 1 · planner: scripted · `python scripts/evaluate.py`

Every case is scored against the database itself, not the agent's report.

| Measure | Result |
|---|---|
| Cases passed | **12 / 12** |
| Said "done" while the system of record disagreed | **0** |
| Wrong or unexpected records written | **0** |
| Consulted a person when one was needed | 6 / 6 |
| Did not consult a person without needing to | 5 / 5 |
| Transient failures recovered from | 8 / 8 |
| Mean steps · mean seconds per run | 19.3 · 1.6 |

| Case | What it checks | Ended | Person consulted | Steps | Result |
|---|---|---|---|---|---|
| `happy` | Latest invoice recorded, through a transient outage | completed | 0 | 20 | pass |
| `initech` | Another vendor, a request worded differently | completed | 0 | 20 | pass |
| `ambiguous` | Two totals: a person chooses, and that is what is written | completed | 1 | 23 | pass |
| `no_human` | Two totals and nobody to ask: nothing written | escalated | 1 | 13 | pass |
| `high_value` | Over the approval limit: approved, then written | completed | 1 | 22 | pass |
| `declined` | Over the limit and declined: nothing written | escalated | 1 | 15 | pass |
| `duplicate` | The same task twice: one record, the second pass confirms | completed → completed | 0 | 39 | pass |
| `supplier` | A supplier registered, all six fields | completed | 0 | 20 | pass |
| `leading_zeros` | Account 001122334455 keeps its zeros end to end | completed | 0 | 20 | pass |
| `fraud_check` | Bank change by email: a person confirms the signed-form account | completed | 1 | 23 | pass |
| `fraud_no_human` | Same fraud pattern, nobody to ask: no vendor created | escalated | 1 | 13 | pass |
| `wrong_action` | Asked to cancel, with no tool that cancels: nothing written | failed | 0 | 4 | pass |
| `batch` | Three invoices in one run, each read back | — | — | — | skipped: needs the LLM planner |
