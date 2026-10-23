"""End-to-end run through the LLM code path, against the real tools.

The unit tests prove that model output parses and that the loop dispatches a
Decision. This proves the join: ``GeminiPlanner``'s replies drive the real
browser, the real server and the real verification read-back.

Only the HTTP client is stubbed, so no key and no network are needed. Everything
downstream of the model's reply is the production path.

Slow (~15s: it boots the app and a browser), so it is marked and skipped by
default. Run it with::

    pytest -m integration
"""

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.core import Agent, AgentConfig  # noqa: E402
from agent.llm import GeminiPlanner  # noqa: E402
from agent.trace import Trace  # noqa: E402

pytestmark = pytest.mark.integration

GOAL = ("Find the latest invoice from Northwind Traders, extract the amount and "
        "due date, record it as a payable in the internal AP system, and confirm "
        "it was stored correctly.")


def _response(text: str):
    return types.SimpleNamespace(
        text=text,
        candidates=[types.SimpleNamespace(finish_reason=None)],
        prompt_feedback=types.SimpleNamespace(block_reason=None),
    )


class _Models:
    """Replies the way a model does: fenced, prefixed with prose, sometimes both."""

    def __init__(self):
        self.calls = 0

    def generate_content(self, *, model, contents, config):
        self.calls += 1
        history = contents.split("HISTORY:", 1)[1] if "HISTORY:" in contents else ""

        if "list_invoices(" not in history:
            return _response('{"action":"list_invoices","args":{},'
                             '"reasoning":"see what exists","confidence":0.9}')

        if "read_invoice(" not in history:
            return _response('```json\n{"action":"read_invoice",'
                             '"args":{"vendor":"Northwind","latest":true},'
                             '"reasoning":"the newest one","confidence":0.9}\n```')

        last = history.rfind("submit_payable(")
        if last == -1:
            return _response('{"action":"submit_payable","args":{"use_extracted":true},'
                             '"reasoning":"enter it","confidence":0.9}')

        tail = history[last:]
        if "ERROR (retriable)" in tail and history.count("submit_payable(") < 4:
            return _response('Let me retry that.\n'
                             '{"action":"submit_payable","args":{"use_extracted":true},'
                             '"reasoning":"transient fault","confidence":0.9}')

        if "verify_payable(" not in history:
            return _response('{"action":"verify_payable","args":{"use_extracted":true},'
                             '"reasoning":"confirm it landed","confidence":0.95}')

        return _response('{"action":"finish","args":{},'
                         '"reasoning":"verified","confidence":1.0}')


class _Client:
    def __init__(self):
        self.models = _Models()


def test_llm_replies_drive_the_real_tools_to_a_verified_outcome(tmp_path):
    import run_agent as ra
    from tools import browser as browser_tools
    from tools.human import HumanChannel

    proc = ra.start_environment()
    trace = Trace(GOAL, root=tmp_path)
    session = browser_tools.BrowserSession(headless=True, artifacts=trace.artifacts_dir)
    planner = GeminiPlanner(client=_Client(), model="gemini-2.0-flash")

    try:
        session.start()
        registry = ra.build_registry(session, HumanChannel(interactive=False, preset={}))
        agent = Agent(planner, registry,
                      AgentConfig(max_steps=12,
                                  approval_policy=ra.value_threshold_policy(100000)))
        result = agent.run(GOAL, trace)
    finally:
        session.stop()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    assert result.status == "completed"
    assert result.verified
    assert result.evidence["invoice_number"] == "NW-2026-0671"
    assert result.evidence["amount"] == pytest.approx(73219.00)
    assert result.evidence["due_date"] == "2026-07-22"

    # The transient outage really happened and was really recovered from.
    observations = [s for s in trace.steps if s.kind == "observation"]
    assert any("SYS-503" in s.summary for s in observations)

    # And the browser produced real evidence along the way.
    assert list(trace.artifacts_dir.glob("*.png"))
