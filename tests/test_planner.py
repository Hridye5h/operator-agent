"""Planner coverage: parsing model output, and the Gemini path end to end.

The Gemini client is stubbed, so these run with no API key and no network while
still exercising the real ``GeminiPlanner`` code — the request it builds, the
response shapes it has to survive, and how the agent reacts when a reply cannot
be used.
"""

import pathlib
import sys
import types as pytypes

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from agent.core import Agent, AgentConfig  # noqa: E402
from agent.llm import (Decision, GeminiPlanner, ScriptedPlanner,  # noqa: E402
                       _extract_json, build_planner)
from agent.tools import ToolRegistry, ToolResult  # noqa: E402
from agent.trace import Trace  # noqa: E402


# --------------------------------------------------------------------------
# Parsing what a model actually sends back
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,action", [
    ('{"action":"list_invoices","args":{}}', "list_invoices"),
    ('```json\n{"action":"read_invoice","args":{"latest":true}}\n```', "read_invoice"),
    ('```\n{"action":"finish","args":{}}\n```', "finish"),
    ('Sure! Here is my decision:\n{"action":"finish","args":{}}', "finish"),
    ('{"action":"finish","args":{}}\nHope that helps!', "finish"),
    ('{"tool":"finish","arguments":{}}', "finish"),
    ('{"action":"ask_human","args":{"question":"use {this} one?"}}', "ask_human"),
    ('{"action":"submit_payable","args":{"meta":{"a":1},"amount":"5"}}', "submit_payable"),
])
def test_realistic_model_output_parses(raw, action):
    assert Decision.parse(raw).action == action


@pytest.mark.parametrize("raw", [
    "", "   ",
    "I will list the invoices now.",      # prose, no JSON
    "[1,2,3]",                            # array, not an object
    "{'action':'finish'}",                # python quoting
    '{"action":"finish",}',               # trailing comma
    '{"args":{}}',                        # no action
    '{"action":"finish","args":{',        # truncated mid-object
])
def test_unusable_output_is_rejected_loudly(raw):
    with pytest.raises(ValueError):
        Decision.parse(raw)


def test_brace_inside_a_string_does_not_end_the_object():
    d = Decision.parse('{"action":"ask_human","args":{"question":"pick {a} or {b}"}}')
    assert d.args["question"] == "pick {a} or {b}"


def test_escaped_quotes_survive():
    d = Decision.parse(r'{"action":"ask_human","args":{"question":"say \"hi\""}}')
    assert d.args["question"] == 'say "hi"'


def test_args_of_the_wrong_type_degrade_to_empty():
    assert Decision.parse('{"action":"finish","args":[1,2]}').args == {}


def test_confidence_is_carried_through():
    assert Decision.parse('{"action":"finish","args":{},"confidence":0.2}').confidence == 0.2


def test_extract_json_finds_the_first_complete_object():
    blob = _extract_json('noise {"a": {"b": 1}} trailing {"c": 2}')
    assert blob == {"a": {"b": 1}}


# --------------------------------------------------------------------------
# GeminiPlanner against a stubbed client
# --------------------------------------------------------------------------

class StubModels:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.reply(contents) if callable(self.reply) else self.reply


class StubClient:
    def __init__(self, reply):
        self.models = StubModels(reply)


def response(text=None, finish_reason=None, block_reason=None):
    candidate = pytypes.SimpleNamespace(finish_reason=finish_reason)
    feedback = pytypes.SimpleNamespace(block_reason=block_reason)
    return pytypes.SimpleNamespace(text=text, candidates=[candidate],
                                   prompt_feedback=feedback)


def test_gemini_planner_builds_the_request_we_expect():
    client = StubClient(response('{"action":"finish","args":{}}'))
    planner = GeminiPlanner(client=client, model="gemini-2.0-flash")

    decision = planner.decide("SYSTEM RULES", "GOAL: do the thing")

    assert decision.action == "finish"
    call = client.models.calls[0]
    assert call["model"] == "gemini-2.0-flash"
    assert call["contents"] == "GOAL: do the thing"
    assert call["config"].system_instruction == "SYSTEM RULES"
    assert call["config"].temperature == 0.0
    assert call["config"].response_mime_type == "application/json"


def test_gemini_planner_needs_no_key_when_a_client_is_supplied():
    GeminiPlanner(client=StubClient(response('{"action":"finish","args":{}}')))


def test_gemini_planner_reports_a_blocked_prompt_clearly():
    planner = GeminiPlanner(client=StubClient(response(text=None, block_reason="SAFETY")))
    with pytest.raises(RuntimeError, match="prompt blocked: SAFETY"):
        planner.decide("sys", "ctx")


def test_gemini_planner_reports_a_truncated_reply_clearly():
    planner = GeminiPlanner(client=StubClient(response(text=None, finish_reason="MAX_TOKENS")))
    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        planner.decide("sys", "ctx")


def test_gemini_planner_surfaces_a_transport_error():
    def boom(_contents):
        raise ConnectionError("tls handshake failed")

    planner = GeminiPlanner(client=StubClient(boom))
    with pytest.raises(ConnectionError):
        planner.decide("sys", "ctx")


# --------------------------------------------------------------------------
# How the agent reacts to a planner that misbehaves
# --------------------------------------------------------------------------

@pytest.fixture()
def trace(tmp_path):
    return Trace("test goal", root=tmp_path)


def registry_with_verify():
    reg = ToolRegistry()
    reg.tool("verify_payable", "stub", {"type": "object", "properties": {}})(
        lambda **k: ToolResult(True, data={"id": 1}, display="VERIFIED"))
    return reg


def test_one_bad_reply_does_not_kill_the_run(trace):
    """A model that answers with prose should be re-prompted, not fatal."""
    replies = iter([
        response("I think I should probably list the invoices."),   # unusable
        response('{"action":"verify_payable","args":{}}'),
        response('{"action":"finish","args":{}}'),
    ])
    planner = GeminiPlanner(client=StubClient(lambda _c: next(replies)))
    agent = Agent(planner, registry_with_verify(), AgentConfig(max_steps=5))

    result = agent.run("do the thing", trace)

    assert result.status == "completed"
    assert result.verified


def test_the_parse_error_is_shown_to_the_model(trace):
    seen = []

    def reply(contents):
        seen.append(contents)
        if len(seen) == 1:
            return response("no json here")
        return response('{"action":"verify_payable","args":{}}') if len(seen) == 2 \
            else response('{"action":"finish","args":{}}')

    planner = GeminiPlanner(client=StubClient(reply))
    agent = Agent(planner, registry_with_verify(), AgentConfig(max_steps=5))
    agent.run("do the thing", trace)

    assert "COULD NOT BE USED" in seen[1]


def test_a_planner_that_never_recovers_fails_cleanly(trace):
    planner = GeminiPlanner(client=StubClient(response("never valid json")))
    agent = Agent(planner, registry_with_verify(),
                  AgentConfig(max_steps=4, planner_retries=2))

    result = agent.run("do the thing", trace)

    assert result.status == "failed"
    assert not result.verified
    assert trace.outcome == "failed"


def test_retry_budget_is_respected(trace):
    attempts = {"n": 0}

    def reply(_contents):
        attempts["n"] += 1
        return response("still not json")

    planner = GeminiPlanner(client=StubClient(reply))
    agent = Agent(planner, registry_with_verify(),
                  AgentConfig(max_steps=4, planner_retries=2))
    agent.run("do the thing", trace)

    assert attempts["n"] == 3            # one attempt plus two retries, then stop


# --------------------------------------------------------------------------
# Planner selection
# --------------------------------------------------------------------------

def test_without_a_key_it_falls_back_and_says_why(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    planner, note = build_planner("auto")

    assert isinstance(planner, ScriptedPlanner)
    assert note and "GEMINI_API_KEY" in note


def test_asking_for_gemini_without_a_key_is_an_error_not_a_downgrade(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        build_planner("gemini")


def test_scripted_can_be_requested_explicitly():
    planner, note = build_planner("scripted")
    assert isinstance(planner, ScriptedPlanner)
    assert note is None
