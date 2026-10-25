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


# --------------------------------------------------------------------------
# Live-API hardening: model selection, rate limits, .env
# --------------------------------------------------------------------------

class FakeAPIError(Exception):
    """Shaped like google.genai.errors.APIError."""

    def __init__(self, code, status="", message="", details=None):
        super().__init__(f"{code} {status} {message}")
        self.code, self.status, self.message, self.details = code, status, message, details


class ListingModels(StubModels):
    def __init__(self, reply, names):
        super().__init__(reply)
        self._names = names

    def list(self):
        return [pytypes.SimpleNamespace(name=f"models/{n}",
                                        supported_actions=["generateContent"])
                for n in self._names]


class ListingClient:
    def __init__(self, reply, names):
        self.models = ListingModels(reply, names)


OK = response('{"action":"finish","args":{}}')


def test_model_comes_from_env_when_set(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom")
    assert GeminiPlanner(client=StubClient(OK)).name == "gemini:gemini-custom"


def test_model_is_the_first_preferred_one_the_key_can_use(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = ListingClient(OK, ["gemini-1.5-pro", "gemini-flash-latest", "gemini-2.0-flash"])
    assert GeminiPlanner(client=client).name == "gemini:gemini-flash-latest"


def test_an_unknown_lineup_falls_back_to_the_newest_flash(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = ListingClient(OK, ["gemini-3.0-flash", "gemini-3.1-flash", "gemini-3.1-flash-lite"])
    assert GeminiPlanner(client=client).name == "gemini:gemini-3.1-flash"


def test_a_listing_failure_does_not_stop_the_planner(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    from agent.llm import PREFERRED_MODELS
    assert GeminiPlanner(client=StubClient(OK)).name == f"gemini:{PREFERRED_MODELS[0]}"


def flaky(errors):
    """A reply function that raises each queued error once, then succeeds."""
    queue = list(errors)

    def reply(_contents):
        if queue:
            raise queue.pop(0)
        return OK
    return reply


def test_a_rate_limit_waits_for_the_servers_own_delay():
    slept = []
    err = FakeAPIError(429, "RESOURCE_EXHAUSTED", "rate limited", details={"error": {
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                     "retryDelay": "17s"}]}})
    planner = GeminiPlanner(client=StubClient(flaky([err])), model="m", sleep=slept.append)

    assert planner.decide("sys", "ctx").action == "finish"
    assert slept == [17.0]
    assert planner.waits == [(429, 17.0)]


def test_without_a_delay_it_backs_off_exponentially():
    slept = []
    errs = [FakeAPIError(503, "UNAVAILABLE") for _ in range(3)]
    planner = GeminiPlanner(client=StubClient(flaky(errs)), model="m", sleep=slept.append)

    planner.decide("sys", "ctx")
    assert slept == [2.0, 4.0, 8.0]


def test_a_bad_request_is_not_retried():
    slept = []
    planner = GeminiPlanner(client=StubClient(flaky([FakeAPIError(400, "INVALID_ARGUMENT")])),
                            model="m", sleep=slept.append)
    with pytest.raises(FakeAPIError):
        planner.decide("sys", "ctx")
    assert slept == []


def test_a_daily_quota_is_not_waited_out():
    slept = []
    err = FakeAPIError(429, "RESOURCE_EXHAUSTED",
                       "Quota exceeded for metric: generate_content_free_tier_requests, "
                       "limit: 250, per day")
    planner = GeminiPlanner(client=StubClient(flaky([err])), model="m", sleep=slept.append)
    with pytest.raises(FakeAPIError):
        planner.decide("sys", "ctx")
    assert slept == []


def test_retries_stop_at_the_attempt_limit():
    slept = []
    errs = [FakeAPIError(500, "INTERNAL") for _ in range(10)]
    planner = GeminiPlanner(client=StubClient(flaky(errs)), model="m",
                            max_attempts=3, sleep=slept.append)
    with pytest.raises(FakeAPIError):
        planner.decide("sys", "ctx")
    assert len(slept) == 2


def test_the_output_cap_leaves_room_for_reasoning():
    client = StubClient(OK)
    GeminiPlanner(client=client, model="m").decide("sys", "ctx")
    assert client.models.calls[0]["config"].max_output_tokens >= 4096


def test_dotenv_is_loaded_but_never_overrides_the_shell(tmp_path, monkeypatch):
    from run_agent import load_env

    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-file\nGEMINI_MODEL=file-model\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "shell-model")

    assert load_env(env) is True
    import os
    assert os.environ["GEMINI_API_KEY"] == "from-file"
    assert os.environ["GEMINI_MODEL"] == "shell-model"


def test_a_missing_dotenv_is_not_an_error(tmp_path):
    from run_agent import load_env
    assert load_env(tmp_path / "nope.env") is False


def test_preset_answers_match_any_listed_phrasing():
    from tools.human import HumanChannel

    human = HumanChannel(interactive=False,
                         preset={"amount|total|figure": "96400.00"})
    assert human.ask("Which total should I record?").data["answer"] == "96400.00"
    assert human.ask("Conflicting values for 'amount'").data["answer"] == "96400.00"
    assert not human.ask("Should I book a meeting?").ok
