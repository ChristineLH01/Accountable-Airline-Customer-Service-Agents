"""Tests for Method B: the cancellation checklist, the gate and the gated agent.

No API key is needed. The LLM is replaced by a scripted fake.
"""

import json
from pathlib import Path

import pytest

import tau2.agent.gated_cancel_agent as gated
from tau2.agent.eligibility import (
    ALLOW,
    DENY,
    NEED_INFO,
    TRANSFER,
    CancelGate,
    evaluate_cancellation,
)
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.domains.airline.environment import get_environment
from tau2.registry import registry

GOLD_ROWS = json.loads((Path(__file__).parent / "gold_cancel_rows.json").read_text())


def _reason(text: str) -> str:
    t = text.lower()
    for prefix, reason in (
        ("health", "health"),
        ("weather", "weather"),
        ("change of plan", "change_of_plan"),
    ):
        if t.startswith(prefix):
            return reason
    return "other"  # "not given" or "other (...)": not covered by insurance


# --- checklist logic ---------------------------------------------------------


@pytest.mark.parametrize("row", GOLD_ROWS, ids=[r["task_id"] for r in GOLD_ROWS])
def test_checklist_matches_gold(row):
    out = evaluate_cancellation(
        reservation_id=row["reservation"],
        cancel_reason=_reason(row["cancel_reason"]),
        already_flown=row["already_flown"].lower(),
        booked_within_24h=row["booked_within_24h"].lower(),
        cancelled_by_airline=row["cancelled_by_airline"].lower(),
        business=row["business"].lower(),
        has_insurance=row["has_insurance"],
    )
    assert out["decision"] == row["final_decision"]


def _check(**overrides):
    args = dict(
        reservation_id="abc123",
        cancel_reason="change_of_plan",
        already_flown="no",
        booked_within_24h="no",
        cancelled_by_airline="no",
        business="no",
        has_insurance="no",
    )
    args.update(overrides)
    return evaluate_cancellation(**args)


def test_checklist_edge_cases():
    assert _check()["decision"] == DENY
    assert _check(already_flown="yes", business="yes")["decision"] == TRANSFER
    assert _check(cancel_reason="unknown")["decision"] == DENY
    assert _check(cancel_reason="unknown")["reason_obtained"] is False
    assert _check(cancel_reason="unknown", has_insurance="yes")["decision"] == NEED_INFO
    assert _check(cancel_reason="unknown", business="yes")["decision"] == ALLOW
    assert _check(cancel_reason="unknown", already_flown="yes")["decision"] == TRANSFER
    assert _check(booked_within_24h="unknown")["decision"] == NEED_INFO
    assert _check(has_insurance="yes", cancel_reason="health")["decision"] == ALLOW
    assert _check(has_insurance="yes")["decision"] == DENY
    assert _check()["reservation_id"] == "ABC123"
    with pytest.raises(ValueError):
        _check(business="maybe")


# --- gate --------------------------------------------------------------------


def test_gate():
    gate = CancelGate()
    assert gate.check("get_reservation_details", {"reservation_id": "X1"})[0]
    assert not gate.check("cancel_reservation", {"reservation_id": "X1"})[0]
    gate.record(_check(reservation_id="X1"))  # DENY
    assert not gate.check("cancel_reservation", {"reservation_id": "x1"})[0]
    gate.record(_check(reservation_id="X1", business="yes"))  # ALLOW
    assert gate.check("cancel_reservation", {"reservation_id": "x1"})[0]
    assert not gate.check("cancel_reservation", {"reservation_id": "OTHER"})[0]


# --- agent with a scripted LLM -----------------------------------------------


def _call(name, **arguments):
    return AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)],
    )


@pytest.fixture
def agent():
    env = get_environment()
    return gated.create_gated_cancel_agent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="fake-model",
        llm_args={},
    )


def _script(monkeypatch, replies):
    replies = list(replies)
    monkeypatch.setattr(gated, "generate", lambda **kwargs: replies.pop(0))


def test_agent_is_registered():
    assert "gated_cancel_agent" in registry.get_info().agents


def test_agent_adds_tool_and_prompt(agent):
    assert "check_eligibility" in [t.name for t in agent.tools]
    assert "Cancellation checklist (mandatory)" in agent.system_prompt


def test_cancel_without_checklist_is_blocked(agent, monkeypatch):
    _script(
        monkeypatch,
        [
            _call("cancel_reservation", reservation_id="EHGLP3"),
            AssistantMessage.text("I need to check eligibility first."),
        ],
    )
    state = agent.get_init_state()
    reply, state = agent.generate_next_message(
        UserMessage(role="user", content="Cancel EHGLP3"), state
    )
    assert reply.content == "I need to check eligibility first."
    assert state.blocked_calls == [
        {"tool": "cancel_reservation", "arguments": {"reservation_id": "EHGLP3"}}
    ]
    rejected = [m for m in state.messages if getattr(m, "role", "") == "tool"]
    assert rejected[0].execution_status == "rejected"


def test_cancel_after_allow_goes_to_environment(agent, monkeypatch):
    allow = dict(
        reservation_id="EHGLP3",
        cancel_reason="change_of_plan",
        already_flown="no",
        booked_within_24h="yes",
        cancelled_by_airline="no",
        business="no",
        has_insurance="no",
        evidence="test",
    )
    cancel = _call("cancel_reservation", reservation_id="EHGLP3")
    _script(monkeypatch, [_call("check_eligibility", **allow), cancel])
    state = agent.get_init_state()
    reply, state = agent.generate_next_message(
        UserMessage(role="user", content="Yes, cancel it"), state
    )
    assert reply is cancel  # returned to the orchestrator, so the env executes it
    assert state.approvals == {"EHGLP3": ALLOW}
    assert state.checklists[0]["decision"] == ALLOW
