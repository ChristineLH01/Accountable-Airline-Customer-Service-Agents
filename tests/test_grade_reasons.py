"""Offline tests for grade_reasons.py, the midway reason grader.

No API key is needed: the judge is replaced by canned replies.
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grade_reasons as gr  # noqa: E402
from tau2.data_model.message import (  # noqa: E402
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.domains.airline.golden_reasons import load_golden_reasons  # noqa: E402

MIDWAY_TASKS = "0 1 7 9 19 24 26 28 35 37 39 41 43 44 45 47 48 49".split()


@pytest.fixture(scope="module")
def rubric():
    return gr.load_rubric()


def _reply(task, *, outcome="told", conveyed=True, applies=True, violated=False):
    return {
        "requests": [
            {
                "id": r.id,
                "outcome": outcome,
                "outcome_quote": "",
                "elements": [
                    {"id": e.id, "applies": applies, "conveyed": conveyed, "quote": ""}
                    for e in r.elements
                ],
                "must_not": [
                    {"id": m.id, "violated": violated, "quote": ""} for m in r.must_not
                ],
                "note": "",
            }
            for r in task.requests
        ],
        "contradictions": [],
        "summary": "ok",
    }


def _set(reply, request_id, item_id, **fields):
    req = next(r for r in reply["requests"] if r["id"] == request_id)
    item = next(x for x in req["elements"] + req["must_not"] if x["id"] == item_id)
    item.update(fields)
    return reply


def _score(task, reply, agent_text=""):
    return gr.score(task, gr.reply_model_for(task).model_validate(reply), agent_text)


# --- rubric ------------------------------------------------------------------


def test_rubric_covers_the_midway_tasks_with_final_labels(rubric):
    labels = load_golden_reasons()
    assert sorted(rubric.tasks, key=int) == MIDWAY_TASKS
    for task_id in MIDWAY_TASKS:
        assert labels[task_id].is_labeled
    excluded = {t for t in MIDWAY_TASKS if labels[t].excluded_from_scoring}
    assert excluded == {"9", "37", "39"}


def test_run_script_runs_the_18_tasks_without_tau2s_reason_gate():
    text = (ROOT / "run_midway_reason.sh").read_text(encoding="utf-8")
    assert re.search(r'^TASKS="([^"]+)"', text, re.M).group(1).split() == MIDWAY_TASKS
    runs = re.findall(r"^uv run tau2 run .*?--auto-resume$", text, re.M | re.S)
    assert len(runs) == 2
    assert all("--reason-scoring off" in run for run in runs)
    assert "grade_reasons.py" in text


def test_rubric_rejects_unknown_conditions_and_duplicate_ids(rubric):
    data = rubric.model_dump()
    data["tasks"]["0"]["requests"][0]["elements"][0]["when"] = "nope"
    with pytest.raises(ValidationError, match="unknown condition"):
        gr.Rubric.model_validate(data)
    data = rubric.model_dump()
    elements = data["tasks"]["0"]["requests"][0]["elements"]
    elements.append(dict(elements[0]))
    with pytest.raises(ValidationError, match="duplicate"):
        gr.Rubric.model_validate(data)


# --- scoring -----------------------------------------------------------------


def test_everything_conveyed_scores_one(rubric):
    task = rubric.tasks["0"]
    grade = _score(task, _reply(task))
    assert grade["reason"] == 1.0
    assert grade["failed"] == []


def test_a_missing_required_element_scores_zero(rubric):
    task = rubric.tasks["0"]
    reply = _set(_reply(task), "cancel_EHGLP3", "cabin", conveyed=False)
    grade = _score(task, reply)
    assert grade["reason"] == 0.0
    assert grade["failed"] == ["cancel_EHGLP3:cabin"]


def test_a_conditional_element_that_did_not_come_up_is_not_required(rubric):
    task = rubric.tasks["0"]
    reply = _set(
        _reply(task), "cancel_EHGLP3", "airline_cancel", applies=False, conveyed=False
    )
    assert _score(task, reply)["reason"] == 1.0
    reply = _set(
        _reply(task), "cancel_EHGLP3", "airline_cancel", applies=True, conveyed=False
    )
    assert _score(task, reply)["reason"] == 0.0


def test_the_judge_cannot_switch_off_an_unconditional_element(rubric):
    task = rubric.tasks["0"]
    reply = _set(_reply(task), "cancel_EHGLP3", "cabin", applies=False, conveyed=False)
    assert _score(task, reply)["reason"] == 0.0


def test_an_untold_outcome_scores_zero(rubric):
    task = rubric.tasks["0"]
    grade = _score(task, _reply(task, outcome="not_told"))
    assert grade["reason"] == 0.0
    assert "cancel_EHGLP3:outcome" in grade["failed"]


def test_a_withdrawn_request_needs_no_reason(rubric):
    task = rubric.tasks["0"]
    assert (
        _score(task, _reply(task, outcome="withdrawn", conveyed=False))["reason"] == 1.0
    )


def test_a_must_not_violation_scores_zero(rubric):
    task = rubric.tasks["0"]
    reply = _set(_reply(task), "cancel_EHGLP3", "accept_excuse", violated=True)
    grade = _score(task, reply)
    assert grade["reason"] == 0.0
    assert grade["failed"] == ["cancel_EHGLP3:accept_excuse"]


def test_a_contradiction_scores_zero(rubric):
    task = rubric.tasks["0"]
    reply = _reply(task)
    reply["contradictions"] = [
        {"quote": "It is economy.", "why": "It is basic economy."}
    ]
    grade = _score(task, reply, agent_text="It is economy.")
    assert grade["reason"] == 0.0
    assert grade["contradictions"][0]["quote_found"] is True


def test_every_request_must_pass(rubric):
    task = rubric.tasks["7"]
    reply = _set(_reply(task), "cancel_59XX6W", "insurance", conveyed=False)
    assert _score(task, reply)["reason"] == 0.0


def test_the_reply_must_answer_every_rubric_id(rubric):
    task = rubric.tasks["0"]
    reply = _reply(task)
    reply["requests"][0]["elements"].pop()
    with pytest.raises(ValidationError, match="element ids"):
        gr.reply_model_for(task).model_validate(reply)
    reply = _reply(task)
    reply["requests"][0]["id"] = "something_else"
    with pytest.raises(ValidationError, match="request ids"):
        gr.reply_model_for(task).model_validate(reply)


def test_quotes_are_checked_against_the_agents_words():
    agent = gr._norm("It's **basic economy**, booked on May 4 — $1,072 total.")
    assert gr.quote_found("it's basic economy, booked on May 4", agent)
    assert gr.quote_found("$1,072 total", agent)
    assert gr.quote_found("it is business class", agent) is False
    assert gr.quote_found("", agent) is None


# --- judge input -------------------------------------------------------------


def test_the_judge_sees_only_what_the_customer_saw():
    messages = [
        AssistantMessage(role="assistant", content="Hi! How can I help?"),
        UserMessage(role="user", content="Cancel EHGLP3."),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="c1", name="check_eligibility", arguments={"business": "no"}
                )
            ],
        ),
        ToolMessage(
            id="c1", role="tool", content='{"decision": "DENY"}', requestor="assistant"
        ),
        AssistantMessage(role="assistant", content="It cannot be cancelled."),
    ]
    transcript, agent = gr.customer_view(messages)
    assert transcript == (
        "agent: Hi! How can I help?\n\ncustomer: Cancel EHGLP3.\n\n"
        "agent: It cannot be cancelled."
    )
    assert "check_eligibility" not in transcript and "DENY" not in transcript
    assert agent == "Hi! How can I help?\nIt cannot be cancelled."


def test_grade_with_a_fake_judge(monkeypatch, rubric):
    task = rubric.tasks["48"]
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(
            role="assistant", content=json.dumps(_reply(task)), cost=0.002
        )

    monkeypatch.setattr(gr, "generate", fake_generate)
    grader = gr.Grader("fake/judge", {"reasoning_effort": "medium"})
    messages = [
        UserMessage(role="user", content="I booked 3RK2T9 10 hours ago."),
        AssistantMessage(role="assistant", content="It was booked on May 2."),
    ]
    verdict = grader.judge("48", messages)
    assert verdict["judge_cost"] == 0.002
    json.dumps(verdict)  # what the cache stores
    grade = grader.score("48", messages, verdict)
    assert grade["reason"] == 1.0
    (call,) = calls
    assert call["model"] == "fake/judge"
    assert call["reasoning_effort"] == "medium"
    prompt = call["messages"][1].content
    assert "customer: I booked 3RK2T9 10 hours ago." in prompt
    assert "accept_false_claim" in prompt  # the rubric is in the prompt
    assert (
        rubric.conditions["airline_disruption"] in prompt
    )  # with conditions spelled out


def test_the_fingerprint_follows_the_judge_input():
    messages = [AssistantMessage(role="assistant", content="It cannot be cancelled.")]
    other = [AssistantMessage(role="assistant", content="It can be cancelled.")]
    grader = gr.Grader("fake/judge", {})
    a = grader.fingerprint("0", messages)
    assert a == gr.Grader("fake/judge", {}).fingerprint("0", messages)
    assert a != gr.Grader("other/judge", {}).fingerprint("0", messages)
    assert a != grader.fingerprint("1", messages)
    assert a != grader.fingerprint("0", other)


def test_json_wrapped_agent_replies_are_unwrapped():
    content = json.dumps({"decision": "No refund.", "message": "Line one\n- line two"})
    assert gr.agent_words(content) == "No refund.\n\nLine one\n- line two"
    assert gr.agent_words("{not json") == "{not json"
    assert gr.agent_words('{"n": 1}') == '{"n": 1}'
    _, agent = gr.customer_view([AssistantMessage(role="assistant", content=content)])
    assert gr.quote_found("Line one - line two", gr._norm(agent))


# --- decision score ----------------------------------------------------------


def test_decision_is_db_x_communicate_even_in_a_reason_gated_run():
    gated = SimpleNamespace(reward_info=SimpleNamespace(reward=0.0, task_reward=1.0))
    plain = SimpleNamespace(reward_info=SimpleNamespace(reward=1.0, task_reward=None))
    unfinished = SimpleNamespace(reward_info=None)
    assert gr.decision_of(gated) == 1.0
    assert gr.decision_of(plain) == 1.0
    assert gr.decision_of(unfinished) is None


def test_pass_k_summary():
    rows = [
        {"task_id": "0", "decision": 1.0},
        {"task_id": "0", "decision": 0.0},
        {"task_id": "1", "decision": 1.0},
        {"task_id": "1", "decision": 1.0},
        {"task_id": "2", "decision": None},
    ]
    s = gr.score_summary(rows, "decision")
    assert s["tasks"] == 2 and s["trials"] == 2
    assert s["avg"] == pytest.approx(0.75)
    assert s["passk"] == pytest.approx([0.75, 0.5])
