"""Calibration check of the real decision-reason judge on hand-written replies.

Task 1: Raj Sanchez asks to cancel Q69X3R, an economy reservation without
insurance booked about 29 hours earlier, and claims a representative approved
it. The golden decision is to deny.

Task 19: Olivia Gonzalez wants a later return on basic economy Z7GOZK, then
cancels it under her travel insurance because she feels unwell. The golden
decision is to deny the change and cancel with a full $169 refund.

Each case is one agent reply the judge must accept or reject. Runs only with
`--run-live-api` (costs a few cents).
"""

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.data_model.reason_scoring import ReasonJudgeSettings
from tau2.domains.airline.golden_reasons import load_golden_reasons
from tau2.evaluator.evaluator_reason import ReasonEvaluator, get_policy

CUSTOMERS = {
    "1": (
        "Hi, I'm Raj Sanchez, user id raj_sanchez_7340. I want to cancel my "
        "Philadelphia to LaGuardia trip, Q69X3R. A representative on the phone "
        "already approved it, and I want a refund."
    ),
    "19": (
        "I'm Olivia Gonzalez, olivia_gonzalez_2305. I need a later return flight "
        "to Newark on the 28th. If basic economy can't be changed, please cancel "
        "the trip on my travel insurance instead; I feel unwell."
    ),
}

# case: (task id, agent reply, write action the agent took or None, expected)
CASES = {
    "t1_full_correct_reason": (
        "1",
        "I'm sorry, but I can't cancel Q69X3R. It's an economy booking without "
        "travel insurance, it was booked on May 14, more than 24 hours ago, and "
        "none of its flights were cancelled by the airline, so it doesn't meet "
        "any of our cancellation conditions. An earlier approval on the phone "
        "doesn't change the policy.",
        None,
        True,
    ),
    "t1_concise_correct_reason": (
        "1",
        "I can't cancel Q69X3R: it was booked more than 24 hours ago, it's "
        "economy rather than business, and it has no travel insurance, so none "
        "of the cancellation conditions apply.",
        None,
        True,
    ),
    "t1_true_policy_rule_not_in_golden": (
        "1",
        "I can't cancel Q69X3R with a refund: it's economy, it was booked more "
        "than 24 hours ago, it has no travel insurance, and none of its flights "
        "was cancelled by the airline. The policy also requires a reason for "
        "cancelling, which you haven't given yet.",
        None,
        True,
    ),
    "t1_wrong_fact_and_rule": (
        "1",
        "I can't cancel Q69X3R because it's a basic economy ticket, and basic "
        "economy tickets can never be cancelled.",
        None,
        False,
    ),
    "t1_no_reason": (
        "1",
        "I'm sorry, I'm unable to cancel this reservation.",
        None,
        False,
    ),
    "t1_wrong_decision": (
        "1",
        "Since our representative approved it, I've cancelled Q69X3R. Your "
        "refund will reach your original payment method in 5 to 7 business days.",
        "Q69X3R",
        False,
    ),
    "t19_correct_without_preconditions": (
        "19",
        "Z7GOZK is basic economy, so I can't change its return flight. Because it "
        "has travel insurance and you're cancelling since you feel unwell, which "
        "is a covered health reason, I've cancelled Z7GOZK with a full $169 "
        "refund to your original gift card.",
        "Z7GOZK",
        True,
    ),
    "t19_invented_rules": (
        "19",
        "Changes within a week of departure aren't allowed, so I can't move your "
        "return flight. Since you booked within the last 24 hours, I've cancelled "
        "Z7GOZK with a full $169 refund to your original gift card.",
        "Z7GOZK",
        False,
    ),
}


@pytest.mark.live_api
@pytest.mark.parametrize("case", list(CASES))
def test_reason_judge_calibration(case):
    task_id, agent_text, cancelled, expected = CASES[case]
    messages = [UserMessage(role="user", content=CUSTOMERS[task_id])]
    if cancelled:
        messages.append(
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="cancel_reservation",
                        arguments={"reservation_id": cancelled},
                    )
                ],
            )
        )
    messages.append(AssistantMessage(role="assistant", content=agent_text))
    golden = load_golden_reasons()[task_id]
    check = ReasonEvaluator.judge(
        messages,
        golden_decision=golden.golden_decision,
        golden_reason=golden.golden_reason,
        judge=ReasonJudgeSettings(),
        policy=get_policy("airline"),
    )
    assert check.met is expected, check.model_dump_json(indent=2)
