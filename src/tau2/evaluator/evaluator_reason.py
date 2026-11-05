"""Decision-reason evaluation: did the agent tell the customer the right decision
and the right policy-based reason?

Each scored task has a final ``golden_decision`` and ``golden_reason`` in
``data/tau2-airline/mannual_label.json``. One LLM judge call per simulation
compares them with what the agent told the customer and returns a verdict per
customer request. The REASON reward is computed here from those verdicts (see
``ReasonCheck.from_reply``), not taken from the judge, so the pass rule is fixed
in code: every request's decision and reason must be correct and the agent must
have made no incorrect claims.

Tasks without a final label, and tasks marked ``excluded_from_scoring``, are
skipped without an LLM call; the skip and its reason are recorded in
``RewardInfo.info["reason"]``.
"""

import json
from functools import lru_cache
from typing import Optional

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from tau2.data_model.reason_scoring import (
    ReasonCheck,
    ReasonJudgeReply,
    ReasonJudgeSettings,
    ReasonScoring,
)
from tau2.data_model.tasks import Task
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

REASON_JUDGE_CALL_NAME = "reason_eval"
REASON_JUDGE_VALIDATION_RETRIES = 2

# Pinned prompt; any change must bump REASON_JUDGE_PROMPT_VERSION and the digest
# in tests/test_evaluator/test_reason_judge.py.
REASON_JUDGE_SYSTEM_PROMPT = """\
You grade an airline customer-service agent on one thing: whether it told the
customer the correct decision on each of their requests, and the correct
policy-based reason for that decision.

You are given:
- POLICY: the airline policy the agent must follow.
- GOLDEN DECISION: what the agent should decide on each of the customer's
  requests under the airline policy.
- GOLDEN REASON: the policy rule or rules and the facts (from the customer's
  profile, reservations or flights) that justify the golden decision.
- TRANSCRIPT: the conversation. Lines starting with "customer:" and "agent:" are
  messages between them. Lines starting with "agent action:" are tool calls the
  agent made; the customer did not see them.

How to grade

1. Split the golden decision into the customer's separate requests (for example:
   cancel reservation A; compensation for a delayed flight; a question about
   baggage allowance). Grade each request on its own, in the order of the golden
   decision.

2. decision_correct: the agent told the customer an outcome for this request that
   matches the golden decision: approved, denied, or partly approved, with the
   same scope (the same reservations, flights, cabin and passengers). Grade what
   the agent told the customer; use agent actions only to check that what it said
   matches what it did. The decision is incorrect if the agent announced a
   different outcome, never told the customer the outcome, said one thing and did
   another, or ended on the wrong outcome after changing its answer. If the
   customer dropped the request before a decision was needed and the golden
   decision expects no action, the decision is correct.

3. reason_correct: the agent told the customer why, and its reason rests on the
   same deciding policy rule and the same deciding facts as the golden reason.
   The golden reason often lists more than the deciding facts; the agent only
   has to give the deciding ones.
   - Paraphrase is fine. The agent need not repeat exact dates, every flight
     number, or other detail of the golden reason.
   - A deciding fact is one the outcome turns on in this conversation: a fact
     that, if it were different, would change the decision, and that is in play,
     because the customer raised it or because it is the reservation's own
     cabin, insurance, booking time or the customer's membership.
   - Background preconditions that nobody disputes need not be stated: for
     example, that no flight has been flown yet, that the number of passengers
     stays the same, or that a payment method is on file.
   - When the golden reason denies a request because none of the policy's
     allowing conditions holds (for example, a cancellation that is not within 24
     hours of booking, not business class, not covered by insurance and not
     cancelled by the airline), the agent must make clear that no allowing
     condition applies and give the facts behind the conditions that are in
     play. Leaving out a condition that plainly does not apply and that the
     customer never raised (for example, an airline cancellation or delay when
     no flight was cancelled or delayed) is acceptable. This holds for
     compensation as well as cancellations.
   - When the agent approves a request as the customer asked it, the reason only
     needs the facts that set the outcome the customer could question, such as a
     covered cancellation reason, a fare difference, a refund or charge amount,
     or a free or paid bag. It need not explain routine rules that merely permit
     the transaction, such as how a payment was split across allowed methods.
   - When the golden reason corrects a false claim by the customer (a wrong
     membership level, booking time, cabin, passenger count or insurance), the
     agent must not accept the false claim as true.
   - The agent need not rebut customer arguments that are not policy
     conditions, such as a claimed earlier approval from another
     representative, urgency, loyalty or a personal hardship, even when the
     golden reason mentions them. It only must not accept them as grounds for a
     different decision.
   - When the golden reason rests on a number (a price difference against the
     customer's budget, a refund amount, a number of free bags), any number the
     agent gives for it must agree with the golden reason.
   - The reason is incorrect if it is missing, only implied, given for a
     different request, cites a different or invented rule, or states a fact
     that contradicts the golden reason.
   - If decision_correct is false, reason_correct is false.

4. incorrect_claims: list every explanation the agent gave the customer, for any
   request, that is factually wrong (it contradicts the golden reason or the
   facts in the conversation) or that cites a rule that is not in the POLICY or
   does not apply, even if the agent also gave the correct reason. A true
   statement of a POLICY rule is not an incorrect claim just because the golden
   reason does not mention it. Leave the list empty if there are none. Wording
   slips that do not change the meaning are not incorrect claims.

Do not grade politeness, authentication, confirmation steps or anything else
outside the decisions and their reasons. Extra correct information is fine.

Respond with only a JSON object, with no text before or after it:
{
  "requests": [
    {
      "request": "<the customer request, in a few words>",
      "golden_outcome": "<the golden decision for this request>",
      "agent_outcome": "<what the agent told the customer, or 'not communicated'>",
      "decision_correct": <true or false>,
      "reason_correct": <true or false>,
      "evidence": "<short quote of the agent's decision and reason, or empty>",
      "reasoning": "<one or two sentences>"
    }
  ],
  "incorrect_claims": ["<incorrect claim>"],
  "reasoning": "<one-sentence overall summary>"
}
"""

REASON_JUDGE_USER_PROMPT_TEMPLATE = """\
POLICY:
{policy}

GOLDEN DECISION:
{golden_decision}

GOLDEN REASON:
{golden_reason}

TRANSCRIPT:
{transcript}
"""


@lru_cache(maxsize=1)
def _airline_golden_reasons() -> dict:
    from tau2.domains.airline.golden_reasons import load_golden_reasons

    return load_golden_reasons()


def get_golden_reason(domain: str, task_id: str):
    """The task's golden label, or None when the domain has no label file."""
    if domain != "airline":
        return None
    return _airline_golden_reasons().get(str(task_id))


@lru_cache(maxsize=1)
def _airline_policy() -> str:
    from tau2.domains.airline.utils import AIRLINE_POLICY_PATH

    return AIRLINE_POLICY_PATH.read_text(encoding="utf-8")


def get_policy(domain: str) -> str:
    """The domain policy the judge checks claimed rules against."""
    if domain != "airline":
        raise ValueError(f"No reason-judge policy for domain {domain!r}")
    return _airline_policy()


def format_reason_transcript(trajectory: list[Message]) -> str:
    """Customer and agent messages plus the agent's tool calls, without results.

    Tool results are left out so the judge grades against the golden facts
    rather than its own reading of the database.
    """
    lines = []
    for message in trajectory:
        if isinstance(message, UserMessage) and message.has_text_content():
            lines.append(f"customer: {message.content}")
        elif isinstance(message, AssistantMessage):
            if message.has_text_content():
                lines.append(f"agent: {message.content}")
            for call in message.tool_calls or []:
                arguments = json.dumps(call.arguments, sort_keys=True)
                lines.append(f"agent action: {call.name}({arguments})")
    return "\n".join(lines)


class ReasonEvaluator:
    """Grade the agent's decision explanations against the golden labels."""

    @classmethod
    def evaluate(
        cls,
        task: Task,
        full_trajectory: list[Message],
        *,
        domain: str,
        scoring: ReasonScoring,
    ) -> tuple[Optional[ReasonCheck], dict]:
        """Return the check (None when skipped) and the info to record with it."""
        if not scoring.enabled:
            return None, {"skipped": "reason scoring is off"}
        golden = get_golden_reason(domain, task.id)
        if golden is None or not golden.is_labeled:
            return None, {"skipped": "no final golden label for this task"}
        if golden.excluded_from_scoring:
            return None, {
                "skipped": f"excluded from scoring: {golden.exclusion_reason}"
            }
        check = cls.judge(
            full_trajectory,
            golden_decision=golden.golden_decision,
            golden_reason=golden.golden_reason,
            judge=scoring.judge,
            policy=get_policy(domain),
        )
        logger.info(
            f"Reason check: task={task.id} met={check.met} "
            f"decision_correct={check.decision_correct} "
            f"reason_correct={check.reason_correct} "
            f"partial={check.partial_score:.2f} "
            f"incorrect_claims={len(check.incorrect_claims)}"
        )
        return check, {"reason_scoring": scoring.model_dump(mode="json")}

    @classmethod
    def judge(
        cls,
        full_trajectory: list[Message],
        *,
        golden_decision: str,
        golden_reason: str,
        judge: ReasonJudgeSettings,
        policy: str,
    ) -> ReasonCheck:
        """One validated judge call; invalid JSON is retried, then raises."""
        response = generate(
            model=judge.model,
            messages=[
                SystemMessage(role="system", content=REASON_JUDGE_SYSTEM_PROMPT),
                UserMessage(
                    role="user",
                    content=REASON_JUDGE_USER_PROMPT_TEMPLATE.format(
                        policy=policy,
                        golden_decision=golden_decision,
                        golden_reason=golden_reason,
                        transcript=format_reason_transcript(full_trajectory),
                    ),
                ),
            ],
            call_name=REASON_JUDGE_CALL_NAME,
            response_model=ReasonJudgeReply,
            response_validation_retries=REASON_JUDGE_VALIDATION_RETRIES,
            **judge.model_args,
        )
        reply = ReasonJudgeReply.model_validate_json(
            extract_json_from_llm_response(response.content or "")
        )
        return ReasonCheck.from_reply(
            reply,
            golden_decision=golden_decision,
            golden_reason=golden_reason,
            judge=judge,
            judge_cost=getattr(response, "cost", None),
        )
