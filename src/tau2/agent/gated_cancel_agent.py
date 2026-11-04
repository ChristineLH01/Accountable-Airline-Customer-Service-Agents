"""Method B: the tau2-bench LLMAgent plus a cancellation checklist and a gate.

What B adds on top of the baseline agent (method A):
  1. A new tool, check_eligibility. Before cancelling, the agent fills the policy
     checklist from tool results. The decision (ALLOW, DENY, TRANSFER, NEED_INFO)
     is computed by fixed rules in eligibility.py.
  2. A gate. If the agent calls cancel_reservation without an ALLOW checklist for
     that reservation, the call is not sent to the environment. The agent gets a
     BLOCKED message back instead and has to respond to the user.

check_eligibility is handled inside the agent, so the tau2 environment and the
grading are unchanged. Both the checklist calls and the blocked calls stay in the
agent's own message history, so they can be inspected after a run.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import Field

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.eligibility import CancelGate, evaluate_cancellation
from tau2.agent.llm_agent import (
    AGENT_INSTRUCTION,
    SYSTEM_PROMPT,
    LLMAgent,
    LLMAgentState,
)
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolMessage
from tau2.environment.tool import as_tool
from tau2.utils.llm_utils import generate

CHECK_TOOL = "check_eligibility"

B_INSTRUCTION = (
    AGENT_INSTRUCTION
    + """

Cancellation checklist (mandatory):
- Before you call cancel_reservation, you must call check_eligibility for that reservation.
- Fill every field from tool results such as get_reservation_details, get_user_details and
  get_flight_status, never from what the user claims. Use "unknown" only if you could not check it.
- cancel_reason must come from the user. If you do not know it yet, ask the user first.
- Follow the decision that check_eligibility returns:
  ALLOW -> you may cancel after the user confirms.
  DENY -> do not cancel; explain to the user which conditions are not met.
  TRANSFER -> do not cancel; transfer to a human agent, after handling the user's other requests.
  NEED_INFO -> look up or ask for the missing information, then call check_eligibility again.
- If a reservation changes (for example after a cabin upgrade), call check_eligibility again.
- cancel_reservation is blocked by the system unless the latest checklist for that reservation is ALLOW.
""".strip()
)


def check_eligibility(
    reservation_id: str,
    cancel_reason: str,
    already_flown: str,
    booked_within_24h: str,
    cancelled_by_airline: str,
    business: str,
    has_insurance: str,
    evidence: str,
) -> str:
    """Fill the cancellation checklist for one reservation and get the policy decision.

    Call this before cancel_reservation. Fill every field from tool results, not from
    the user's claims. The current time is 2024-05-15 15:00 EST.

    Args:
        reservation_id: The reservation to check, such as "EHGLP3".
        cancel_reason: The user's reason. One of change_of_plan, airline_cancelled, health, weather, other, unknown.
        already_flown: Whether any segment has already been flown (status landed or flying). yes, no or unknown.
        booked_within_24h: Whether created_at is within 24 hours of the current time. yes, no or unknown.
        cancelled_by_airline: Whether any segment has status cancelled. yes, no or unknown.
        business: Whether the reservation cabin is business. yes, no or unknown.
        has_insurance: Whether the reservation has travel insurance. yes, no or unknown.
        evidence: Short notes on where each value came from, such as "created_at 2024-05-04, cabin basic_economy".

    Returns:
        A JSON string with the decision (ALLOW, DENY, TRANSFER or NEED_INFO), the conditions and an explanation.
    """
    # Never executed by the environment. The agent intercepts this call itself.
    raise RuntimeError("check_eligibility is handled inside GatedCancelAgent")


class GatedAgentState(LLMAgentState):
    """LLMAgentState plus the gate's memory for this conversation."""

    approvals: dict[str, str] = Field(default_factory=dict)
    checklists: list[dict] = Field(default_factory=list)
    blocked_calls: list[dict] = Field(default_factory=list)


class GatedCancelAgent(LLMAgent[GatedAgentState]):
    """LLMAgent with a mandatory cancellation checklist and a gate on cancel_reservation."""

    def __init__(
        self,
        tools,
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        max_internal_steps: int = 8,
    ):
        super().__init__(
            tools=list(tools) + [as_tool(check_eligibility)],
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
        )
        self.max_internal_steps = max_internal_steps

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy, agent_instruction=B_INSTRUCTION
        )

    def get_init_state(self, message_history=None) -> GatedAgentState:
        base = super().get_init_state(message_history)
        return GatedAgentState(
            system_messages=base.system_messages, messages=base.messages
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GatedAgentState
    ):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        gate = CancelGate(state.approvals)
        for _ in range(self.max_internal_steps):
            reply = generate(
                model=self.llm,
                tools=self.tools,
                messages=state.system_messages + state.messages,
                call_name="agent_response",
                **self.llm_args,
            )
            calls = reply.tool_calls or []
            needs_intercept = any(
                c.name == CHECK_TOOL or not gate.check(c.name, c.arguments)[0]
                for c in calls
            )
            state.messages.append(reply)
            if not needs_intercept:
                return (
                    reply,
                    state,
                )  # normal case: a user message or an allowed tool call

            # Answer every call in this reply ourselves, then let the LLM continue.
            for c in calls:
                if c.name == CHECK_TOOL:
                    try:
                        result = evaluate_cancellation(**c.arguments)
                        gate.record(result)
                        state.checklists.append(result)
                        content, error = json.dumps(result), False
                    except (TypeError, ValueError) as e:
                        content, error = f"Invalid checklist: {e}", True
                else:
                    allowed, why = gate.check(c.name, c.arguments)
                    if not allowed:
                        state.blocked_calls.append(
                            {"tool": c.name, "arguments": c.arguments}
                        )
                        content, error = why, True
                        state.messages.append(
                            ToolMessage(
                                id=c.id,
                                role="tool",
                                content=content,
                                requestor="assistant",
                                error=True,
                                execution_status="rejected",
                            )
                        )
                        continue
                    else:
                        content, error = (
                            (
                                "NOT EXECUTED: make one tool call at a time, "
                                "then call this tool again."
                            ),
                            True,
                        )
                state.messages.append(
                    ToolMessage(
                        id=c.id,
                        role="tool",
                        content=content,
                        requestor="assistant",
                        error=error,
                    )
                )

        fallback = AssistantMessage.text(
            "I'm sorry, I need a moment to review the policy for this request. Could you repeat what you need?"
        )
        state.messages.append(fallback)
        return fallback, state


def create_gated_cancel_agent(tools, domain_policy, **kwargs):
    """Factory for the tau2 registry."""
    return GatedCancelAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
