"""Method B core logic: the cancellation checklist and the gate.

This file has no dependency on tau2-bench, so it can be tested on its own.

The checklist follows the airline policy, section "Cancel flight":
  - If any segment has already been flown, the agent cannot help and must transfer.
  - Otherwise the reservation can be cancelled if ANY of these holds:
      1. booked within the last 24 hours
      2. the flight was cancelled by the airline
      3. it is a business reservation
      4. it has travel insurance and the reason is covered (health or weather)
  - Otherwise it cannot be cancelled.

The condition names match the Cancel sheet of our gold labels, so B's checklist
can be compared with the gold answer condition by condition.

An unknown cancellation reason only blocks a decision when it can change the
outcome, that is when the reservation has insurance and no other condition is
met. Otherwise the decision is made, and reason_obtained records that the
policy step of asking for the reason was skipped.
"""

from __future__ import annotations

YES, NO, UNKNOWN = "yes", "no", "unknown"
TRI_STATE = {YES, NO, UNKNOWN}
CANCEL_REASONS = {
    "change_of_plan",
    "airline_cancelled",
    "health",
    "weather",
    "other",
    "unknown",
}
COVERED_BY_INSURANCE = {"health", "weather"}

ALLOW, DENY, TRANSFER, NEED_INFO = "ALLOW", "DENY", "TRANSFER", "NEED_INFO"


def _norm(value: str, allowed: set[str], field: str) -> str:
    v = str(value).strip().lower().replace(" ", "_")
    if v in ("true", "y"):
        v = YES
    if v in ("false", "n"):
        v = NO
    if v not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, got {value!r}")
    return v


def evaluate_cancellation(
    reservation_id: str,
    cancel_reason: str,
    already_flown: str,
    booked_within_24h: str,
    cancelled_by_airline: str,
    business: str,
    has_insurance: str,
    evidence: str = "",
) -> dict:
    """Turn the agent's filled checklist into a decision.

    The agent fills each condition with yes, no or unknown, based on tool results.
    The decision itself is computed here by fixed rules, so the agent cannot
    talk itself into a different outcome.
    """
    reason = _norm(cancel_reason, CANCEL_REASONS, "cancel_reason")
    flown = _norm(already_flown, TRI_STATE, "already_flown")
    within_24h = _norm(booked_within_24h, TRI_STATE, "booked_within_24h")
    airline = _norm(cancelled_by_airline, TRI_STATE, "cancelled_by_airline")
    biz = _norm(business, TRI_STATE, "business")
    insured = _norm(has_insurance, TRI_STATE, "has_insurance")

    if insured == YES and reason in COVERED_BY_INSURANCE:
        covers = YES
    elif insured == NO or reason in {"change_of_plan", "airline_cancelled", "other"}:
        covers = NO
    else:
        covers = UNKNOWN

    conditions = {
        "cancel_reason": reason,
        "already_flown": flown,
        "booked_within_24h": within_24h,
        "cancelled_by_airline": airline,
        "business": biz,
        "insurance_covers_reason": covers,
    }
    met = [
        k
        for k in (
            "booked_within_24h",
            "cancelled_by_airline",
            "business",
            "insurance_covers_reason",
        )
        if conditions[k] == YES
    ]
    missing = [k for k, v in conditions.items() if v == UNKNOWN]
    open_conditions = [
        k
        for k in (
            "booked_within_24h",
            "cancelled_by_airline",
            "business",
            "insurance_covers_reason",
        )
        if conditions[k] == UNKNOWN
    ]
    reason_obtained = reason != "unknown"
    ask_reason = (
        ""
        if reason_obtained
        else " The policy also requires you to ask the user for the reason for cancellation."
    )

    if flown == UNKNOWN:
        decision = NEED_INFO
        explanation = (
            "Check whether any segment has already been flown before deciding."
        )
    elif flown == YES:
        decision = TRANSFER
        explanation = (
            "A segment has already been flown, so the agent cannot cancel. "
            "Policy requires a transfer to a human agent."
        )
    elif met:
        decision = ALLOW
        explanation = (
            "Cancellation is allowed because: " + ", ".join(met) + "." + ask_reason
        )
    elif open_conditions:
        decision = NEED_INFO
        if open_conditions == ["insurance_covers_reason"] and not reason_obtained:
            explanation = (
                "The reservation has insurance, so the decision depends on the "
                "reason for cancellation. Ask the user for it, then check again."
            )
        else:
            explanation = (
                "No condition is met yet, but these are still unknown: "
                + ", ".join(open_conditions)
                + ". Look them up, then check again."
            )
    else:
        decision = DENY
        explanation = (
            "None of the cancellation conditions is met, so the reservation cannot be "
            "cancelled. There is no option to cancel without a refund." + ask_reason
        )

    return {
        "reservation_id": str(reservation_id).strip().upper(),
        "decision": decision,
        "conditions": conditions,
        "conditions_met": met,
        "unknown": missing,
        "reason_obtained": reason_obtained,
        "evidence": evidence,
        "explanation": explanation,
    }


class CancelGate:
    """Blocks cancel_reservation unless the latest checklist for that reservation is ALLOW."""

    GATED_TOOL = "cancel_reservation"

    def __init__(self, approvals: dict[str, str] | None = None):
        self.approvals = approvals if approvals is not None else {}

    def record(self, checklist: dict) -> None:
        self.approvals[checklist["reservation_id"]] = checklist["decision"]

    def check(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        if tool_name != self.GATED_TOOL:
            return True, ""
        rid = str(arguments.get("reservation_id", "")).strip().upper()
        decision = self.approvals.get(rid)
        if decision == ALLOW:
            return True, ""
        if decision is None:
            return False, (
                f"BLOCKED: cancel_reservation({rid}) was not executed. "
                "You must call check_eligibility for this reservation first."
            )
        return False, (
            f"BLOCKED: cancel_reservation({rid}) was not executed. "
            f"The latest eligibility check decided {decision}, not ALLOW. "
            "Explain the decision to the user instead."
        )
