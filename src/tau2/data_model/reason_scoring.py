"""Settings, judge replies and per-simulation results for decision-reason scoring.

The reason judge grades what the agent told the customer against a task's
golden decision and golden reason (``data/tau2-airline/mannual_label.json``).
See ``tau2.evaluator.evaluator_reason`` and ``docs/evaluation.md``.
"""

from enum import Enum
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from tau2.config import DEFAULT_LLM_REASON_JUDGE, DEFAULT_LLM_REASON_JUDGE_ARGS

REASON_JUDGE_PROMPT_VERSION = "reason-judge-v1"


class ReasonJudgeSettings(BaseModel):
    """The decision-reason judge, resolved eagerly so results record it."""

    model_config = ConfigDict(extra="forbid")
    model: Annotated[
        str,
        Field(
            default=DEFAULT_LLM_REASON_JUDGE,
            min_length=1,
            description="OpenRouter model ID of the decision-reason judge.",
        ),
    ]
    model_args: Annotated[
        dict[str, JsonValue],
        Field(
            default_factory=lambda: dict(DEFAULT_LLM_REASON_JUDGE_ARGS),
            description="Arguments passed to the judge model.",
        ),
    ]
    prompt_version: Annotated[
        Literal["reason-judge-v1"],
        Field(
            default=REASON_JUDGE_PROMPT_VERSION,
            description="Version of the pinned decision-reason judge prompt.",
        ),
    ]


class ReasonScoringMode(str, Enum):
    """Whether the decision-reason check runs, and whether it gates the reward."""

    GATE = "gate"  # judged and multiplied into the final reward
    REPORT = "report"  # judged and recorded; the reward is unchanged
    OFF = "off"  # not judged


class ReasonScoring(BaseModel):
    """How a run scores the agent's decision explanations; gating is the default."""

    model_config = ConfigDict(extra="forbid")
    mode: Annotated[
        ReasonScoringMode,
        Field(
            default=ReasonScoringMode.GATE,
            description="gate: REASON multiplies the final reward; report: judged "
            "and recorded only; off: not judged.",
        ),
    ]
    judge: Annotated[
        Optional[ReasonJudgeSettings],
        Field(
            default=None,
            description="Judge settings; present exactly when the mode is not off.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def default_judge(cls, value):
        """Judging modes resolve their judge eagerly so results record it."""
        if isinstance(value, dict):
            mode = ReasonScoringMode(value.get("mode", ReasonScoringMode.GATE))
            if mode != ReasonScoringMode.OFF and value.get("judge") is None:
                value = {**value, "judge": ReasonJudgeSettings()}
        return value

    @model_validator(mode="after")
    def judge_matches_mode(self) -> "ReasonScoring":
        """The off mode never carries (or silently ignores) judge settings."""
        if (self.mode != ReasonScoringMode.OFF) != (self.judge is not None):
            raise ValueError(
                "judge settings apply exactly to the gate and report modes"
            )
        return self

    @property
    def enabled(self) -> bool:
        """Whether the judge runs."""
        return self.mode != ReasonScoringMode.OFF

    @property
    def gates_reward(self) -> bool:
        """Whether REASON is multiplied into the final reward."""
        return self.mode == ReasonScoringMode.GATE


class ReasonRequestCheck(BaseModel):
    """The judge's verdict on one customer request."""

    model_config = ConfigDict(extra="ignore")
    request: Annotated[
        str, Field(min_length=1, description="The customer request, in a few words.")
    ]
    golden_outcome: Annotated[
        str, Field(description="The golden decision for this request.")
    ]
    agent_outcome: Annotated[
        str,
        Field(description="What the agent told the customer, or 'not communicated'."),
    ]
    decision_correct: Annotated[
        bool,
        Field(description="The agent told the customer the golden outcome."),
    ]
    reason_correct: Annotated[
        bool,
        Field(
            description="The agent's reason rests on the golden reason's deciding "
            "rule and facts. Always false when the decision is wrong."
        ),
    ]
    evidence: Annotated[
        str,
        Field(
            default="", description="Short quote of the agent's decision and reason."
        ),
    ]
    reasoning: Annotated[
        str, Field(default="", description="The judge's explanation for this request.")
    ]

    @property
    def met(self) -> bool:
        """Both the decision and the reason are correct."""
        return self.decision_correct and self.reason_correct


class ReasonJudgeReply(BaseModel):
    """The judge's JSON reply; malformed replies fail validation and are retried."""

    model_config = ConfigDict(extra="ignore")
    requests: Annotated[
        list[ReasonRequestCheck],
        Field(min_length=1, description="One verdict per customer request."),
    ]
    incorrect_claims: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Wrong facts or rules the agent told the customer.",
        ),
    ]
    reasoning: Annotated[
        str, Field(min_length=1, description="One-sentence overall summary.")
    ]


class ReasonCheck(BaseModel):
    """The decision-reason check of one simulation, with everything it was based on.

    ``met`` (and so the REASON reward) is computed in code from the judge's
    per-request verdicts: every request's decision and reason must be correct
    and the agent must have made no incorrect claims.
    """

    golden_decision: str = Field(description="The golden decision judged against.")
    golden_reason: str = Field(description="The golden reason judged against.")
    met: bool = Field(description="Decisions and reasons all correct, no wrong claims.")
    decision_correct: bool = Field(description="Every request's decision is correct.")
    reason_correct: bool = Field(
        description="Every request's reason is correct and no incorrect claims were made."
    )
    partial_score: float = Field(
        description="Share of requests whose decision and reason are both correct."
    )
    requests: list[ReasonRequestCheck] = Field(
        description="The judge's verdict per customer request."
    )
    incorrect_claims: list[str] = Field(
        default_factory=list, description="Wrong facts or rules the agent stated."
    )
    justification: str = Field(description="The judge's overall summary.")
    judge_model: str = Field(description="The judge model that produced the verdict.")
    prompt_version: str = Field(description="The judge prompt version.")
    judge_cost: Optional[float] = Field(
        default=None, description="Cost in USD of the judge call, as reported."
    )

    @property
    def reward(self) -> float:
        """The REASON reward component: 1.0 when met, else 0.0."""
        return 1.0 if self.met else 0.0

    @classmethod
    def from_reply(
        cls,
        reply: ReasonJudgeReply,
        *,
        golden_decision: str,
        golden_reason: str,
        judge: ReasonJudgeSettings,
        judge_cost: Optional[float] = None,
    ) -> "ReasonCheck":
        """Aggregate the judge's per-request verdicts into the simulation's check."""
        decision_correct = all(r.decision_correct for r in reply.requests)
        reason_correct = (
            all(r.met for r in reply.requests) and not reply.incorrect_claims
        )
        return cls(
            golden_decision=golden_decision,
            golden_reason=golden_reason,
            met=decision_correct and reason_correct,
            decision_correct=decision_correct,
            reason_correct=reason_correct,
            partial_score=sum(r.met for r in reply.requests) / len(reply.requests),
            requests=reply.requests,
            incorrect_claims=reply.incorrect_claims,
            justification=reply.reasoning,
            judge_model=judge.model,
            prompt_version=judge.prompt_version,
            judge_cost=judge_cost,
        )
