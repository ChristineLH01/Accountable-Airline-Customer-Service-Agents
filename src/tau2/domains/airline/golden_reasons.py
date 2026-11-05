"""Answer key for the agent's decision explanations (airline).

The key is ``data/tau2-airline/mannual_label.json``: one entry per task with
the agreed ``golden_decision`` and ``golden_reason``. While consensus is being
reached, each entry also carries the annotators' ``golden_decision_1/2/3`` and
``golden_reason_1/2/3``; the loader ignores those, so the file keeps working
once they are removed. An empty final label loads as ``None``.

A judge or evaluator looks a task up with ``load_golden_reasons()[task.id]``
and skips entries whose ``excluded_from_scoring`` is set; the ``scored`` task
split holds the tasks that are not excluded.
"""

import hashlib
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from tau2.domains.airline.utils import AIRLINE_MANUAL_LABELS_PATH


class GoldenReason(BaseModel):
    """The agreed decision on a task's customer requests and its reason."""

    model_config = ConfigDict(extra="ignore")

    task_id: str
    golden_decision: Optional[str] = Field(
        default=None,
        description="What the agent should grant or deny, e.g. 'Deny the "
        "cancellation of reservation EHGLP3.' None until labeled.",
    )
    golden_reason: Optional[str] = Field(
        default=None,
        description="The policy-based reason the agent should give the "
        "customer for that decision. None until labeled.",
    )
    excluded_from_scoring: bool = Field(
        default=False,
        description="Whether the task is left out of scoring and evaluation "
        "runs because its label or answer key is disputed. Such tasks are not "
        "in the 'scored' task split.",
    )
    exclusion_reason: Optional[str] = Field(
        default=None,
        description="Why the task is excluded from scoring.",
    )

    @field_validator("task_id", mode="before")
    @classmethod
    def _task_id_as_str(cls, value: Any) -> str:
        return str(value)

    @field_validator("golden_decision", "golden_reason", mode="before")
    @classmethod
    def _empty_as_none(cls, value: Optional[str]) -> Optional[str]:
        value = (value or "").strip()
        return value or None

    @property
    def is_labeled(self) -> bool:
        """Whether both the decision and the reason have a final label."""
        return bool(self.golden_decision and self.golden_reason)


def load_golden_reasons(
    path: Path = AIRLINE_MANUAL_LABELS_PATH,
) -> dict[str, GoldenReason]:
    """Load the final labels as a mapping from task id to its golden reason."""
    entries = TypeAdapter(list[GoldenReason]).validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    reasons = {entry.task_id: entry for entry in entries}
    if len(reasons) != len(entries):
        raise ValueError(f"Duplicate task_id in {path}")
    return reasons


def golden_labels_sha256(path: Path = AIRLINE_MANUAL_LABELS_PATH) -> str:
    """SHA-256 of the label file, recorded by runs that grade against it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
