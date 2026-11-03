"""
Base agent components.

This module exports the foundational building blocks for creating
turn-based (half-duplex) conversation participants.
"""

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base.participant import HalfDuplexParticipant

__all__ = [
    "HalfDuplexParticipant",
    "LLMConfigMixin",
]
