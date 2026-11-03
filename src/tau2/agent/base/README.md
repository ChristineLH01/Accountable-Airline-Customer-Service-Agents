# Base Agent Components

This directory contains the building blocks shared by agents and user
simulators: the turn-based protocol class and the LLM configuration mixin.

| File | Purpose |
|------|---------|
| `participant.py` | `HalfDuplexParticipant`, the turn-based protocol base class |
| `llm_config.py` | `LLMConfigMixin`, shared LLM configuration |

Every participant is turn-based.

---

## `HalfDuplexParticipant[InputMessageType, OutputMessageType, StateType]`

Turn-based communication: a participant receives one message and returns one
message.

```python
class HalfDuplexParticipant(ABC, Generic[InputMessageType, OutputMessageType, StateType]):

    @abstractmethod
    def generate_next_message(
        self, message: InputMessageType, state: StateType
    ) -> tuple[OutputMessageType, StateType]:
        """Generate a complete response to an input message."""

    @abstractmethod
    def get_init_state(self, message_history: Optional[list[Message]] = None) -> StateType:
        """Get the initial state."""

    def stop(self, message=None, state=None) -> None:
        """Stop the participant (cleanup). No-op by default."""

    @classmethod
    def is_stop(cls, message: OutputMessageType) -> bool:
        """Whether the message ends the conversation. False by default."""

    def set_seed(self, seed: int):
        """Set the random seed. Logs a warning by default."""
```

Subclasses:

| Class | Location | Input | Output |
|-------|----------|-------|--------|
| `HalfDuplexAgent` | `tau2/agent/base_agent.py` | `UserMessage \| ToolMessage \| MultiToolMessage` | `AssistantMessage` |
| `HalfDuplexUser` | `tau2/user/user_simulator_base.py` | `AssistantMessage \| ToolMessage \| MultiToolMessage` | `UserMessage` |

`HalfDuplexAgent.__init__(tools, domain_policy)` stores `self.tools` and
`self.domain_policy`; `HalfDuplexUser.__init__(instructions=None, tools=None)`
stores `self.instructions` and `self.tools`.

---

## `LLMConfigMixin`

Shared LLM configuration for any LLM-powered participant. Used by `LLMAgent`,
`LLMGTAgent` and `UserSimulator`.

```python
class LLMConfigMixin:
    def __init__(self, *args, llm: str, llm_args: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}

    def set_seed(self, seed: int):
        """Write the seed into llm_args (warns if one was already set)."""
        self.llm_args["seed"] = seed
```

List it **before** the protocol base class so its `__init__` consumes `llm`
and `llm_args`:

```python
class MyAgent(LLMConfigMixin, HalfDuplexAgent[MyAgentState]):
    ...
```

The orchestrator calls `set_seed()` on both participants with the per-trial
seed before the conversation starts.

---

## Example: a half-duplex LLM agent

```python
from typing import Optional

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate


class MyHalfDuplexAgent(LLMConfigMixin, HalfDuplexAgent[LLMAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
    ):
        super().__init__(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )

    def get_init_state(self, message_history=None) -> LLMAgentState:
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.domain_policy)],
            messages=list(message_history or []),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            **self.llm_args,
        )
        state.messages.append(response)
        return response, state
```

See `tau2/agent/llm_agent.py` for the built-in `LLMAgent` and
`tau2/user/user_simulator.py` for the built-in `UserSimulator`.
