# Agent Developer Guide

> **Quick start:** see `examples/agents/` for runnable code you can copy and
> modify (`minimal_text_agent.py`, `react_agent.py`, `custom_agent_eval.py`).

## Overview

Agents are the system under test. An agent talks with a simulated customer
and calls the airline tools to resolve the customer's request while following
the airline policy. The evaluator then scores the simulation (see
[`docs/evaluation.md`](../../../docs/evaluation.md)).

Communication is turn-based (half-duplex): the agent receives one message,
returns one message, and the orchestrator routes it to the user or, for tool
calls, to the environment.

## Architecture

```
HalfDuplexParticipant   (base/participant.py)
        |
  HalfDuplexAgent       (base_agent.py)
        |
  LLMAgent, LLMGTAgent  (llm_agent.py; both also mix in LLMConfigMixin)
```

### HalfDuplexAgent

You must implement:

- `get_init_state(message_history=None) -> StateType`: the initial agent
  state. `message_history` is non-empty only when a task starts mid-conversation.
- `generate_next_message(message, state) -> (AssistantMessage, StateType)`:
  given a user or tool message and the current state, produce the next
  assistant message.

Input types: `UserMessage | ToolMessage | MultiToolMessage`
(`ValidAgentInputMessage`). Output type: `AssistantMessage`.

Optional overrides: `stop(message, state)` for cleanup, `set_seed(seed)`, and
`is_stop(message)` to end the conversation from the agent side (the default
never stops; conversations normally end when the user simulator emits
`###STOP###`, `###TRANSFER###` or `###OUT-OF-SCOPE###`).

### Constructor contract

```python
def __init__(self, tools: list[Tool], domain_policy: str):
```

- `tools`: the environment tools the agent can call.
- `domain_policy`: the policy text the agent must follow.

If your agent uses an LLM, mix in `LLMConfigMixin` (from
`tau2.agent.base.llm_config`), which adds `llm: str` and `llm_args: dict`
parameters and a `set_seed()` that writes the seed into `llm_args`.

## Built-in agents

| Registered name | Class | What it does |
|-----------------|-------|--------------|
| `llm_agent` (default) | `LLMAgent` | An LLM with tool calling. The system prompt is a short instruction block plus the policy. |
| `llm_agent_gt` | `LLMGTAgent` | Same, but the prompt also lists the task's reference `actions` as resolution steps. Used to sanity-check the user simulator; only runs on tasks with at least one action. |

```python
from tau2.agent import LLMAgent
from tau2.runner import build_environment
from tau2.runner.build import agent_domain_policy

env = build_environment("airline")
agent = LLMAgent(
    tools=env.get_tools(),
    domain_policy=agent_domain_policy(env),
    llm="openai/gpt-6.1-sol",
    llm_args={"reasoning_effort": "medium"},
)

state = agent.get_init_state()
response, state = agent.generate_next_message(user_msg, state)
```

`agent_domain_policy(env)` is the policy the CLI gives every agent: the
airline `policy.md` followed by an English-language clause
(`AGENT_LANGUAGE_CLAUSE` in `tau2/runner/build.py`). Use it to keep prompts
identical to the τ-bench 2.0 English runs.

## How to develop a new agent

### Step 1: Define a state class

Use a Pydantic `BaseModel` to hold the conversation history and anything else
you track between turns:

```python
from pydantic import BaseModel

from tau2.data_model.message import APICompatibleMessage, SystemMessage


class MyAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
```

### Step 2: Implement the agent

```python
from typing import Optional

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate


class MyAgent(LLMConfigMixin, HalfDuplexAgent[MyAgentState]):
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

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> MyAgentState:
        return MyAgentState(
            system_messages=[SystemMessage(role="system", content=self.domain_policy)],
            messages=list(message_history or []),
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: MyAgentState
    ) -> tuple[AssistantMessage, MyAgentState]:
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

Put `LLMConfigMixin` before `HalfDuplexAgent` in the bases so its `__init__`
consumes `llm` and `llm_args` before the base class runs.

### Step 3: Provide a factory and register it

The runner builds agents through factory functions:

```python
def create_my_agent(tools, domain_policy, **kwargs):
    """kwargs: llm (from --agent-llm), llm_args (--agent-llm-args), task."""
    return MyAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )
```

Register it in `src/tau2/registry.py` so `tau2 run --agent my_agent` can use
it:

```python
registry.register_agent_factory(create_my_agent, "my_agent")
```

`register_agent_factory` also takes an optional `task_filter` (a callable
`Task -> bool`; tasks it rejects are skipped by `run_domain`) and `metadata`.
In a script you can register on the global `tau2.registry.registry` instead
of editing the file (as `examples/agents/minimal_text_agent.py` does).

## Understanding the environment

```bash
tau2 domain airline
```

prints the agent policy and a table of the airline tools (name, read/write
type, description, parameters). `make env-cli` lets you query the tools
interactively.

## Testing your agent

```bash
tau2 run --agent my_agent --agent-llm <model> --user-llm <model> --num-tasks 5
```

See `tau2 run --help` or [`docs/cli-reference.md`](../../../docs/cli-reference.md)
for all options, and [`docs/running_simulations.md`](../../../docs/running_simulations.md)
to run an unregistered agent from Python.
