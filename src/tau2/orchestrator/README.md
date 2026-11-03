# Orchestrator Module

`Orchestrator` (`orchestrator.py`) runs one simulation: it passes messages
between the agent, the user simulator and the environment in turns, executes
tool calls, enforces the termination limits and returns a `SimulationRun`.
It is the only orchestrator; communication is always turn-based
(half-duplex).

## Communication pattern

```
Agent ──message──> User ──message──> Agent ──tool_call──> Environment ──result──> Agent
```

Each step sends the current message from one role to another:

| From → To | What happens |
|-----------|--------------|
| Agent/Env → User | `user.generate_next_message()`; a tool call goes to the environment, text goes to the agent |
| User/Env → Agent | `agent.generate_next_message()`; a tool call goes to the environment, text goes to the user |
| Agent/User → Env | Each tool call runs synchronously via `environment.get_response()`; several results are wrapped in a `MultiToolMessage` and returned to the caller |

A message carries either text or tool calls, never both.

## Starting a conversation

Without `task.initial_state.message_history`, the agent opens with
`"Hi! How can I help you today?"`, or with `task.agent_opener` when the task
sets one. Otherwise the environment is initialized from
`initialization_data`, `initialization_actions` and the message history, and
the conversation resumes from the last message. Seeds are set on both
participants with `set_seed()` before the first turn.

## Termination

| Condition | `TerminationReason` |
|-----------|---------------------|
| User message contains `###STOP###`, `###TRANSFER###` or `###OUT-OF-SCOPE###` | `USER_STOP` |
| `agent.is_stop(message)` is true | `AGENT_STOP` |
| `max_steps` steps taken | `MAX_STEPS` |
| `max_errors` tool calls returned errors | `TOO_MANY_ERRORS` |
| `timeout` seconds elapsed | `TIMEOUT` |
| Message with both text and tool calls, with `validate_communication=True` | `AGENT_ERROR` / `USER_ERROR` |

An empty message (no text, no tool calls) always raises `ValueError` from
`message.validate()`. Step, error and timeout limits are not checked while a
tool call is pending.
Only `USER_STOP` and `AGENT_STOP` are scored; any other reason gives reward 0
(see `docs/evaluation.md`).

## Usage

```python
from tau2.orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator(
    domain="airline",
    agent=agent,  # any HalfDuplexAgent, e.g. LLMAgent
    user=user,  # any HalfDuplexUser, e.g. UserSimulator
    environment=environment,
    task=task,
    max_steps=200,
    max_errors=10,
    seed=42,
    validate_communication=False,
    timeout=None,
)
simulation = orchestrator.run()  # SimulationRun, without reward_info
```

`orchestrator.run()` does not score the simulation. Use
`tau2.runner.run_simulation(orchestrator)` to run and evaluate in one call, or
`tau2.runner.build_orchestrator(config, task)` to build an orchestrator from a
`RunConfig` (see `docs/running_simulations.md`). The constructor defaults are
`max_steps=100` and `max_errors=10`; `RunConfig` and the CLI default to
`max_steps=200`.

## Trajectory

`SimulationRun.messages` is a flat `list[Message]` (`AssistantMessage`,
`UserMessage`, `ToolMessage`), sorted by timestamp, with `turn_idx` set on
each message. Agent and user costs are summed from the messages into
`agent_cost` and `user_cost`.

## Compatible classes

| Role | Classes |
|------|---------|
| Agent | `LLMAgent`, `LLMGTAgent` (`tau2/agent/llm_agent.py`), or any `HalfDuplexAgent` |
| User | `UserSimulator` (`tau2/user/user_simulator.py`), or any `HalfDuplexUser` |
