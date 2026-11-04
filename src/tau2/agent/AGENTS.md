# AGENTS.md — src/tau2/agent/

> See `README.md` for the developer guide with code examples.
> See `base/README.md` for the protocol class and `LLMConfigMixin`.

## Rules for Working in This Directory

### One Base Class

All agents are turn-based (half-duplex) and subclass `HalfDuplexAgent`
(`base_agent.py`), implementing `get_init_state()` and
`generate_next_message()`.

### Constructor Contract

Agents MUST accept `__init__(self, tools: list[Tool], domain_policy: str)`.
For LLM-powered agents, add `LLMConfigMixin`, which provides `llm: str` and
`llm_args: dict`.

### Inheritance Order

Put mixins before the protocol base class:

```python
class MyAgent(LLMConfigMixin, HalfDuplexAgent[MyAgentState]):
    ...
```

`LLMConfigMixin.__init__` consumes `llm` and `llm_args` and forwards the rest
via `super()`. Reversing the order makes those arguments disappear silently.

### Message Constraints

- A message MUST have either `content` (text) OR `tool_calls`, NEVER both and
  never neither. An empty message always raises in `message.validate()`; a
  message with both is ended as `AGENT_ERROR`/`USER_ERROR` only when
  `enforce_communication_protocol` is set.
- `AssistantMessage.is_tool_call()` returns `True` when `tool_calls` is not
  `None`; use `None`, not `[]`, for a text-only message.
- A `MultiToolMessage` input wraps several tool results; unpack
  `message.tool_messages` into the history.

### State Classes

Use Pydantic `BaseModel` for state classes (see `LLMAgentState` in
`llm_agent.py`).

### Prompts Are Pinned

`LLMAgent`'s system prompt (`AGENT_INSTRUCTION` and `SYSTEM_PROMPT` in
`llm_agent.py`), the policy file, `AGENT_DECISION_EXPLANATION_CLAUSE` and
`AGENT_LANGUAGE_CLAUSE` in `tau2/runner/build.py` make up the agent prompt;
without the explanation clause it is the τ-bench 2.0 English agent prompt.
`tests/test_domains/test_airline/test_prompts_airline.py` pins them with
digests; changing any of them changes results comparability, so update the
digests only deliberately.

### Registration

Every agent needs a factory function and a registry entry:

```python
# Factory signature (kwargs: llm, llm_args, task):
def create_my_agent(tools, domain_policy, **kwargs):
    return MyAgent(tools=tools, domain_policy=domain_policy, ...)

# In src/tau2/registry.py:
registry.register_agent_factory(create_my_agent, "my_agent")
```

The registered name is what users pass to `--agent`. Pass
`task_filter=` if the agent can only run some tasks (as `llm_agent_gt` does
with `LLMGTAgent.check_valid_task`).

### Stop Signals

`is_stop()` defaults to `False` for agents; `LLMAgent` does not override it.
Conversations end when the user simulator emits `###STOP###`,
`###TRANSFER###` or `###OUT-OF-SCOPE###`, or when the orchestrator hits
`max_steps`, `max_errors` or `timeout`.

### Testing

- Unit tests: `tests/test_agent.py`
- Orchestration with agents: `tests/test_orchestrator.py`
