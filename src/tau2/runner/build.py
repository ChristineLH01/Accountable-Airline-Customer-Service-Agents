"""
Layer 2: Build functions.

Turn config/names into live instances (environment, agent, user, orchestrator).
Uses the registry for name resolution. Callers who want full control can skip
this layer and construct instances directly.
"""

import uuid
from typing import Optional

from loguru import logger

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.persona import PersonaConfig
from tau2.data_model.simulation import RunConfig
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import Tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.personas import assign_user_persona
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import HalfDuplexUser

# Language instruction appended to the domain policy the agent sees. It is the
# English clause of tau-bench 2.0, so prompts match the released English runs.
AGENT_LANGUAGE_CLAUSE = (
    "Respond in English using Latin script; keep all tool calls, arguments, "
    "and structured outputs in English exactly as specified."
)

# =============================================================================
# Low-level build functions (no RunConfig needed)
# =============================================================================


def build_environment(
    domain: str,
    *,
    env_kwargs: Optional[dict] = None,
) -> Environment:
    """Build an environment from a domain name.

    Uses the registry to resolve the domain name to an environment constructor.

    Args:
        domain: Domain name (e.g., "airline").
        env_kwargs: Additional keyword arguments passed to the environment
            constructor (e.g., a custom ``db`` or ``policy``).

    Returns:
        A fully constructed Environment instance.
    """
    env_constructor = registry.get_env_constructor(domain)
    return env_constructor(**dict(env_kwargs or {}))


def agent_domain_policy(environment: Environment) -> str:
    """The policy given to the agent: the domain policy plus the language clause."""
    return environment.get_policy() + "\n\n" + AGENT_LANGUAGE_CLAUSE


def user_tools_for_task(environment: Environment, task: Task) -> Optional[list[Tool]]:
    """Select the same optional caller tools for execution and prompt review."""
    try:
        return environment.get_user_tools(include=task.user_tools) or None
    except Exception:
        return None


def build_agent(
    agent_name: str,
    environment: Environment,
    *,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    task: Optional[Task] = None,
) -> HalfDuplexAgent:
    """Build an agent from a registered name and an environment.

    Uses the registry to resolve the agent name to a factory function,
    then calls it with the appropriate parameters.

    Args:
        agent_name: Registered agent name (e.g., "llm_agent", "llm_agent_gt").
        environment: The environment to extract tools and policy from.
        llm: LLM model name for the agent.
        llm_args: LLM arguments for the agent.
        task: The task (required for some agents like llm_agent_gt).

    Returns:
        A fully constructed agent instance.

    Raises:
        ValueError: If the agent name has no factory registered.
    """
    agent_factory = registry.get_agent_factory(agent_name)
    if agent_factory is None:
        raise ValueError(
            f"Agent '{agent_name}' has no factory registered. "
            f"Register a factory with registry.register_agent_factory()."
        )
    return agent_factory(
        tools=environment.get_tools(),
        domain_policy=agent_domain_policy(environment),
        llm=llm,
        llm_args=llm_args,
        task=task,
    )


def build_user(
    user_name: str,
    environment: Environment,
    task: Task,
    *,
    llm: Optional[str] = None,
    llm_args: Optional[dict] = None,
    persona_config: Optional[PersonaConfig] = None,
) -> HalfDuplexUser:
    """Build a user simulator from a registered name.

    Uses the registry to resolve the user name to a constructor.

    Args:
        user_name: Registered user name (e.g., "user_simulator").
        environment: The environment to extract user tools from.
        task: The task (used for user instructions).
        llm: LLM model name for the user simulator.
        llm_args: LLM arguments for the user simulator.
        persona_config: Persona of the caller (see ``tau2.user.personas``).

    Returns:
        A fully constructed user instance.
    """
    UserConstructor = registry.get_user_constructor(user_name)
    user_kwargs = {
        "tools": user_tools_for_task(environment, task),
        "instructions": str(task.user_scenario),
        "llm": llm,
        "llm_args": llm_args,
    }
    if issubclass(UserConstructor, UserSimulator):
        user_kwargs["persona_config"] = persona_config
    return UserConstructor(**user_kwargs)


# =============================================================================
# High-level build function (uses RunConfig)
# =============================================================================


def build_orchestrator(
    config: RunConfig,
    task: Task,
    *,
    seed: Optional[int] = None,
    simulation_id: Optional[str] = None,
) -> Orchestrator:
    """Build a ready-to-run orchestrator from a RunConfig and task.

    The caller persona is assigned from the run seed (``config.seed``) and the
    task ID, so every trial of a task uses the same persona.

    Args:
        config: Run configuration.
        task: The task to run.
        seed: Per-trial seed. If None, uses config.seed.
        simulation_id: Unique simulation ID. If None, a UUID is generated.

    Returns:
        A fully constructed Orchestrator, ready for run_simulation().

    Example:
        config = RunConfig(domain="airline", agent="llm_agent")
        tasks = get_tasks("airline")
        orchestrator = build_orchestrator(config, tasks[0], seed=42)
        result = run_simulation(orchestrator)
    """
    if simulation_id is None:
        simulation_id = str(uuid.uuid4())
    if seed is None:
        seed = config.seed

    domain = config.domain
    environment = build_environment(domain)

    agent = build_agent(
        config.agent,
        environment,
        llm=config.llm_agent,
        llm_args=config.llm_args_agent,
        task=task,
    )

    user = build_user(
        config.user,
        environment,
        task,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
        persona_config=assign_user_persona(domain, task.id, config.seed),
    )

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        seed=seed,
        simulation_id=simulation_id,
        validate_communication=config.enforce_communication_protocol,
        timeout=config.timeout,
    )

    logger.debug(
        f"Built orchestrator: domain={domain}, agent={config.agent}, "
        f"user={config.user}, task={task.id}"
    )

    return orchestrator
