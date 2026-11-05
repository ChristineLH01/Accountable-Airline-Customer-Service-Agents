"""
Layer 1: Simulation execution.

Runs a pre-built orchestrator and evaluates the result.
No registry dependency, no config parsing, no side effects.
"""

from typing import Optional

from loguru import logger

from tau2.data_model.reason_scoring import ReasonScoring
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import SimulationRun
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.personas import UserPersona


def run_simulation(
    orchestrator: Orchestrator,
    *,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    env_kwargs: Optional[dict] = None,
    communicate_scoring: Optional[CommunicateScoring] = None,
    reason_scoring: Optional[ReasonScoring] = None,
) -> SimulationRun:
    """Run a simulation and evaluate the result.

    Takes a fully constructed orchestrator (with agent, user, environment, and task
    already wired in), runs the simulation, evaluates it, and returns the result
    with reward_info attached.

    This is the lowest-level entry point. It has no dependency on the registry
    or RunConfig -- everything is already encapsulated in the orchestrator.

    Args:
        orchestrator: A fully constructed Orchestrator. Must have agent, user,
            environment, and task set.
        evaluation_type: The type of evaluation to perform. Defaults to ALL.
        env_kwargs: Additional kwargs passed to the evaluator's environment
            constructor.
        communicate_scoring: How `communicate_info` is scored; defaults to the
            pinned semantic LLM judge.
        reason_scoring: How the agent's decision explanations are scored against
            the golden labels; defaults to the reason judge gating the reward.

    Returns:
        SimulationRun with reward_info attached.

    Example:
        # Build your own instances (no registry needed):
        env = MyEnvironment()
        agent = MyAgent(tools=env.get_tools(), domain_policy=env.get_policy())
        user = UserSimulator(llm="openai/gpt-6.1-sol", instructions=str(task.user_scenario),
                             tools=env.get_user_tools())
        orchestrator = Orchestrator(
            domain="airline", agent=agent, user=user,
            environment=env, task=task, max_steps=100,
        )
        result = run_simulation(orchestrator)
        print(result.reward_info.reward)
    """
    communicate_scoring = communicate_scoring or CommunicateScoring()
    reason_scoring = reason_scoring or ReasonScoring()

    # Run the orchestrator
    simulation = orchestrator.run()

    # Save the actual policy used for this simulation
    simulation.policy = orchestrator.environment.get_policy()

    # Record the caller persona, when the user simulator has a named one
    persona = getattr(orchestrator.user, "persona_config", None)
    if isinstance(persona, UserPersona):
        simulation.user_persona = persona.persona_id

    # Extract context from the orchestrator -- no external params needed
    domain = orchestrator.environment.get_domain_name()
    task = orchestrator.task

    # Evaluate
    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=evaluation_type,
        domain=domain,
        env_kwargs=env_kwargs,
        communicate_scoring=communicate_scoring,
        reason_scoring=reason_scoring,
    )
    simulation.reward_info = reward_info

    reason_check = reward_info.reason_check
    logger.info(
        f"Simulation complete: domain={domain}, task={task.id}, "
        f"reward={reward_info.reward}, task_reward={reward_info.task_reward}, "
        f"reason={'n/a' if reason_check is None else reason_check.reward}"
    )

    return simulation
