from copy import deepcopy

import pytest

from tau2.config import (
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_USER,
)
from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.simulation import RunConfig
from tau2.data_model.tasks import EnvAssertion, RewardType, Task, make_task
from tau2.run import (
    EvaluationType,
    get_options,
    get_tasks,
    load_tasks,
    run_domain,
    run_single_task,
    run_tasks,
)

LIVE_LLM = "openai/gpt-6-luna"


def _config(domain_name: str, **overrides) -> RunConfig:
    values = dict(
        domain=domain_name,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=LIVE_LLM,
        llm_args_agent={},
        llm_user=LIVE_LLM,
        llm_args_user={},
        max_concurrency=1,
    )
    values.update(overrides)
    return RunConfig(**values)


def _run_task(
    domain_name: str,
    task: Task,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    **overrides,
):
    return run_single_task(
        _config(domain_name, **overrides), task, evaluation_type=evaluation_type
    )


@pytest.fixture
def run_config() -> RunConfig:
    return _config(
        "mock",
        task_ids=["create_task_1"],
        num_trials=3,
        max_steps=20,
        max_errors=10,
        save_to=None,
        max_concurrency=3,
    )


def test_get_options():
    """Test that we can get available options from the registry"""
    options = get_options()
    assert "airline" in options.domains
    assert set(options.agents) == {"llm_agent", "llm_agent_gt", "gated_cancel_agent"}
    assert options.users == ["user_simulator"]


def test_load_tasks():
    """Test that we can load tasks from the domain"""
    tasks = load_tasks("mock")
    assert len(tasks) > 0


def test_get_tasks():
    """Test that we can load tasks from the domain by id"""
    tasks = get_tasks("mock", task_ids=["create_task_1"])
    assert len(tasks) == 1
    assert tasks[0].id == "create_task_1"


def test_airline_task_splits():
    tasks = load_tasks("airline")
    assert len(tasks) == 50
    train = load_tasks("airline", task_split_name="train")
    test = load_tasks("airline", task_split_name="test")
    assert train and test
    assert not {t.id for t in train} & {t.id for t in test}


def test_run_tasks_offline_with_scripted_llm(monkeypatch, base_task: Task):
    """End-to-end batch run on the mock domain with a scripted LLM."""
    import tau2.agent.llm_agent as llm_agent
    import tau2.user.user_simulator as user_simulator
    from tau2.data_model.semantic_communication import CommunicateScoring

    # Replies are built at call time: trajectories are ordered by timestamp.
    agent_replies = iter(
        [
            lambda: AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="create_task",
                        arguments={"user_id": "user_1", "title": "Important Meeting"},
                    )
                ],
                cost=0.0,
            ),
            lambda: AssistantMessage(role="assistant", content="Done.", cost=0.0),
        ]
    )
    user_replies = iter(
        [
            lambda: AssistantMessage(
                role="assistant",
                content="Please create a task called 'Important Meeting' for user_1.",
                cost=0.0,
            ),
            lambda: AssistantMessage(
                role="assistant", content="Thanks! ###STOP###", cost=0.0
            ),
        ]
    )
    monkeypatch.setattr(llm_agent, "generate", lambda **_: next(agent_replies)())
    monkeypatch.setattr(user_simulator, "generate", lambda **_: next(user_replies)())

    results = run_tasks(
        _config(
            "mock",
            communicate_scoring=CommunicateScoring(scorer="substring"),
        ),
        [base_task],
        console_display=False,
    )
    assert len(results.simulations) == 1
    simulation = results.simulations[0]
    assert simulation.termination_reason == "user_stop"
    assert simulation.reward_info.reward == 1.0
    assert simulation.reward_info.db_check.db_match is True
    assert simulation.agent_cost == 0.0
    assert simulation.user_persona is None  # the mock domain ships no personas


@pytest.mark.live_api
def test_simplified_run(domain_name: str):
    """Test that we can run a task with the mock domain"""
    task = make_task(
        user_instructions="create a task called 'test' for user_1",
        eval_criteria=None,
        initialization_data=None,
        message_history=None,
    )
    simulation = run_tasks(
        _config(
            domain_name,
            llm_agent=DEFAULT_LLM_AGENT,
            llm_args_agent=deepcopy(DEFAULT_LLM_ARGS_AGENT),
            llm_user=DEFAULT_LLM_USER,
            llm_args_user=deepcopy(DEFAULT_LLM_ARGS_USER),
            max_steps=5,
            max_errors=5,
        ),
        [task],
        evaluation_type=EvaluationType.ENV,
        console_display=False,
    )
    assert simulation is not None


@pytest.mark.live_api
def test_run_tasks_base(domain_name: str, base_task: Task):
    """Test running a task with the mock domain"""
    results = run_tasks(_config(domain_name), [base_task])
    # Check that simulation ran and has expected structure
    assert len(results.simulations) == 1
    simulation = results.simulations[0]
    assert simulation.messages is not None
    assert len(simulation.messages) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward is not None


@pytest.mark.live_api
def test_run_task_base(domain_name: str, base_task: Task):
    """Test running a task with the mock domain"""
    simulation = _run_task(domain_name, base_task, EvaluationType.ENV)
    # Check that simulation ran and has expected structure
    assert simulation.messages is not None
    assert len(simulation.messages) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward is not None


@pytest.mark.live_api
def test_run_tasks_message_history(domain_name: str, task_with_message_history: Task):
    """Test running a task with message history"""
    simulation = _run_task(domain_name, task_with_message_history)
    assert simulation is not None


@pytest.mark.live_api
def test_run_tasks_initialization_data(
    domain_name: str, task_with_initialization_data: Task
):
    """Test running a task with initialization data"""
    simulation = _run_task(domain_name, task_with_initialization_data)
    assert simulation is not None


@pytest.mark.live_api
def test_run_tasks_initialization_actions(
    domain_name: str, task_with_initialization_actions: Task
):
    """Test running a task with initialization actions"""
    simulation = _run_task(domain_name, task_with_initialization_actions)
    assert simulation is not None


@pytest.mark.live_api
def test_run_tasks_env_assertions(domain_name: str, task_with_env_assertions: Task):
    """Test running a task with env assertions"""
    simulation = _run_task(domain_name, task_with_env_assertions, EvaluationType.ENV)
    # Check that simulation ran and has expected structure
    assert simulation.messages is not None
    assert len(simulation.messages) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    # These assertions can fail if model is not good enough
    assert simulation.reward_info.reward == 1.0
    assert len(simulation.reward_info.env_assertions) == 1
    assert simulation.reward_info.env_assertions[0].met is True
    # Add an env_assertion that will fail and test that the reward is 0.0
    task_with_env_assertions.evaluation_criteria.env_assertions.append(
        EnvAssertion(
            env_type="assistant",
            func_name="assert_task_status",
            arguments={"task_id": "task_1", "expected_status": "made_up_status"},
        )
    )
    simulation = _run_task(domain_name, task_with_env_assertions, EvaluationType.ENV)
    assert simulation.reward_info.reward == 0.0
    assert len(simulation.reward_info.env_assertions) == 2
    assert simulation.reward_info.env_assertions[0].met is True
    assert simulation.reward_info.env_assertions[1].met is False


@pytest.mark.live_api
def test_run_tasks_history_and_env_assertions(
    domain_name: str, task_with_history_and_env_assertions: Task
):
    """Test running a task with history and env assertions"""
    simulation = _run_task(domain_name, task_with_history_and_env_assertions)
    assert simulation is not None


@pytest.mark.live_api
def test_run_tasks_nl_assertions(domain_name: str):
    """Test running a task with the mock domain"""
    task = get_tasks(domain_name, task_ids=["create_task_1_nl_eval"])[0]
    simulation = _run_task(domain_name, task, EvaluationType.NL_ASSERTIONS)
    # Check that simulation ran and has expected structure
    assert simulation.messages is not None
    assert len(simulation.messages) > 0
    assert simulation.start_time is not None
    assert simulation.end_time is not None
    assert simulation.reward_info.reward == 1.0
    assert len(simulation.reward_info.nl_assertions) == 2
    assert simulation.reward_info.nl_assertions[0].met is True
    assert simulation.reward_info.nl_assertions[1].met is True

    # Add an nl_assertion that will fail and test that the reward is 0.0
    task.evaluation_criteria.nl_assertions.append("The user is not complimented")
    simulation = _run_task(domain_name, task, EvaluationType.NL_ASSERTIONS)
    assert simulation.reward_info.reward == 0.0

    assert len(simulation.reward_info.nl_assertions) == 3
    assert simulation.reward_info.nl_assertions[0].met is True
    assert simulation.reward_info.nl_assertions[1].met is True
    assert simulation.reward_info.nl_assertions[2].met is False


@pytest.mark.xfail(
    reason="Test depends on LLM quality - the model may not consistently recognize impossible tasks"
)
@pytest.mark.live_api
def test_run_tasks_action_checks(domain_name: str, task_with_action_checks: Task):
    """Test running a task with action checks"""
    simulation = _run_task(domain_name, task_with_action_checks)
    assert simulation is not None
    # Following assertions can fail if model is not good enough
    assert simulation.reward_info.reward == 1.0
    assert simulation.reward_info.reward_breakdown[RewardType.DB] == 1.0
    assert simulation.reward_info.reward_breakdown[RewardType.ACTION] == 1.0


@pytest.mark.live_api
def test_run_domain(run_config: RunConfig):
    """Test running a domain with the mock domain"""
    simulation_results = run_domain(run_config)
    assert simulation_results is not None


@pytest.mark.live_api
def test_run_gt_agent(domain_name: str, base_task: Task):
    """Test running gt agent"""
    simulation_results = run_tasks(
        _config(domain_name, agent="llm_agent_gt"), [base_task]
    )
    assert simulation_results is not None
