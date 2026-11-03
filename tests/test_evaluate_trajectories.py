"""Tests for rescoring trajectories with tau2.scripts.evaluate_trajectories."""

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.simulation import (
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.run import get_tasks
from tau2.scripts import evaluate_trajectories as evaluate_trajectories_module
from tau2.scripts.evaluate_trajectories import compute_simulation_rewards

# ---- Fixtures ----


def _make_info() -> Info:
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
    )


def _make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def _make_sim(task_id: str, messages: list | None = None) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=messages if messages is not None else [],
    )


# ---- Rescoring replays the recorded messages ----


def test_rescoring_replays_recorded_tool_calls():
    """The golden create_task action exists only in the recorded messages, so
    the DB check passes only if rescoring replays them."""
    task = get_tasks("mock", task_ids=["create_task_1"])[0]
    messages = [
        AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="create_task",
                    arguments={"user_id": "user_1", "title": "Important Meeting"},
                )
            ],
            timestamp="2026-01-01T00:00:30",
        ),
        ToolMessage(
            id="call_1",
            role="tool",
            content='{"task_id": "task_2", "title": "Important Meeting", '
            '"description": null, "status": "pending"}',
            requestor="assistant",
            timestamp="2026-01-01T00:00:31",
        ),
    ]
    results = Results(
        info=_make_info(),
        tasks=[task],
        simulations=[_make_sim(task.id, messages=messages)],
    )

    rescored = compute_simulation_rewards(results)

    reward_info = rescored.simulations[0].reward_info
    assert reward_info.db_check is not None
    assert reward_info.db_check.db_match is True
    assert reward_info.action_checks is not None
    assert all(check.action_match for check in reward_info.action_checks)


# ---- Re-grading options: strict_replay, fresh tasks, recorded scorer ----


class TestRegradingOptions:
    def _capture_eval_kwargs(self, monkeypatch, results, **compute_kwargs):
        captured = []

        def fake_evaluate_simulation(**kwargs):
            captured.append(kwargs)
            return RewardInfo(reward=1.0)

        monkeypatch.setattr(
            evaluate_trajectories_module,
            "evaluate_simulation",
            fake_evaluate_simulation,
        )
        compute_simulation_rewards(results, **compute_kwargs)
        return captured

    def test_rescoring_uses_lenient_replay(self, monkeypatch):
        """Re-grading replays historical trajectories whose recorded tool
        outputs may cosmetically predate current tool code; the replay must
        not abort on output-text drift."""
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[_make_sim("t0")],
        )
        captured = self._capture_eval_kwargs(monkeypatch, results)
        assert captured[0]["strict_replay"] is False

    def test_fresh_tasks_reloads_task_definitions(self, monkeypatch):
        """--fresh-tasks must grade against the current data-dir task
        definitions, not the ones embedded in the results file."""
        embedded_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        embedded_task = embedded_task.model_copy(deep=True)
        embedded_task.evaluation_criteria = EvaluationCriteria()  # stale criteria
        results = Results(
            info=_make_info(),
            tasks=[embedded_task],
            simulations=[_make_sim(embedded_task.id)],
        )

        captured = self._capture_eval_kwargs(monkeypatch, results, fresh_tasks=True)
        current_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        assert (
            captured[0]["task"].evaluation_criteria == current_task.evaluation_criteria
        )
        assert captured[0]["task"].evaluation_criteria != EvaluationCriteria()

    def test_embedded_tasks_used_by_default(self, monkeypatch):
        embedded_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        embedded_task = embedded_task.model_copy(deep=True)
        embedded_task.evaluation_criteria = EvaluationCriteria()
        results = Results(
            info=_make_info(),
            tasks=[embedded_task],
            simulations=[_make_sim(embedded_task.id)],
        )
        captured = self._capture_eval_kwargs(monkeypatch, results)
        assert captured[0]["task"].evaluation_criteria == EvaluationCriteria()


def test_rescoring_preserves_recorded_semantic_settings(monkeypatch):
    from tau2.data_model.semantic_communication import (
        CommunicateScoring,
        SemanticCommunicateSettings,
    )
    from tau2.data_model.simulation import RunConfig
    from tau2.runner.batch import _load_run_tasks
    from tau2.runner.helpers import get_info

    config = RunConfig(
        domain="airline",
        num_tasks=1,
        communicate_scoring=CommunicateScoring(
            judge=SemanticCommunicateSettings(
                model="recorded-judge", model_args={"reasoning_effort": "low"}
            )
        ),
    )
    tasks = _load_run_tasks(config)
    info = get_info(config)
    seen = []

    def evaluate(**kwargs):
        seen.append(kwargs)
        return RewardInfo(reward=1)

    monkeypatch.setattr(evaluate_trajectories_module, "evaluate_simulation", evaluate)
    original = Results(
        info=info,
        tasks=tasks,
        simulations=[_make_sim(tasks[0].id), _make_sim(tasks[0].id)],
    )
    restored = Results.model_validate_json(original.model_dump_json())
    rescored = compute_simulation_rewards(restored)
    assert rescored.info == info
    assert len(seen) == 2
    assert all(call["communicate_scoring"] == info.communicate_scoring for call in seen)
    assert all(call["domain"] == "airline" for call in seen)
    assert all(sim.reward_info.reward == 1 for sim in rescored.simulations)
