"""Decision-reason scoring: the judge contract, the pass rule, reward gating,
skips, recorded settings, metrics and CLI plumbing. No API calls."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tau2.config import DEFAULT_LLM_REASON_JUDGE, DEFAULT_LLM_REASON_JUDGE_ARGS
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.reason_scoring import (
    ReasonJudgeReply,
    ReasonJudgeSettings,
    ReasonScoring,
    ReasonScoringMode,
)
from tau2.data_model.simulation import (
    Info,
    Results,
    RewardInfo,
    RunConfig,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import RewardType, Task
from tau2.domains.airline.golden_reasons import GoldenReason
from tau2.environment.environment import EnvironmentInfo
from tau2.evaluator import evaluator_reason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

GOLDEN = GoldenReason(
    task_id="fixture",
    golden_decision="Deny the cancellation of reservation Q69X3R.",
    golden_reason="Q69X3R is economy with no insurance, booked 29 hours ago.",
)


def task():
    return Task(
        id="fixture",
        user_scenario={"instructions": "Ask to cancel Q69X3R."},
        evaluation_criteria={"reward_basis": ["DB"]},
    )


def simulation(termination=TerminationReason.USER_STOP):
    return SimulationRun(
        id="sim",
        task_id="fixture",
        start_time="start",
        end_time="end",
        duration=1,
        termination_reason=termination,
        messages=[
            UserMessage(role="user", content="Please cancel Q69X3R."),
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="get_reservation_details",
                        arguments={"reservation_id": "Q69X3R"},
                    )
                ],
            ),
            ToolMessage(id="c1", role="tool", content='{"secret": "TOOL-RESULT"}'),
            AssistantMessage(
                role="assistant",
                content="I can't cancel it: it was booked over 24 hours ago.",
            ),
        ],
    )


def reply(*requests, incorrect_claims=()):
    return json.dumps(
        {
            "requests": [
                {
                    "request": f"request {i}",
                    "golden_outcome": "deny",
                    "agent_outcome": "denied",
                    "decision_correct": decision,
                    "reason_correct": reason,
                    "evidence": "booked over 24 hours ago",
                    "reasoning": "checked",
                }
                for i, (decision, reason) in enumerate(requests)
            ],
            "incorrect_claims": list(incorrect_claims),
            "reasoning": "summary",
        }
    )


def gate():
    return ReasonScoring(
        judge=ReasonJudgeSettings(
            model="explicit-reason-judge", model_args={"reasoning_effort": "low"}
        )
    )


@pytest.fixture
def judged(monkeypatch):
    """Patch the DB/action evaluators to pass and the judge to a fixed reply."""
    import tau2.evaluator.evaluator as evaluator

    monkeypatch.setattr(
        evaluator.EnvironmentEvaluator,
        "calculate_reward",
        lambda **_: RewardInfo(reward=1, reward_breakdown={RewardType.DB: 1}),
    )
    monkeypatch.setattr(
        evaluator.ActionEvaluator,
        "calculate_reward",
        lambda **_: RewardInfo(reward=1, reward_breakdown={RewardType.ACTION: 1}),
    )
    monkeypatch.setattr(evaluator_reason, "get_golden_reason", lambda *_: GOLDEN)
    calls = []

    def set_reply(content, cost=0.002):
        def generate(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=content, cost=cost)

        monkeypatch.setattr(evaluator_reason, "generate", generate)

    return SimpleNamespace(calls=calls, set_reply=set_reply)


def evaluate(scoring, *, evaluation_type=EvaluationType.ALL, sim=None):
    return evaluate_simulation(
        sim or simulation(),
        task(),
        evaluation_type,
        "airline",
        communicate_scoring=None,
        reason_scoring=scoring,
    )


# ---- Settings ----


def test_default_scoring_gates_with_the_default_judge():
    scoring = RunConfig().reason_scoring
    assert scoring.mode == ReasonScoringMode.GATE and scoring.gates_reward
    assert scoring.judge.model == DEFAULT_LLM_REASON_JUDGE
    assert scoring.judge.model_args == DEFAULT_LLM_REASON_JUDGE_ARGS
    assert scoring.judge.prompt_version == "reason-judge-v1"
    assert ReasonScoring(mode="report").judge is not None
    assert not ReasonScoring(mode="report").gates_reward
    assert ReasonScoring(mode="off").judge is None
    with pytest.raises(ValueError, match="judge settings"):
        ReasonScoring(mode="off", judge=ReasonJudgeSettings())


def test_judge_prompt_is_pinned():
    digest = hashlib.sha256(
        (
            evaluator_reason.REASON_JUDGE_SYSTEM_PROMPT
            + evaluator_reason.REASON_JUDGE_USER_PROMPT_TEMPLATE
        ).encode()
    ).hexdigest()
    # Changing the judge prompt changes scores: bump ReasonJudgeSettings.
    # prompt_version and update this digest together.
    assert digest == "7b896b50cb65b9d78d33b55c0f99d18c8f6f2ed7c1354d018fa7958f45044e84"


def test_transcript_shows_messages_and_tool_calls_but_not_tool_results():
    transcript = evaluator_reason.format_reason_transcript(simulation().messages)
    assert transcript.splitlines() == [
        "customer: Please cancel Q69X3R.",
        'agent action: get_reservation_details({"reservation_id": "Q69X3R"})',
        "agent: I can't cancel it: it was booked over 24 hours ago.",
    ]
    assert "TOOL-RESULT" not in transcript


# ---- Judging and the pass rule ----


def test_wrong_reason_gates_the_reward_and_records_everything(judged):
    judged.set_reply(reply((True, False)))
    result = evaluate(gate())
    assert result.reward == 0
    assert result.task_reward == 1
    assert result.reward_breakdown[RewardType.REASON] == 0
    assert result.reward_breakdown[RewardType.DB] == 1
    assert RewardType.REASON in result.reward_basis
    check = result.reason_check
    assert check.decision_correct and not check.reason_correct and not check.met
    assert check.golden_decision == GOLDEN.golden_decision
    assert check.golden_reason == GOLDEN.golden_reason
    assert check.judge_model == "explicit-reason-judge"
    assert check.prompt_version == "reason-judge-v1"
    assert check.judge_cost == 0.002
    assert result.info["reason"]["reason_scoring"] == gate().model_dump(mode="json")

    (call,) = judged.calls
    assert call["call_name"] == "reason_eval"
    assert call["model"] == "explicit-reason-judge"
    assert call["reasoning_effort"] == "low"
    assert call["response_model"] is ReasonJudgeReply
    assert call["response_validation_retries"] == 2
    assert call["messages"][0].content == evaluator_reason.REASON_JUDGE_SYSTEM_PROMPT
    prompt = call["messages"][1].content
    assert GOLDEN.golden_decision in prompt and GOLDEN.golden_reason in prompt
    assert evaluator_reason.get_policy("airline") in prompt
    assert prompt.startswith("POLICY:\n# Airline Agent Policy")
    assert "agent: I can't cancel it" in prompt


def test_correct_decision_and_reason_pass(judged):
    judged.set_reply(reply((True, True)))
    result = evaluate(gate())
    assert result.reward == 1 and result.task_reward == 1
    assert result.reason_check.met and result.reason_check.partial_score == 1


def test_an_incorrect_claim_fails_even_when_every_request_passes(judged):
    judged.set_reply(
        reply((True, True), incorrect_claims=["said it was basic economy"])
    )
    check = evaluate(gate()).reason_check
    assert check.decision_correct and not check.reason_correct and not check.met
    assert check.partial_score == 1
    assert check.incorrect_claims == ["said it was basic economy"]


def test_every_request_must_pass_and_partial_score_counts_them(judged):
    judged.set_reply(reply((True, True), (False, False)))
    result = evaluate(gate())
    assert result.reward == 0
    assert not result.reason_check.decision_correct
    assert result.reason_check.partial_score == 0.5


def test_failed_db_check_is_not_rescued_by_a_correct_reason(judged, monkeypatch):
    import tau2.evaluator.evaluator as evaluator

    monkeypatch.setattr(
        evaluator.EnvironmentEvaluator,
        "calculate_reward",
        lambda **_: RewardInfo(reward=0, reward_breakdown={RewardType.DB: 0}),
    )
    judged.set_reply(reply((True, True)))
    result = evaluate(gate())
    assert result.reward == 0 and result.task_reward == 0
    assert result.reward_breakdown[RewardType.REASON] == 1


def test_report_mode_records_the_verdict_without_gating(judged):
    judged.set_reply(reply((True, False)))
    result = evaluate(ReasonScoring(mode="report"))
    assert result.reward == 1 and result.task_reward == 1
    assert not result.reason_check.met
    assert RewardType.REASON not in result.reward_breakdown
    assert RewardType.REASON not in result.reward_basis


def test_malformed_judge_reply_raises(judged):
    judged.set_reply('{"requests": [], "reasoning": "nothing"}')
    with pytest.raises(ValueError):
        evaluate(gate())


# ---- Skips ----


def no_judge_calls(monkeypatch):
    def generate(**_):
        raise AssertionError("the judge must not be called")

    monkeypatch.setattr(evaluator_reason, "generate", generate)


def test_excluded_and_unlabeled_tasks_are_skipped_without_a_call(judged, monkeypatch):
    no_judge_calls(monkeypatch)
    excluded = GOLDEN.model_copy(
        update={"excluded_from_scoring": True, "exclusion_reason": "disputed key"}
    )
    for golden, why in [
        (excluded, "excluded from scoring: disputed key"),
        (GoldenReason(task_id="fixture"), "no final golden label for this task"),
        (None, "no final golden label for this task"),
    ]:
        monkeypatch.setattr(evaluator_reason, "get_golden_reason", lambda *_: golden)
        result = evaluate(gate())
        assert result.reward == 1 and result.task_reward == 1
        assert result.reason_check is None
        assert result.info["reason"] == {"skipped": why}
        assert RewardType.REASON not in result.reward_breakdown


def test_off_none_single_components_and_unfinished_runs_skip(judged, monkeypatch):
    no_judge_calls(monkeypatch)
    for scoring in (None, ReasonScoring(mode="off")):
        result = evaluate(scoring)
        assert result.reason_check is None and result.task_reward is None
        assert "reason" not in (result.info or {})
    assert evaluate(gate(), evaluation_type=EvaluationType.ENV).reason_check is None
    unfinished = evaluate(gate(), sim=simulation(TerminationReason.MAX_STEPS))
    assert unfinished.reward == 0 and unfinished.reason_check is None


def test_airline_labels_are_looked_up_by_task_id():
    scored = evaluator_reason.get_golden_reason("airline", "1")
    assert scored.is_labeled and not scored.excluded_from_scoring
    assert evaluator_reason.get_golden_reason("airline", "39").excluded_from_scoring
    assert evaluator_reason.get_golden_reason("mock", "1") is None


# ---- Recording, resume, metrics and CLI ----


def test_run_info_records_scorer_and_label_digest_and_resume_rejects_changes(
    tmp_path,
):
    from tau2.domains.airline.golden_reasons import golden_labels_sha256
    from tau2.runner.checkpoint import try_resume
    from tau2.runner.helpers import get_info

    info = get_info(RunConfig(domain="airline", reason_scoring=gate()))
    assert info.reason_scoring == gate()
    assert info.golden_labels_sha256 == golden_labels_sha256()
    assert (
        get_info(
            RunConfig(domain="airline", reason_scoring=ReasonScoring(mode="off"))
        ).golden_labels_sha256
        is None
    )

    previous = Results(info=info, tasks=[task()], simulations=[])
    path = tmp_path / "results.json"
    previous.save(path)
    before = path.read_bytes()

    other_mode = previous.model_copy(deep=True)
    other_mode.info.reason_scoring = ReasonScoring(mode="report")
    with pytest.raises(ValueError, match="Decision-reason scorer changed"):
        try_resume(path, other_mode, [task()], 1, auto_resume=True)

    other_labels = previous.model_copy(deep=True)
    other_labels.info.golden_labels_sha256 = "0" * 64
    with pytest.raises(ValueError, match="Golden labels changed"):
        try_resume(path, other_labels, [task()], 1, auto_resume=True)
    assert path.read_bytes() == before


def test_results_without_reason_scoring_read_as_off():
    info = Info(
        git_commit="abc",
        num_trials=1,
        max_steps=10,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="airline", policy="p"),
    )
    assert info.reason_scoring is None
    assert info.effective_reason_scoring == ReasonScoring(mode="off")


def test_metrics_report_reason_accuracy_and_task_reward(judged, monkeypatch):
    from tau2.metrics.agent_metrics import compute_metrics

    info = Info(
        git_commit="abc",
        num_trials=1,
        max_steps=10,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="airline", policy="p"),
    )
    sims = []
    for sim_id, content in [("a", reply((True, True))), ("b", reply((True, False)))]:
        judged.set_reply(content, cost=0.01)
        sim = simulation().model_copy(update={"id": sim_id})
        sim.reward_info = evaluate(gate(), sim=sim)
        sims.append(sim)
    monkeypatch.setattr(evaluator_reason, "get_golden_reason", lambda *_: None)
    skipped = simulation().model_copy(update={"id": "c"})
    skipped.reward_info = evaluate(gate(), sim=skipped)
    sims.append(skipped)
    sims = [sim.model_copy(update={"task_id": f"t{i}"}) for i, sim in enumerate(sims)]
    tasks = [task().model_copy(update={"id": f"t{i}"}) for i in range(3)]

    metrics = compute_metrics(Results(info=info, tasks=tasks, simulations=sims))
    assert metrics.avg_reward == pytest.approx(2 / 3)  # b fails REASON
    assert metrics.avg_task_reward == pytest.approx(1)
    assert metrics.reason_in_reward
    assert metrics.reason_judged == 2 and metrics.reason_met == 1
    assert metrics.reason_decision_correct == 2 and metrics.reason_reason_correct == 1
    assert metrics.reason_skipped == 1
    assert metrics.reason_accuracy == 0.5
    assert metrics.avg_reason_partial == pytest.approx(0.5)
    assert metrics.reason_judge_cost == pytest.approx(0.02)
    assert metrics.as_dict()["reason_accuracy"] == 0.5


def test_cli_reason_flags_build_the_run_config():
    import argparse

    from tau2.cli import add_run_args, config_from_args

    parser = argparse.ArgumentParser()
    add_run_args(parser)
    default = config_from_args(parser.parse_args([]))
    assert default.reason_scoring == ReasonScoring()

    report = config_from_args(
        parser.parse_args(
            [
                "--reason-scoring",
                "report",
                "--reason-judge",
                '{"model": "other-judge", "model_args": {"reasoning_effort": "high"}}',
            ]
        )
    )
    assert report.reason_scoring.mode == ReasonScoringMode.REPORT
    assert report.reason_scoring.judge.model == "other-judge"
    off = config_from_args(parser.parse_args(["--reason-scoring", "off"]))
    assert off.reason_scoring == ReasonScoring(mode="off")
    with pytest.raises(ValueError, match="--reason-judge"):
        config_from_args(
            parser.parse_args(["--reason-scoring", "off", "--reason-judge", "{}"])
        )


def test_rescoring_can_add_reason_scoring_to_old_results(monkeypatch):
    import tau2.scripts.evaluate_trajectories as module
    from tau2.domains.airline.golden_reasons import golden_labels_sha256

    info = Info(
        git_commit="abc",
        num_trials=1,
        max_steps=10,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="airline", policy="p"),
    )
    seen = []
    monkeypatch.setattr(
        module,
        "evaluate_simulation",
        lambda **kwargs: seen.append(kwargs) or RewardInfo(reward=1),
    )
    results = Results(info=info, tasks=[task()], simulations=[simulation()])

    recorded = module.compute_simulation_rewards(results)
    assert seen[-1]["reason_scoring"] == ReasonScoring(mode="off")
    assert recorded.info.golden_labels_sha256 is None

    rescored = module.compute_simulation_rewards(results, reason_scoring=gate())
    assert seen[-1]["reason_scoring"] == gate()
    assert rescored.info.reason_scoring == gate()
    assert rescored.info.golden_labels_sha256 == golden_labels_sha256()
