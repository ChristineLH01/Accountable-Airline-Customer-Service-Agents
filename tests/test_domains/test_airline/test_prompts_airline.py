"""The airline agent and user-simulator prompts are pinned.

They are tau-bench 2.0's English airline prompts (English text run, default
seed) with one project addition: the agent's policy also carries
``AGENT_DECISION_EXPLANATION_CLAUSE``, which asks the agent to explain each
decision on a customer request. ``TAU_BENCH_2_ENGLISH_AIRLINE_PROMPTS_SHA256``
is the digest of the unmodified tau-bench 2.0 prompts and must still match once
that clause is removed; ``PROJECT_AIRLINE_PROMPTS_SHA256`` pins the prompts as
they are sent. If a change to the policy, tasks, scaffold, personas or clauses
is intended, update the digests and say so in the change description: results
are only comparable between runs with identical prompts.
"""

import hashlib
import json

from tau2.data_model.simulation import RunConfig
from tau2.run import get_tasks
from tau2.runner.build import (
    AGENT_DECISION_EXPLANATION_CLAUSE,
    AGENT_LANGUAGE_CLAUSE,
    build_orchestrator,
)
from tau2.user.personas import load_persona_catalog

TAU_BENCH_2_ENGLISH_AIRLINE_PROMPTS_SHA256 = (
    "5babeadd9f68e53f9493cafa2d57cb4c81c06c606249aba931e778c7d1a22e44"
)
PROJECT_AIRLINE_PROMPTS_SHA256 = (
    "828d98e583c96fdf87ec05ed5a9b864c37bb6845c795d044bdb7606e1d6f76d3"
)


def _system_prompts() -> tuple[list[str], list[str]]:
    config = RunConfig()
    agent_prompts, user_prompts = [], []
    for task in get_tasks("airline"):
        orchestrator = build_orchestrator(config, task)
        agent_prompts.append(orchestrator.agent.system_prompt)
        user_prompts.append(orchestrator.user.system_prompt)
    return agent_prompts, user_prompts


def _digest(agent_prompts: list[str], user_prompts: list[str]) -> str:
    return hashlib.sha256(
        json.dumps([agent_prompts, user_prompts]).encode()
    ).hexdigest()


def test_prompts_are_pinned():
    agent_prompts, user_prompts = _system_prompts()
    assert _digest(agent_prompts, user_prompts) == PROJECT_AIRLINE_PROMPTS_SHA256


def test_prompts_match_tau_bench_2_english_apart_from_explanation_clause():
    agent_prompts, user_prompts = _system_prompts()
    without_clause = [
        prompt.replace("\n\n" + AGENT_DECISION_EXPLANATION_CLAUSE, "", 1)
        for prompt in agent_prompts
    ]
    assert _digest(without_clause, user_prompts) == (
        TAU_BENCH_2_ENGLISH_AIRLINE_PROMPTS_SHA256
    )


def test_prompt_components():
    agent_prompts, user_prompts = _system_prompts()
    catalog = load_persona_catalog("airline")
    clauses = (
        AGENT_DECISION_EXPLANATION_CLAUSE
        + "\n\n"
        + AGENT_LANGUAGE_CLAUSE
        + "\n</policy>"
    )
    assert all(prompt.endswith(clauses) for prompt in agent_prompts)
    assert all(
        prompt.count(AGENT_DECISION_EXPLANATION_CLAUSE) == 1 for prompt in agent_prompts
    )
    # Task 0's caller (Emma Kim) is female; every task names exactly one persona.
    assert "You are a female customer (she/her)." in user_prompts[0]
    for prompt in user_prompts:
        assert "<PERSONA_GUIDELINES>" not in prompt
        matches = [
            persona_id
            for persona_id, persona in catalog.personas.items()
            if persona.gender_guidance in prompt
        ]
        assert len(matches) == 1


def test_persona_assignment_follows_caller_gender():
    catalog = load_persona_catalog("airline")
    assert len(catalog.caller_genders) == 50
    for task_id, gender in catalog.caller_genders.items():
        for seed in (0, 300, 12345):
            assert catalog.assign(task_id, seed).gender == gender
    # Tasks without a recorded caller gender still get a deterministic persona.
    assert catalog.assign("new-task", 300) == catalog.assign("new-task", 300)
