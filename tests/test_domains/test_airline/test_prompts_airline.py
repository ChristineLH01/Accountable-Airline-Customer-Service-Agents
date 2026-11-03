"""The airline agent and user-simulator prompts stay those of tau-bench 2.0 (English).

The digest below was computed from the system prompts that the unmodified
tau-bench 2.0 code sent for every airline task (English text run, default
seed). If a change to the policy, tasks, scaffold, personas or language clause
is intended, update the digest and say so in the change description: results
are only comparable between runs with identical prompts.
"""

import hashlib
import json

from tau2.data_model.simulation import RunConfig
from tau2.run import get_tasks
from tau2.runner.build import AGENT_LANGUAGE_CLAUSE, build_orchestrator
from tau2.user.personas import load_persona_catalog

TAU_BENCH_2_ENGLISH_AIRLINE_PROMPTS_SHA256 = (
    "5babeadd9f68e53f9493cafa2d57cb4c81c06c606249aba931e778c7d1a22e44"
)


def _system_prompts() -> tuple[list[str], list[str]]:
    config = RunConfig()
    agent_prompts, user_prompts = [], []
    for task in get_tasks("airline"):
        orchestrator = build_orchestrator(config, task)
        agent_prompts.append(orchestrator.agent.system_prompt)
        user_prompts.append(orchestrator.user.system_prompt)
    return agent_prompts, user_prompts


def test_prompts_match_tau_bench_2_english():
    agent_prompts, user_prompts = _system_prompts()
    digest = hashlib.sha256(json.dumps([agent_prompts, user_prompts]).encode())
    assert digest.hexdigest() == TAU_BENCH_2_ENGLISH_AIRLINE_PROMPTS_SHA256


def test_prompt_components():
    agent_prompts, user_prompts = _system_prompts()
    catalog = load_persona_catalog("airline")
    assert all(
        prompt.endswith(AGENT_LANGUAGE_CLAUSE + "\n</policy>")
        for prompt in agent_prompts
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
