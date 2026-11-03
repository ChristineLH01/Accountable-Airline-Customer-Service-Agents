from types import SimpleNamespace

from tau2.evaluator import reviewer
from tau2.runner import batch


def test_review_simulation_uses_custom_model_for_full_review(monkeypatch):
    captured = {}
    expected_review = object()
    expected_auth = object()
    review_model = "openai/gpt-6-luna"

    def fake_review(**kwargs):
        captured["review_model"] = kwargs["review_model"]
        captured["full_trajectory"] = kwargs["full_trajectory"]
        return expected_review

    def fake_classify(**kwargs):
        captured["auth_model"] = kwargs["model"]
        captured["auth_messages"] = kwargs["messages"]
        return expected_auth

    monkeypatch.setattr(
        reviewer.ConversationReviewer,
        "review",
        staticmethod(fake_review),
    )
    monkeypatch.setattr(
        reviewer.AuthenticationClassifier,
        "classify",
        staticmethod(fake_classify),
    )

    simulation = SimpleNamespace(messages=["message"])

    review, auth = reviewer.review_simulation(
        simulation=simulation,
        task=object(),
        mode=reviewer.ReviewMode.FULL,
        user_info=object(),
        policy="policy",
        review_model=review_model,
    )

    assert review is expected_review
    assert auth is expected_auth
    assert captured == {
        "review_model": review_model,
        "full_trajectory": ["message"],
        "auth_model": review_model,
        "auth_messages": ["message"],
    }


def test_review_simulation_uses_custom_model_for_user_only_review(
    monkeypatch,
):
    captured = {}
    expected_review = object()
    review_model = "gemini/gemini-2.5-pro"

    def fake_review(**kwargs):
        captured["review_model"] = kwargs["review_model"]
        captured["full_trajectory"] = kwargs["full_trajectory"]
        return expected_review

    monkeypatch.setattr(
        reviewer.UserOnlyReviewer,
        "review",
        staticmethod(fake_review),
    )

    simulation = SimpleNamespace(messages=["message"])

    review, auth = reviewer.review_simulation(
        simulation=simulation,
        task=object(),
        mode=reviewer.ReviewMode.USER,
        user_info=object(),
        review_model=review_model,
    )

    assert review is expected_review
    assert auth is None
    assert captured == {
        "review_model": review_model,
        "full_trajectory": ["message"],
    }


def test_auto_review_uses_custom_review_model(monkeypatch):
    captured = {}
    expected_review = SimpleNamespace(has_errors=False)
    expected_auth = object()
    review_model = "openai/gpt-6-luna"

    def fake_review_simulation(**kwargs):
        captured["review_model"] = kwargs["review_model"]
        captured["mode"] = kwargs["mode"]
        return expected_review, expected_auth

    monkeypatch.setattr(
        reviewer,
        "review_simulation",
        fake_review_simulation,
    )

    simulation = SimpleNamespace()
    task = SimpleNamespace(id="task-1")

    batch.run_auto_review(
        simulation=simulation,
        task=task,
        review_mode="full",
        review_model=review_model,
        user_info=batch.review_user_info(
            "user_simulator", "openai/gpt-6.1-sol", {}, None, use_tools=False
        ),
        policy="policy",
    )

    assert simulation.review is expected_review
    assert simulation.auth_classification is expected_auth
    assert captured == {
        "review_model": review_model,
        "mode": reviewer.ReviewMode.FULL,
    }


def test_review_user_info_renders_the_callers_persona():
    """Reviewers see the same caller scaffold and persona block as the run."""
    from tau2.user.personas import assign_user_persona
    from tau2.user.user_simulator import build_user_system_prompt

    persona = assign_user_persona("airline", "0", seed=300)
    assert persona is not None and persona.persona_id == "priya_patil"
    info = batch.review_user_info("user_simulator", "m", {}, persona, use_tools=False)
    assert info.persona_config == persona
    assert "<PERSONA_GUIDELINES>" not in info.global_simulation_guidelines
    assert persona.gender_guidance in info.global_simulation_guidelines
    # The caller prompt is exactly these guidelines followed by the scenario.
    from tau2.user.guidelines import simulation_guidelines

    prompt = build_user_system_prompt(
        "scenario",
        persona_config=persona,
        guidelines=simulation_guidelines(use_tools=False),
    )
    assert prompt.startswith(info.global_simulation_guidelines.strip())
