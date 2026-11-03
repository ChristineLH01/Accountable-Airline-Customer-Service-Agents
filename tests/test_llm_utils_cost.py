import pytest
from litellm import ModelResponse

from tau2.utils.llm_utils import get_response_cost


def _response() -> ModelResponse:
    return ModelResponse(
        model="openai/gpt-6.1-sol",
        usage={"prompt_tokens": 1000, "completion_tokens": 100},
    )


def test_openrouter_reported_cost_is_used(monkeypatch):
    monkeypatch.setattr(
        "tau2.utils.llm_utils.completion_cost",
        lambda **kwargs: pytest.fail("price table used despite a reported cost"),
    )
    response = _response()
    response._hidden_params = {
        "additional_headers": {"llm_provider-x-litellm-response-cost": 0.0042}
    }
    assert get_response_cost(response) == pytest.approx(0.0042)


def test_hidden_response_cost_is_used():
    response = _response()
    response._hidden_params = {"response_cost": 0.5}
    assert get_response_cost(response) == 0.5


def test_falls_back_to_litellm_pricing(monkeypatch):
    monkeypatch.setattr("tau2.utils.llm_utils.completion_cost", lambda **kwargs: 0.25)
    response = _response()
    response._hidden_params = {}
    assert get_response_cost(response) == 0.25


def test_unknown_price_is_zero(monkeypatch):
    def fail(**kwargs):
        raise ValueError("model not mapped")

    monkeypatch.setattr("tau2.utils.llm_utils.completion_cost", fail)
    response = _response()
    response._hidden_params = {}
    assert get_response_cost(response) == 0.0
