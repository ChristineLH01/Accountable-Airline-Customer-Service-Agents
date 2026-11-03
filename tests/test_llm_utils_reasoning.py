"""Offline checks of the completion boundary: every model goes through OpenRouter."""

from types import SimpleNamespace

import pytest

import tau2.utils.llm_utils as llm_utils
from tau2.data_model.message import Message, SystemMessage, UserMessage
from tau2.environment.tool import as_tool


@pytest.fixture
def messages() -> list[Message]:
    return [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="Hello."),
        SystemMessage(role="system", content="The user went quiet."),
    ]


def _fake_response():
    message = SimpleNamespace(role="assistant", content="hi", tool_calls=None)
    choice = SimpleNamespace(finish_reason="stop", message=message)

    class _Resp(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

        def to_dict(self):
            return {}

    return _Resp(model="test-model", choices=[choice], usage=None)


@pytest.fixture
def capture_completion(monkeypatch):
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda response: 0.0)
    monkeypatch.setattr(llm_utils, "_write_llm_log", lambda *args, **kwargs: None)
    return captured


@pytest.mark.parametrize(
    "model,route",
    [
        ("openai/gpt-6.1-sol", "openrouter/openai/gpt-6.1-sol"),
        ("anthropic/claude-opus-4.5", "openrouter/anthropic/claude-opus-4.5"),
        ("openrouter/openai/gpt-6-luna", "openrouter/openai/gpt-6-luna"),
    ],
)
def test_models_are_routed_through_openrouter(
    capture_completion, messages, model, route
):
    llm_utils.generate(model, messages)
    assert capture_completion["model"] == route


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_reasoning_effort_and_seed_pass_through(capture_completion, messages, effort):
    llm_utils.generate("openai/gpt-6.1-sol", messages, reasoning_effort=effort, seed=7)
    assert capture_completion["reasoning_effort"] == effort
    assert capture_completion["seed"] == 7
    assert "temperature" not in capture_completion


def test_messages_and_tools_are_sent_unchanged(capture_completion, messages):
    def square(x: int) -> int:
        """Square a number.

        Args:
            x: The number.
        """
        return x * x

    tool = as_tool(square)
    llm_utils.generate("openai/gpt-6.1-sol", messages, tools=[tool])
    assert [m["content"] for m in capture_completion["messages"]] == [
        "You are a helpful assistant.",
        "Hello.",
        "The user went quiet.",
    ]
    assert capture_completion["tools"] == [tool.openai_schema]
    assert capture_completion["tool_choice"] == "auto"


def test_openrouter_request_body(monkeypatch, messages):
    """The real LiteLLM client posts to OpenRouter with one key and our params."""
    import json

    import httpx

    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["url"] = str(request.url)
        sent["auth"] = request.headers.get("authorization")
        sent["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-6.1-sol",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "cost": 0.000123,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    original_send = httpx.Client.send

    def send(self, request, **kwargs):
        return transport.handle_request(request)

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(llm_utils, "_write_llm_log", lambda *args, **kwargs: None)
    try:
        reply = llm_utils.generate(
            "openai/gpt-6.1-sol",
            messages,
            reasoning_effort="medium",
            seed=3,
            num_retries=0,
        )
    finally:
        monkeypatch.setattr(httpx.Client, "send", original_send)

    assert sent["url"].startswith("https://openrouter.ai/api/v1/chat/completions")
    assert sent["auth"] == "Bearer sk-or-test"
    assert sent["body"]["model"] == "openai/gpt-6.1-sol"
    assert sent["body"]["reasoning_effort"] == "medium"
    assert sent["body"]["seed"] == 3
    assert reply.content == "hi"
    assert reply.cost == pytest.approx(0.000123)
