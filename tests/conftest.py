import sys
from pathlib import Path
from typing import Callable

import pytest

# Make the test-only mock domain (tests/mock_domain) importable
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

from mock_domain import environment as mock_environment  # noqa: E402

from tau2.data_model.tasks import Task  # noqa: E402
from tau2.environment.environment import Environment  # noqa: E402
from tau2.registry import registry  # noqa: E402
from tau2.run import get_tasks  # noqa: E402

# Register the mock domain once for the whole test session
if "mock" not in registry.get_domains():
    registry.register_domain(mock_environment.get_environment, "mock")
    registry.register_tasks(
        mock_environment.get_tasks,
        "mock",
        get_task_splits=mock_environment.get_tasks_split,
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-live-api",
        action="store_true",
        default=False,
        help="Run live_api tests; requires provider credentials and may incur costs",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live-api"):
        return
    skip = pytest.mark.skip(reason="Live model calls require --run-live-api")
    for item in items:
        if item.get_closest_marker("live_api") is not None:
            item.add_marker(skip)


@pytest.fixture
def domain_name():
    return "mock"


@pytest.fixture
def get_environment() -> Callable[[], Environment]:
    return registry.get_env_constructor("mock")


@pytest.fixture
def base_task() -> Task:
    return get_tasks("mock", task_ids=["create_task_1"])[0]


@pytest.fixture
def task_with_env_assertions() -> Task:
    return get_tasks("mock", task_ids=["create_task_1_with_env_assertions"])[0]


@pytest.fixture
def task_with_message_history() -> Task:
    return get_tasks("mock", task_ids=["update_task_with_message_history"])[0]


@pytest.fixture
def task_with_initialization_data() -> Task:
    return get_tasks("mock", task_ids=["update_task_with_initialization_data"])[0]


@pytest.fixture
def task_with_initialization_actions() -> Task:
    return get_tasks("mock", task_ids=["update_task_with_initialization_actions"])[0]


@pytest.fixture
def task_with_history_and_env_assertions() -> Task:
    return get_tasks("mock", task_ids=["update_task_with_history_and_env_assertions"])[
        0
    ]


@pytest.fixture
def task_with_user_tools() -> Task:
    return get_tasks("mock", task_ids=["update_task_with_user_tools"])[0]


@pytest.fixture
def task_with_action_checks() -> Task:
    return get_tasks("mock", task_ids=["impossible_task_1"])[0]
