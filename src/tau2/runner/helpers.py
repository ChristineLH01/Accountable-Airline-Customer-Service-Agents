"""
Helper functions for task loading, run configuration, and metadata.
"""

from typing import Optional

from tau2.data_model.simulation import (
    AgentInfo,
    Info,
    RunConfig,
    UserInfo,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import EnvironmentInfo
from tau2.registry import RegistryInfo, registry
from tau2.utils.utils import get_commit_hash, get_now


def get_options() -> RegistryInfo:
    """Returns available options (domains, agents, users, task sets) from the registry."""
    return registry.get_info()


def get_environment_info(
    domain_name: str,
    include_tool_info: bool = False,
    env_kwargs: Optional[dict] = None,
) -> EnvironmentInfo:
    """Get information about the environment for a registered domain."""
    env_constructor = registry.get_env_constructor(domain_name)
    if env_kwargs is None:
        env_kwargs = {}
    return env_constructor(**env_kwargs).get_info(include_tool_info=include_tool_info)


def load_task_splits(task_set_name: str) -> Optional[dict[str, list[str]]]:
    """Load the task splits for a given task set."""
    task_split_loader = registry.get_task_splits_loader(task_set_name)
    if task_split_loader is None:
        return None
    return task_split_loader()


def load_tasks(
    task_set_name: str,
    task_split_name: Optional[str] = None,
    task_kwargs: Optional[dict] = None,
) -> list[Task]:
    """Load tasks for a given task set, optionally filtering by split."""
    task_loader = registry.get_tasks_loader(task_set_name)
    tasks = task_loader(task_split_name=task_split_name, **(task_kwargs or {}))
    return tasks


def get_tasks(
    task_set_name: str,
    task_split_name: Optional[str] = None,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
    task_kwargs: Optional[dict] = None,
) -> list[Task]:
    """Load registered tasks with optional filtering by IDs and count.

    Raises:
        ValueError: If task_ids are specified but some are not found.
    """
    tasks = load_tasks(
        task_set_name=task_set_name,
        task_split_name=task_split_name,
        task_kwargs=task_kwargs,
    )
    return filter_tasks(tasks, task_ids=task_ids, num_tasks=num_tasks)


def filter_tasks(
    tasks: list[Task],
    *,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> list[Task]:
    """Apply explicit task IDs (all must exist) and then a count limit."""
    if task_ids is not None:
        selected = [task for task in tasks if task.id in task_ids]
        if len(selected) != len(task_ids):
            missing = set(task_ids) - {task.id for task in selected}
            raise ValueError(f"Not all tasks were found: {sorted(missing)}")
        tasks = selected
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


def select_run_tasks(config: RunConfig) -> list[Task]:
    """A run's tasks: the configured task set and split, filtered by the config."""
    return get_tasks(
        task_set_name=config.task_set_name or config.domain,
        task_split_name=config.task_split_name,
        task_ids=config.task_ids,
        num_tasks=config.num_tasks,
    )


def make_run_name(config: RunConfig) -> str:
    """Generate a run name from the run config."""
    clean_llm_agent_name = [x for x in config.llm_agent.split("/") if x][-1]
    agent_name = f"{config.agent}_{clean_llm_agent_name}"

    clean_llm_user_name = [x for x in config.llm_user.split("/") if x][-1]
    user_name = f"{config.user}_{clean_llm_user_name}"

    return (
        f"{get_now(use_compact_format=True)}_{config.domain}_{agent_name}_{user_name}"
    )


def get_info(config: RunConfig) -> Info:
    """Create an Info object for storing run configuration metadata.

    Args:
        config: The run configuration.

    Returns:
        Info object with run metadata.
    """
    user_info = UserInfo(
        implementation=config.user,
        llm=config.llm_user,
        llm_args=config.llm_args_user,
    )
    agent_info = AgentInfo(
        implementation=config.agent,
        llm=config.llm_agent,
        llm_args=config.llm_args_agent,
    )
    environment_info = get_environment_info(config.domain, include_tool_info=False)
    return Info(
        git_commit=get_commit_hash(),
        num_trials=config.num_trials,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        user_info=user_info,
        agent_info=agent_info,
        environment_info=environment_info,
        seed=config.seed,
        communicate_scoring=config.communicate_scoring,
    )
