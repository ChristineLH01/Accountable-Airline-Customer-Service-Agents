"""
Layer 3: Batch runner.

Orchestrates batch execution with concurrency, checkpointing, retries,
logging, and optional side effects (auto-review).

Uses Layer 2 (build) to construct instances and Layer 1 (simulation) to
execute them.
"""

import json
import multiprocessing
import os
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from tau2.data_model.persona import PersonaConfig
from tau2.data_model.simulation import (
    Info,
    Results,
    RunConfig,
    SimulationRun,
    UserInfo,
)
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType
from tau2.metrics.agent_metrics import compute_metrics
from tau2.registry import registry
from tau2.runner.build import build_orchestrator
from tau2.runner.checkpoint import create_checkpoint_fns, try_resume
from tau2.runner.helpers import get_info, make_run_name, select_run_tasks
from tau2.runner.progress import StatusMonitor, run_with_retry
from tau2.runner.simulation import run_simulation
from tau2.user.guidelines import render_persona_guidelines, simulation_guidelines
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.errors import NonRetryableError
from tau2.utils.llm_utils import llm_log_mode, set_llm_log_dir, set_llm_log_mode
from tau2.utils.utils import DATA_DIR

# Context variable to track current simulation_id for log filtering
# This ensures task-specific log handlers only receive their own messages
_current_simulation_id: ContextVar[Optional[str]] = ContextVar(
    "_current_simulation_id", default=None
)


# =============================================================================
# Side-effect helpers
# =============================================================================


def review_user_info(
    user: str,
    llm_user: Optional[str],
    llm_args_user: Optional[dict],
    persona_config: Optional[PersonaConfig],
    *,
    use_tools: bool,
) -> UserInfo:
    """The caller's guidelines and persona, as the reviewer needs them.

    Reviewers consume the rendered guidelines (with the persona block) and do
    not render the persona themselves.
    """
    guidelines = simulation_guidelines(use_tools=use_tools)
    if persona_config is not None:
        guidelines = render_persona_guidelines(guidelines, persona_config)
    return UserInfo(
        implementation=user,
        llm=llm_user,
        llm_args=llm_args_user,
        global_simulation_guidelines=guidelines,
        persona_config=persona_config,
    )


def run_auto_review(
    simulation: SimulationRun,
    task: Task,
    review_mode: str,
    review_model: str,
    user_info: UserInfo,
    policy: str,
) -> None:
    """Run LLM conversation review on a simulation and attach results.

    Args:
        simulation: The completed simulation to review.
        task: The task specification.
        review_mode: "full" (agent+user) or "user" (user only).
        review_model: LLM model to use for review and auth classification.
        user_info: The caller's guidelines and persona used in the run.
        policy: Environment policy string.
    """
    from tau2.evaluator.reviewer import ReviewMode, review_simulation

    review_mode_enum = ReviewMode.FULL if review_mode == "full" else ReviewMode.USER

    logger.info(f"Starting review for task {task.id} (mode: {review_mode})...")

    review_result, auth_result = review_simulation(
        simulation=simulation,
        task=task,
        mode=review_mode_enum,
        user_info=user_info,
        policy=policy,
        review_model=review_model,
    )

    if review_mode == "full":
        simulation.review = review_result
        simulation.auth_classification = auth_result
    else:
        simulation.user_only_review = review_result

    logger.info(
        f"Review completed for task {task.id}: has_errors={review_result.has_errors}"
    )


# =============================================================================
# Per-task logging context manager
# =============================================================================


class _TaskLogContext:
    """Manages per-task log files and LLM debug logging."""

    def __init__(
        self,
        simulation_id: str,
        save_dir: Optional[Path],
        task: Task,
        verbose_logs: bool,
    ):
        self.simulation_id = simulation_id
        self.save_dir = save_dir
        self.task = task
        self.verbose_logs = verbose_logs
        self.task_log_dir: Optional[Path] = None
        self._handler_id = None

    def __enter__(self):
        if self.save_dir:
            self.task_log_dir = (
                self.save_dir
                / "artifacts"
                / f"task_{self.task.id}"
                / f"sim_{self.simulation_id}"
            )

        if self.verbose_logs and self.task_log_dir:
            self.task_log_dir.mkdir(parents=True, exist_ok=True)
            _current_simulation_id.set(self.simulation_id)

            def make_simulation_filter(sim_id: str):
                def simulation_filter(record):
                    return _current_simulation_id.get() == sim_id

                return simulation_filter

            log_file_path = self.task_log_dir / "task.log"
            self._handler_id = logger.add(
                log_file_path,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level="DEBUG",
                rotation=None,
                enqueue=True,
                filter=make_simulation_filter(self.simulation_id),
            )
            logger.debug(f"Task log file: {log_file_path}")

        if self.task_log_dir and self.verbose_logs:
            llm_log_dir = self.task_log_dir / "llm_debug"
            set_llm_log_dir(llm_log_dir)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.task_log_dir and self.task_log_dir.exists():
            status = {
                "status": "failed",
                "reason": "infrastructure_error",
                "error": str(exc_val),
                "error_type": exc_type.__name__,
            }
            try:
                status_path = self.task_log_dir / "sim_status.json"
                with open(status_path, "w") as f:
                    json.dump(status, f, indent=2)
            except Exception:
                pass

        if self.save_dir:
            set_llm_log_dir(None)
        if self._handler_id is not None:
            logger.remove(self._handler_id)
            _current_simulation_id.set(None)
        return False


# =============================================================================
# Single task runner (Layer 3 wrapper over Layers 1+2)
# =============================================================================


def run_single_task(
    config: RunConfig,
    task: Task,
    *,
    seed: Optional[int] = None,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    save_dir: Optional[Path] = None,
    verbose_logs: bool = False,
    auto_review: bool = False,
    review_mode: str = "full",
    review_model: Optional[str] = None,
) -> SimulationRun:
    """Run a single task simulation with logging and optional auto-review.

    This is the Layer 3 per-task function. It:
    1. Sets up per-task logging.
    2. Builds an orchestrator via Layer 2 (build_orchestrator).
    3. Runs the simulation via Layer 1 (run_simulation).
    4. Optionally runs auto-review.
    5. Cleans up logging.

    Args:
        config: The run configuration.
        task: The task to run.
        seed: Random seed for this trial.
        evaluation_type: Evaluation type to use.
        save_dir: Directory for saving per-task logs.
        verbose_logs: Enable per-task log files.
        auto_review: Run LLM conversation review after simulation.
        review_mode: Review mode ("full" or "user").
        review_model: LLM model to use for review and auth classification.

    Returns:
        The completed SimulationRun with reward_info attached.
    """
    simulation_id = str(uuid.uuid4())

    logger.info(
        f"STARTING SIMULATION: Domain: {config.domain}, Task: {task.id}, "
        f"Agent: {config.agent}, User: {config.user}"
    )

    with _TaskLogContext(simulation_id, save_dir, task, verbose_logs):
        # Layer 2: Build the orchestrator
        orchestrator = build_orchestrator(
            config,
            task,
            seed=seed,
            simulation_id=simulation_id,
        )

        # Layer 1: Run the simulation
        simulation = run_simulation(
            orchestrator,
            evaluation_type=evaluation_type,
            communicate_scoring=config.communicate_scoring,
            reason_scoring=config.reason_scoring,
        )

        # Side effects
        if auto_review:
            run_auto_review(
                simulation=simulation,
                task=task,
                review_mode=review_mode,
                review_model=review_model or config.review_model,
                user_info=review_user_info(
                    config.user,
                    config.llm_user,
                    config.llm_args_user,
                    getattr(orchestrator.user, "persona_config", None),
                    use_tools=orchestrator.user.tools is not None,
                ),
                policy=orchestrator.environment.get_policy(),
            )

        logger.info(
            f"FINISHED SIMULATION: Domain: {config.domain}, Task: {task.id}, "
            f"Reward: {simulation.reward_info.reward if simulation.reward_info else 'N/A'}"
        )

        return simulation


# =============================================================================
# Batch preparation and per-unit execution
# =============================================================================


@dataclass(frozen=True)
class WorkUnit:
    """One simulation still owed by a run: a task in a given trial."""

    task_id: str
    trial: int
    seed: int
    progress_str: str = ""


@dataclass
class _BatchContext:
    """Everything run_unit() needs besides the unit itself."""

    config: RunConfig
    evaluation_type: EvaluationType
    save_dir: Optional[Path]
    info: Info
    console_display: bool = True
    save_fn: Optional[Callable] = None
    monitor: Optional[StatusMonitor] = None
    shutdown_event: Optional[threading.Event] = None
    llm_log_mode_value: Optional[str] = None


def run_unit(
    ctx: _BatchContext, task: Task, trial: int, seed: int, progress_str: str = ""
) -> SimulationRun:
    """Execute one work unit: a single task/trial with retries."""
    config = ctx.config
    save_dir = ctx.save_dir
    monitor = ctx.monitor

    if ctx.shutdown_event is not None and ctx.shutdown_event.is_set():
        raise KeyboardInterrupt("Shutdown requested")

    if ctx.llm_log_mode_value is not None:
        set_llm_log_mode(ctx.llm_log_mode_value)
    task_key = f"{task.id}.{trial}"
    if monitor:
        monitor.task_started(task_key, trial)

    console_text = Text(
        text=f"{progress_str}. Running task {task.id}, trial {trial + 1}"
        if progress_str
        else f"Running task {task.id}, trial {trial + 1}",
        style="bold green",
    )
    ConsoleDisplay.console.print(console_text)

    def _execute():
        return run_single_task(
            config,
            task,
            seed=seed,
            evaluation_type=ctx.evaluation_type,
            save_dir=save_dir,
            verbose_logs=config.verbose_logs,
            auto_review=config.auto_review,
            review_mode=config.review_mode,
            review_model=config.review_model,
        )

    try:
        result = run_with_retry(
            _execute,
            task=task,
            trial=trial,
            seed=seed,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            console_display=ctx.console_display,
            on_retry=(lambda: monitor.task_restarted(task_key)) if monitor else None,
            shutdown_event=ctx.shutdown_event,
        )

        if ctx.save_fn:
            ctx.save_fn(result)

        # Mark the final sim as the one used in results
        if save_dir is not None:
            sim_dir = save_dir / "artifacts" / f"task_{task.id}" / f"sim_{result.id}"
            if sim_dir.exists():
                try:
                    with open(sim_dir / "sim_status.json", "w") as f:
                        json.dump({"status": "used"}, f, indent=2)
                except Exception:
                    pass

        return result
    finally:
        if monitor:
            monitor.task_finished(task_key)


@dataclass
class BatchPrep:
    """A run made ready to execute: resumed results, checkpoint fns, and the
    work units still owed. Nothing here is persisted."""

    config: RunConfig
    tasks: list[Task]
    simulation_results: Results
    done_runs: set
    units: list[WorkUnit]
    save_fn: Optional[Callable]
    replace_fn: Optional[Callable]
    save_dir: Optional[Path]
    evaluation_type: EvaluationType
    console_display: bool


def prepare_batch(
    config: RunConfig,
    tasks: list[Task],
    *,
    save_path: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    console_display: bool = True,
    results_format: str = "json",
) -> BatchPrep:
    """Validate a run, resume its checkpoint, and compute the units still owed."""
    if isinstance(save_path, str):
        save_path = Path(save_path)

    if len(tasks) == 0:
        raise ValueError("No tasks to run")

    # Seed management
    random.seed(config.seed)
    seeds = [random.randint(0, 1000000) for _ in range(config.num_trials)]
    if config.llm_args_agent and "seed" in config.llm_args_agent:
        logger.warning("Each trial will modify the seed for the agent")
    if config.llm_args_user and "seed" in config.llm_args_user:
        logger.warning("Each trial will modify the seed for the user")

    lock = multiprocessing.Lock()

    # Build Info and initial Results
    simulation_results = Results(
        info=get_info(config),
        tasks=tasks,
        simulations=[],
    )

    # Checkpoint resume
    done_runs: set = set()
    if save_path is not None:
        simulation_results, done_runs, tasks = try_resume(
            save_path=save_path,
            simulation_results=simulation_results,
            tasks=tasks,
            num_trials=config.num_trials,
            auto_resume=config.auto_resume,
            results_format=results_format,
        )

    # Create checkpoint saver and replacer (shared state for dir format)
    save_fn, replace_fn = create_checkpoint_fns(save_path, lock)

    # Build unit list (skip already-completed runs)
    units: list[WorkUnit] = []
    for trial in range(config.num_trials):
        for i, task in enumerate(tasks):
            if (trial, task.id, seeds[trial]) in done_runs:
                console_text = Text(
                    text=f"Skipping task {task.id}, trial {trial + 1} because it has already been run.",
                    style="bold yellow",
                )
                ConsoleDisplay.console.print(console_text)
                continue
            progress_str = f"{i}/{len(tasks)} (trial {trial + 1}/{config.num_trials})"
            units.append(
                WorkUnit(
                    task_id=task.id,
                    trial=trial,
                    seed=seeds[trial],
                    progress_str=progress_str,
                )
            )

    return BatchPrep(
        config=config,
        tasks=tasks,
        simulation_results=simulation_results,
        done_runs=done_runs,
        units=units,
        save_fn=save_fn,
        replace_fn=replace_fn,
        save_dir=save_dir,
        evaluation_type=evaluation_type,
        console_display=console_display,
    )


# =============================================================================
# Batch runner
# =============================================================================


def run_tasks(
    config: RunConfig,
    tasks: list[Task],
    *,
    save_path: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    console_display: bool = True,
    results_format: str = "json",
) -> Results:
    """Run simulations for a list of tasks with concurrency, checkpointing, and retries.

    This is the main batch execution function. It handles:
    - Seed management and trial repetition
    - Checkpoint save/resume
    - Concurrent execution via thread pool
    - Progress monitoring
    - Retry on failure

    Args:
        config: Full run configuration (includes domain, agent, user, LLM settings,
            num_trials, max_concurrency, retry settings, etc.).
        tasks: The tasks to run.
        save_path: Path to the results JSON file. If None, results are not persisted.
        save_dir: Directory for saving per-task logs. If None, no logs are saved.
        evaluation_type: Evaluation type to use for all simulations.
        console_display: Whether to show console output for each simulation.
        results_format: Storage format of a new results file ("json" or "dir").

    Returns:
        Results object with all simulation runs.

    Raises:
        ValueError: If no tasks are provided.
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)

    # Set log level from config
    logger.remove()
    logger.add(lambda msg: print(msg), level=config.log_level)

    prep = prepare_batch(
        config,
        tasks,
        save_path=save_path,
        save_dir=save_dir,
        evaluation_type=evaluation_type,
        console_display=console_display,
        results_format=results_format,
    )
    tasks = prep.tasks
    simulation_results = prep.simulation_results

    # Status monitor
    total_count = len(tasks) * config.num_trials
    monitor = StatusMonitor(total_count, initial_completed=len(prep.done_runs))
    monitor.set_results(simulation_results)
    monitor.start()

    shutdown_event = threading.Event()

    ctx = _BatchContext(
        config=config,
        evaluation_type=evaluation_type,
        save_dir=save_dir,
        info=simulation_results.info,
        console_display=console_display,
        save_fn=prep.save_fn,
        monitor=monitor,
        shutdown_event=shutdown_event,
        # Capture ContextVar values from the main thread so worker threads
        # (which get a fresh default context) can re-apply them.
        llm_log_mode_value=llm_log_mode.get(),
    )

    tasks_by_id = {task.id: task for task in tasks}
    results_lock = threading.Lock()
    fatal_stop = threading.Event()

    def _run(unit: WorkUnit) -> None:
        if fatal_stop.is_set():
            return
        try:
            result = run_unit(
                ctx,
                tasks_by_id[unit.task_id],
                unit.trial,
                unit.seed,
                unit.progress_str,
            )
        except NonRetryableError:
            fatal_stop.set()
            raise
        with results_lock:
            simulation_results.simulations.append(result)

    executor = ThreadPoolExecutor(max_workers=config.max_concurrency)
    try:
        futures = [executor.submit(_run, unit) for unit in prep.units]
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        ConsoleDisplay.console.print(
            "\n[bold red]Ctrl+C received — cancelling remaining tasks...[/bold red]"
        )
        shutdown_event.set()
        executor.shutdown(wait=False, cancel_futures=True)

        n = len(simulation_results.simulations)
        ConsoleDisplay.console.print(
            f"[bold yellow]{n} simulation(s) already checkpointed. "
            f"Use --auto-resume to continue later.[/bold yellow]"
        )
        monitor.stop()

        # Force-exit: background threads (e.g. litellm) hold the process alive
        # and produce noisy errors during interpreter shutdown. All completed
        # results are already on disk via save_fn.
        os._exit(130)
    except BaseException:
        fatal_stop.set()
        raise
    finally:
        monitor.stop()
        if not shutdown_event.is_set():
            executor.shutdown(
                wait=not fatal_stop.is_set(), cancel_futures=fatal_stop.is_set()
            )

    ConsoleDisplay.console.print(
        "\n[bold green]Successfully completed all simulations![/bold green]\n"
        "To review the simulations, run: [bold blue]tau2 view[/bold blue]"
    )
    return simulation_results


# =============================================================================
# Top-level entry point
# =============================================================================


def _load_run_tasks(config: RunConfig) -> list[Task]:
    """Load a config's tasks and apply the agent's registered task filter."""
    tasks = select_run_tasks(config)

    task_filter = registry.get_agent_task_filter(config.agent)
    if task_filter is not None:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if task_filter(task)]
        num_tasks = len(tasks)
        console_text = Text(
            text=f"Running {num_tasks} out of {total_num_tasks} tasks for {config.agent} (filtered).",
            style="bold green",
        )
        ConsoleDisplay.console.print(console_text)
    return tasks


def run_save_paths(config: RunConfig) -> tuple[str, Path, Path]:
    """(run_name, save_dir, save_path) for a config.

    ``save_to`` names a directory under ``data/simulations/``; an absolute
    path is used as is.
    """
    run_name = config.save_to or make_run_name(config)
    save_dir = DATA_DIR / "simulations" / run_name
    save_path = save_dir / "results.json"
    return run_name, save_dir, save_path


def run_domain(config: RunConfig) -> Results:
    """Run simulations for a domain from a RunConfig.

    This is the main entry point for the CLI. It:
    1. Loads and filters tasks.
    2. Determines save paths.
    3. Delegates to run_tasks() for batch execution.
    4. Computes and displays metrics.

    Args:
        config: Full run configuration.

    Returns:
        Results object with all simulation runs.
    """
    ConsoleDisplay.display_run_config(config)

    tasks = _load_run_tasks(config)
    _, save_dir, save_path = run_save_paths(config)

    simulation_results = run_tasks(
        config,
        tasks,
        save_path=save_path,
        save_dir=save_dir,
    )

    metrics = compute_metrics(simulation_results)
    ConsoleDisplay.display_agent_metrics(metrics)

    return simulation_results
