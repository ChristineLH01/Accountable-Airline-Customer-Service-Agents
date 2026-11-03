import argparse
import json

from tau2.config import (
    DEFAULT_AGENT_IMPLEMENTATION,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_LOG_MODE,
    DEFAULT_LLM_REVIEW,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_MIN_WAIT,
    DEFAULT_SEED,
    DEFAULT_USER_IMPLEMENTATION,
    TEXT_BENCHMARK_LLM_ARGS_USER,
    TEXT_BENCHMARK_LLM_USER,
)
from tau2.data_model.semantic_communication import (
    CommunicateScorer,
    CommunicateScoring,
    SemanticCommunicateSettings,
)
from tau2.data_model.simulation import RunConfig
from tau2.run import get_options, run_domain


def add_run_args(parser):
    """Add run arguments to a parser."""
    options = get_options()
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=options.domains,
        default="airline",
        help="The domain to run the simulation on. Default is airline.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="The number of times each task is run. Default is 1.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_AGENT_IMPLEMENTATION,
        choices=options.agents,
        help=f"The agent implementation to use. Default is {DEFAULT_AGENT_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=DEFAULT_LLM_AGENT,
        help=f"The OpenRouter model ID of the agent's LLM. Default is {DEFAULT_LLM_AGENT}.",
    )
    parser.add_argument(
        "--agent-llm-args",
        type=json.loads,
        default=None,
        help=f"The arguments to pass to the LLM for the agent, as JSON. Default is '{json.dumps(DEFAULT_LLM_ARGS_AGENT)}'.",
    )
    parser.add_argument(
        "--user",
        type=str,
        choices=options.users,
        default=DEFAULT_USER_IMPLEMENTATION,
        help=f"The user implementation to use. Default is {DEFAULT_USER_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default=None,
        help=f"The OpenRouter model ID of the user simulator's LLM. Default is {DEFAULT_LLM_USER}. "
        f"The released tau-bench 2.0 text benchmark uses {TEXT_BENCHMARK_LLM_USER}.",
    )
    parser.add_argument(
        "--user-llm-args",
        type=json.loads,
        default=None,
        help=f"The arguments to pass to the LLM for the user, as JSON. Default is '{json.dumps(DEFAULT_LLM_ARGS_USER)}'. "
        f"The released text benchmark uses '{json.dumps(TEXT_BENCHMARK_LLM_ARGS_USER)}'.",
    )
    parser.add_argument(
        "--task-set-name",
        type=str,
        default=None,
        choices=options.task_sets,
        help="The task set to run the simulation on. If not provided, will load default task set for the domain.",
    )
    parser.add_argument(
        "--task-split-name",
        type=str,
        default="base",
        help="The task split to run the simulation on (base, train or test). Default is 'base'.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="Run only these task IDs.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Run only the first N selected tasks.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"The maximum number of steps to run the simulation. Default is {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"The maximum number of tool errors allowed in a simulation before it is stopped. Default is {DEFAULT_MAX_ERRORS}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Maximum wallclock time in seconds for each simulation. No timeout by default.",
    )
    parser.add_argument(
        "--save-to",
        "--output",
        type=str,
        required=False,
        help="The path to save the simulation results. Will be saved to data/simulations/<save_to>/results.json. If not provided, will save to <timestamp>_<domain>_<agent>_<agent-llm>_<user>_<user-llm>. If the file already exists, it will try to resume the run.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"The maximum number of concurrent simulations to run. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"The seed to use for the simulation. Default is {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help=f"The log level to use for the simulation. Default is {DEFAULT_LOG_LEVEL}.",
    )
    parser.add_argument(
        "--verbose-logs",
        action="store_true",
        default=False,
        help="Enable verbose logging: saves LLM call logs and per-task logs. "
        "Files are saved to the save directory (auto-generated if --save-to not specified).",
    )
    parser.add_argument(
        "--llm-log-mode",
        type=str,
        choices=["all", "latest"],
        default=DEFAULT_LLM_LOG_MODE,
        help="LLM debug logging mode. Only takes effect when --verbose-logs is enabled. "
        "'all' saves every LLM call (can generate many files), "
        "'latest' keeps only the most recent call of each type (saves space). "
        f"Default is '{DEFAULT_LLM_LOG_MODE}'.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help=f"Maximum number of retries for failed tasks. Default is {DEFAULT_RETRY_ATTEMPTS}.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_MIN_WAIT,
        help=f"Delay in seconds between retries. Default is {DEFAULT_RETRY_MIN_WAIT}.",
    )
    parser.add_argument(
        "--enforce-communication-protocol",
        action="store_true",
        default=False,
        help="Enforce communication protocol rules (e.g., no mixed messages with text and tool calls). Default is False.",
    )

    # Scoring
    parser.add_argument(
        "--communicate-scorer",
        choices=[scorer.value for scorer in CommunicateScorer],
        default=None,
        help="Communicate-info scorer: the pinned semantic LLM judge (default) or "
        "the original substring matching of tau2-bench 1.x.",
    )
    parser.add_argument(
        "--communicate-judge",
        type=json.loads,
        default=None,
        metavar="JSON",
        help='Semantic judge overrides, e.g. {"model": "...", "model_args": {...}}.',
    )

    # Resume mode
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=False,
        help="Automatically resume from existing save file without prompting (for non-interactive runs).",
    )

    # Auto-review mode
    parser.add_argument(
        "--auto-review",
        action="store_true",
        default=False,
        help="Automatically run LLM conversation review after each simulation.",
    )
    parser.add_argument(
        "--review-mode",
        type=str,
        choices=["full", "user"],
        default="full",
        help="Review mode when --auto-review is enabled: 'full' (agent+user errors, default) or 'user' (user simulator only).",
    )
    parser.add_argument(
        "--review-model",
        type=str,
        default=DEFAULT_LLM_REVIEW,
        help=f"OpenRouter model ID used for review calls. Default is {DEFAULT_LLM_REVIEW}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the resolved run configuration as JSON without running anything.",
    )


def communicate_scoring_from_args(args: argparse.Namespace):
    """The explicit communicate-info scorer a run command requests, if any."""
    if args.communicate_scorer is None and args.communicate_judge is None:
        return None
    scorer = CommunicateScorer(args.communicate_scorer or CommunicateScorer.SEMANTIC)
    if scorer == CommunicateScorer.SUBSTRING:
        if args.communicate_judge is not None:
            raise ValueError("--communicate-judge applies to the semantic scorer")
        return CommunicateScoring(scorer=scorer)
    return CommunicateScoring(
        judge=SemanticCommunicateSettings.model_validate(args.communicate_judge or {})
    )


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build the run configuration a `tau2 run` command line requests."""
    kwargs = dict(
        domain=args.domain,
        task_set_name=args.task_set_name,
        task_split_name=args.task_split_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        agent=args.agent,
        llm_agent=args.agent_llm,
        user=args.user,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        timeout=args.timeout,
        enforce_communication_protocol=args.enforce_communication_protocol,
        save_to=args.save_to,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
        verbose_logs=args.verbose_logs,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        auto_resume=args.auto_resume,
        auto_review=args.auto_review,
        review_mode=args.review_mode,
        review_model=args.review_model,
    )
    if args.agent_llm_args is not None:
        kwargs["llm_args_agent"] = args.agent_llm_args
    if args.user_llm is not None:
        kwargs["llm_user"] = args.user_llm
    if args.user_llm_args is not None:
        kwargs["llm_args_user"] = args.user_llm_args
    communicate_scoring = communicate_scoring_from_args(args)
    if communicate_scoring is not None:
        kwargs["communicate_scoring"] = communicate_scoring
    return RunConfig(**kwargs)


def _get_version() -> str:
    from tau2.utils.utils import get_tau2_version

    return get_tau2_version()


def run_intro():
    """Display an intro page describing what this framework can do."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    console = Console()
    version = _get_version()

    # ── Banner ──────────────────────────────────────────────────────────
    banner = Text(justify="center")
    banner.append("\n")
    banner.append("tau-bench airline (text, English)", style="bold cyan")
    banner.append(f"  v{version}\n", style="dim")
    banner.append(
        "A simulation framework for evaluating\n"
        "conversational airline customer-service agents\n",
        style="italic",
    )
    console.print(Panel(banner, border_style="cyan", padding=(1, 4)))

    # ── Commands ────────────────────────────────────────────────────────
    cmd_table = Table(
        title="Commands",
        box=box.SIMPLE_HEAVY,
        title_style="bold magenta",
        show_edge=False,
        pad_edge=False,
        padding=(0, 2),
    )
    cmd_table.add_column("Command", style="bold green", no_wrap=True)
    cmd_table.add_column("What it does")
    cmd_table.add_row("tau2 run", "Run agents on the airline tasks")
    cmd_table.add_row("tau2 view", "Browse and inspect simulation results")
    cmd_table.add_row("tau2 domain airline", "Show the airline domain documentation")
    cmd_table.add_row(
        "tau2 review <path>", "Run LLM-based conversation review on results"
    )
    cmd_table.add_row(
        "tau2 evaluate-trajs <paths>", "Re-evaluate trajectories and recompute rewards"
    )
    cmd_table.add_row(
        "tau2 convert-results <path>", "Convert results between json and dir formats"
    )
    cmd_table.add_row("tau2 check-data", "Verify data directory is set up correctly")
    cmd_table.add_row("tau2 intro", "Show this page")
    console.print(cmd_table)
    console.print()

    # ── Quick Start ─────────────────────────────────────────────────────
    quick_start = (
        "# 1. Verify your setup\n"
        "tau2 check-data\n"
        "\n"
        "# 2. Run an evaluation on 5 airline tasks\n"
        "tau2 run --agent-llm openai/gpt-6.1-sol --user-llm openai/gpt-6.1-sol "
        "--num-trials 1 --num-tasks 5\n"
        "\n"
        "# 3. Browse results\n"
        "tau2 view"
    )
    console.print(
        Panel(
            Syntax(quick_start, "bash", theme="monokai", line_numbers=False),
            title="[bold yellow]Quick Start[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    # ── Tip ─────────────────────────────────────────────────────────────
    console.print(
        "\n[dim]Tip: Run any command with [bold]--help[/bold] for detailed usage, "
        "e.g. [bold green]tau2 run --help[/bold green][/dim]\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Tau2 command line interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run agents on the airline tasks")
    add_run_args(run_parser)

    def run_command(args):
        # Set global LLM log mode (used by verbose logging)
        from tau2.utils.llm_utils import set_llm_log_mode

        set_llm_log_mode(args.llm_log_mode)
        config = config_from_args(args)
        if args.dry_run:
            print(config.model_dump_json(indent=2))
            return config
        return run_domain(config)

    run_parser.set_defaults(func=run_command)

    # View command
    view_parser = subparsers.add_parser("view", help="View simulation results")
    view_parser.add_argument(
        "--dir",
        type=str,
        help="Directory containing simulation run directories. Defaults to data/simulations.",
    )
    view_parser.add_argument(
        "--file",
        type=str,
        help="Path to a simulation results file or run directory",
    )
    view_parser.add_argument(
        "--only-show-failed",
        action="store_true",
        help="Only show failed tasks.",
    )
    view_parser.add_argument(
        "--only-show-all-failed",
        action="store_true",
        help="Only show tasks that failed in all trials.",
    )
    view_parser.add_argument(
        "--max-tool-result-chars",
        type=int,
        default=500,
        help="Truncate tool results to this many characters. Default: 500.",
    )
    view_parser.add_argument(
        "--full-tool-results",
        action="store_true",
        help="Show full tool results without truncation.",
    )
    view_parser.set_defaults(func=lambda args: run_view_simulations(args))

    # Domain command
    domain_parser = subparsers.add_parser("domain", help="Show domain documentation")
    domain_parser.add_argument(
        "domain",
        type=str,
        help="Name of the domain to show documentation for (e.g., 'airline')",
    )
    domain_parser.set_defaults(func=lambda args: run_show_domain(args))

    # Intro command
    intro_parser = subparsers.add_parser(
        "intro", help="Show an overview of the framework and available commands"
    )
    intro_parser.set_defaults(func=lambda args: run_intro())

    # Check data command
    check_data_parser = subparsers.add_parser(
        "check-data", help="Check if data directory is properly configured"
    )
    check_data_parser.set_defaults(func=lambda args: run_check_data())

    # Evaluate trajectories command
    evaluate_parser = subparsers.add_parser(
        "evaluate-trajs", help="Evaluate trajectories and update rewards"
    )
    evaluate_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    evaluate_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save updated trajectory files with recomputed rewards. If not provided, only displays metrics.",
    )
    evaluate_parser.add_argument(
        "--fresh-tasks",
        action="store_true",
        help="Re-grade against the current task definitions from the data directory instead of the ones embedded in each results file.",
    )
    evaluate_parser.set_defaults(func=lambda args: run_evaluate_trajectories(args))

    # Review command - LLM-based conversation review
    review_parser = subparsers.add_parser(
        "review", help="Run LLM-based conversation review on simulation results"
    )
    review_parser.add_argument(
        "path",
        help="Path to a results.json file or a directory containing results.json files",
    )
    review_parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["full", "user"],
        default="full",
        help="Review mode: 'full' (agent+user, default) or 'user' (user simulator only)",
    )
    review_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for the reviewed results (only used for single file)",
    )
    review_parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed review results for each simulation",
    )
    review_parser.add_argument(
        "-c",
        "--max-concurrency",
        type=int,
        default=32,
        help="Maximum number of concurrent reviews (default: 32)",
    )
    review_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit review to first N simulations",
    )
    review_parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        default=None,
        help="Only review simulations for these task IDs",
    )
    review_parser.add_argument(
        "--log-llm",
        action="store_true",
        help="Log LLM request/response for each review call",
    )
    review_parser.add_argument(
        "--review-model",
        type=str,
        default=DEFAULT_LLM_REVIEW,
        help=f"OpenRouter model ID used for review calls. Default is {DEFAULT_LLM_REVIEW}.",
    )
    review_parser.set_defaults(func=lambda args: run_review(args))

    # Convert results format command
    convert_parser = subparsers.add_parser(
        "convert-results",
        help="Convert simulation results between storage formats",
    )
    convert_parser.add_argument(
        "path",
        help="Path to results.json or results directory to convert",
    )
    convert_parser.add_argument(
        "--to",
        dest="target_format",
        choices=["json", "dir"],
        default=None,
        help="Target format: 'json' (monolithic) or 'dir' (directory with individual sim files). "
        "If omitted, converts to the opposite of the current format.",
    )
    convert_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a .bak backup of the original file",
    )
    convert_parser.set_defaults(func=lambda args: run_convert_results(args))

    args = parser.parse_args()
    if not hasattr(args, "func"):
        run_intro()
        return

    args.func(args)


def run_view_simulations(args):
    from tau2.scripts.view_simulations import main as view_main

    if args.full_tool_results or args.max_tool_result_chars <= 0:
        max_tool_result_length = None
    else:
        max_tool_result_length = args.max_tool_result_chars

    view_main(
        sim_file=args.file,
        only_show_failed=args.only_show_failed,
        only_show_all_failed=args.only_show_all_failed,
        sim_dir=args.dir,
        max_tool_result_length=max_tool_result_length,
    )


def run_show_domain(args):
    from tau2.scripts.show_domain_doc import main as domain_main

    domain_main(args.domain)


def run_check_data():
    from tau2.scripts.check_data import main as check_data_main

    check_data_main()


def run_evaluate_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.evaluate_trajectories import evaluate_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    evaluate_trajectories(
        args.paths, args.output_dir, fresh_tasks=getattr(args, "fresh_tasks", False)
    )


def run_review(args):
    """Run LLM-based conversation review."""
    import sys
    from pathlib import Path

    from loguru import logger
    from rich.console import Console

    from tau2.scripts.review_conversation import ReviewMode, find_results_files, review

    logger.configure(handlers=[{"sink": sys.stderr, "level": "WARNING"}])

    # Find all results files
    input_path = Path(args.path)
    results_files = find_results_files(input_path)

    if not results_files:
        console = Console()
        console.print(f"[red]No results.json files found in: {args.path}[/red]")
        sys.exit(1)

    # Run review for each results file
    mode = ReviewMode.FULL if args.mode == "full" else ReviewMode.USER
    console = Console()

    if len(results_files) > 1:
        console.print(
            f"\n📁 Found {len(results_files)} results files to review:",
            style="bold blue",
        )
        for i, rf in enumerate(results_files, 1):
            console.print(f"  {i}. {rf.parent.name}/results.json")
        console.print()

    for i, results_file in enumerate(results_files):
        if len(results_files) > 1:
            console.print(
                f"\n{'=' * 60}\n[bold cyan]Processing ({i + 1}/{len(results_files)}): {results_file.parent.name}[/bold cyan]\n{'=' * 60}"
            )

        review(
            results_path=str(results_file),
            mode=mode,
            output_path=args.output if len(results_files) == 1 else None,
            show_details=args.show_details,
            max_concurrency=args.max_concurrency,
            limit=args.limit,
            task_ids=args.task_ids,
            log_llm=args.log_llm,
            review_model=args.review_model,
        )


def run_convert_results(args):
    """Convert simulation results between storage formats."""
    import shutil
    from pathlib import Path

    from tau2.data_model.simulation import Results

    path = Path(args.path)
    current_fmt = Results._detect_format(path)
    target_fmt = args.target_format

    if target_fmt is None:
        target_fmt = "json" if current_fmt == "dir" else "dir"

    if current_fmt == target_fmt:
        print(f"Results at {path} are already in '{target_fmt}' format.")
        return

    print(f"Converting {path}: '{current_fmt}' -> '{target_fmt}'")
    results = Results.load(path)

    meta_path = path if path.suffix == ".json" else path / "results.json"

    if not args.no_backup:
        if current_fmt == "json":
            backup = meta_path.with_suffix(".json.bak")
            shutil.copy2(meta_path, backup)
            print(f"  Backup: {backup}")
        else:
            sims_dir = meta_path.parent / "simulations"
            backup_dir = meta_path.parent / "simulations.bak"
            if sims_dir.exists():
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(sims_dir, backup_dir)
            backup = meta_path.with_suffix(".json.bak")
            shutil.copy2(meta_path, backup)
            print(f"  Backup: {backup}")
            if backup_dir.exists():
                print(f"  Backup: {backup_dir}")

    if target_fmt == "dir":
        results.save(meta_path, format="dir")
    else:
        sims_dir = meta_path.parent / "simulations"
        results.save(meta_path, format="json")
        if sims_dir.exists():
            shutil.rmtree(sims_dir)

    n = len(results.simulations)
    print(f"  Done. {n} simulation(s) converted to '{target_fmt}' format.")


if __name__ == "__main__":
    main()
