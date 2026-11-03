"""Clean CLI output: --help prints only usage, and --dry-run prints the config."""

import json
import subprocess
import sys

import pytest


def _run_cli(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "from tau2.cli import main; main()", *command],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )


@pytest.mark.parametrize(
    "command",
    [["--help"], ["run", "--help"], ["view", "--help"], ["review", "--help"]],
    ids=" ".join,
)
def test_help_is_only_help(command):
    """Importing the CLI must not log the registry or data directory."""
    completed = _run_cli(*command)
    assert completed.stdout.startswith("usage: ")
    assert completed.stderr == ""


def test_dry_run_prints_resolved_airline_config():
    completed = _run_cli(
        "run",
        "--dry-run",
        "--num-tasks",
        "3",
        "--user-llm",
        "openai/gpt-5.5",
        "--user-llm-args",
        '{"reasoning_effort": "low"}',
        "--communicate-scorer",
        "substring",
    )
    config = json.loads(completed.stdout)
    assert config["domain"] == "airline"
    assert config["agent"] == "llm_agent"
    assert config["num_tasks"] == 3
    assert config["llm_user"] == "openai/gpt-5.5"
    assert config["llm_args_user"] == {"reasoning_effort": "low"}
    assert config["communicate_scoring"]["scorer"] == "substring"


def test_run_rejects_unknown_domain():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tau2.cli import main; main()",
            "run",
            "--domain",
            "retail",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode != 0
    assert "invalid choice: 'retail'" in completed.stderr
