#!/usr/bin/env python3
"""Show a domain's agent policy and tool reference in the terminal."""

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from tau2.registry import registry


def _params_text(schema: dict) -> str:
    """One line per parameter: name (type, required): description."""
    parameters = schema.get("function", {}).get("parameters", {})
    required = set(parameters.get("required", []))
    lines = []
    for name, spec in parameters.get("properties", {}).items():
        kind = spec.get("type") or ", ".join(
            option.get("type", "object") for option in spec.get("anyOf", [])
        )
        flag = "required" if name in required else "optional"
        description = spec.get("description", "")
        lines.append(f"{name} ({kind or 'object'}, {flag}): {description}".rstrip(": "))
    return "\n".join(lines) or "-"


def main(domain: str, show_policy: bool = True):
    """
    Show documentation for a specific domain.

    Args:
        domain: Name of the domain to show documentation for
        show_policy: Whether to print the agent policy before the tools
    """
    console = Console()
    if domain not in registry.get_domains():
        console.print(
            f"[red]Domain '{domain}' not found. Available domains: "
            f"{registry.get_domains()}[/red]"
        )
        sys.exit(1)

    environment = registry.get_env_constructor(domain)()
    if show_policy:
        console.print(
            Panel(Markdown(environment.get_policy()), title=f"{domain} agent policy")
        )

    toolkits = [("Agent tools", environment.tools)]
    if environment.user_tools is not None:
        toolkits.append(("User tools", environment.user_tools))
    for title, toolkit in toolkits:
        if toolkit is None:
            continue
        table = Table(title=title, show_lines=True)
        table.add_column("Tool", style="bold green", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Description")
        table.add_column("Parameters")
        for name, tool in sorted(toolkit.get_tools().items()):
            table.add_row(
                name,
                toolkit.tool_type(name).value,
                tool.short_desc or "",
                _params_text(tool.openai_schema),
            )
        console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show domain documentation")
    parser.add_argument(
        "domain",
        type=str,
        help="Name of the domain to show documentation for (e.g., 'airline')",
    )
    args = parser.parse_args()
    main(domain=args.domain)
