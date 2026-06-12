"""
main.py

CLI entrypoint for the Gift Agent.

Usage:
  Single contact:
    python main.py --input data/sample_input.json

  Bulk contacts (list of contacts):
    python main.py --input data/bulk_contacts.json --bulk

  With auto-approve (skip interactive review):
    python main.py --input data/sample_input.json --auto-approve

  With specific output directory:
    python main.py --input data/sample_input.json --output-dir my_results/

The CLI runs the full workflow interactively, pausing at human_review
to let the user review recommendations and choose an action.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Load .env before anything else
load_dotenv()

app = typer.Typer(
    name="gift-agent",
    help="AI-powered hyper-personalised gift recommendation system",
    add_completion=False,
)

console = Console()


@app.command()
def main(
    input: Path = typer.Option(..., "--input", "-i", help="Input JSON file (contact or contacts list)"),
    bulk: bool = typer.Option(False, "--bulk", "-b", help="Process multiple contacts from a list"),
    auto_approve: bool = typer.Option(False, "--auto-approve", "-y", help="Auto-approve recommendations (skip interactive review)"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to write output JSON files"),
):
    """
    Run the gift recommendation workflow for one or more contacts.
    """
    if not input.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input}")
        raise typer.Exit(1)

    with open(input) as f:
        data = json.load(f)

    contacts = data if bulk else [data]

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            f"[bold]Gift Agent[/bold]\n"
            f"Processing {len(contacts)} contact(s)\n"
            f"Auto-approve: {auto_approve}",
            title="Starting",
            border_style="blue",
        )
    )

    # Check LLM provider availability
    from services.llm.factory import get_provider_info
    provider_info = get_provider_info()
    _print_provider_status(provider_info)

    for i, contact in enumerate(contacts, 1):
        console.print(f"\n[bold cyan]── Contact {i}/{len(contacts)}: {contact.get('name', 'Unknown')} ──[/bold cyan]")
        result = _run_single(
            contact=contact,
            auto_approve=auto_approve,
        )

        if result and output_dir:
            thread_id = result.get("thread_id", f"contact_{i}")
            out_path = output_dir / f"{thread_id}_output.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            console.print(f"[green]Output saved:[/green] {out_path}")

    console.print("\n[bold green]All contacts processed.[/bold green]")


def _run_single(contact: dict, auto_approve: bool) -> Optional[dict]:
    """Run the workflow for a single contact."""
    from agent.graph import get_graph

    graph = get_graph()
    import uuid
    thread_id = f"run_{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {"contact": contact}

    console.print("[dim]Running workflow...[/dim]")

    try:
        result = graph.invoke(initial_state, config=config)
    except Exception as e:
        console.print(f"[red]Workflow error:[/red] {e}")
        return None

    # Display results
    _display_recommendations(result)

    if auto_approve:
        console.print("\n[yellow]Auto-approving recommendations...[/yellow]")
        graph.update_state(config, {"review_action": "approve", "review_notes": "Auto-approved via CLI"})
        result = graph.invoke(None, config=config)
        console.print("[green]Approved and saved.[/green]")
        return result.get("final_recommendations", {})

    # Interactive review
    action = _interactive_review()

    state_update: dict[str, Any] = {"review_action": action}

    if action == "reject":
        reason = typer.prompt("Rejection reason")
        state_update["review_notes"] = reason

    elif action == "regenerate":
        notes = typer.prompt("Notes for regeneration (what to change?)")
        state_update["review_notes"] = notes

    elif action == "edit":
        gift_index = int(typer.prompt("Gift index to edit (0=first, 1=second, 2=third)", default="0"))
        field = typer.prompt("Field to edit", default="personalised_message")
        new_value = typer.prompt("New value")
        state_update["edit_payload"] = {
            "gift_index": gift_index,
            "field": field,
            "new_value": new_value,
        }

    graph.update_state(config, state_update)
    result = graph.invoke(None, config=config)

    if action in ("reject", "regenerate"):
        # Show updated results and ask again
        console.print("\n[bold]Updated recommendations:[/bold]")
        return _run_single_from_result(result, config, graph)

    console.print(f"\n[green]Action '{action}' applied successfully.[/green]")
    return result.get("final_recommendations", {})


def _run_single_from_result(result: dict, config: dict, graph) -> Optional[dict]:
    """Handle subsequent review cycles after reject/regenerate."""
    _display_recommendations(result)
    action = _interactive_review()

    state_update: dict[str, Any] = {"review_action": action}
    if action == "reject":
        reason = typer.prompt("Rejection reason")
        state_update["review_notes"] = reason
    elif action == "regenerate":
        notes = typer.prompt("Regeneration notes")
        state_update["review_notes"] = notes

    graph.update_state(config, state_update)
    result = graph.invoke(None, config=config)
    return result.get("final_recommendations", {})


def _display_recommendations(result: dict):
    """Display recommendations in the terminal."""
    ranked_gifts = result.get("ranked_gifts", [])
    contact = result.get("contact", {})
    escalation_flag = result.get("escalation_flag", False)

    console.print(
        Panel(
            f"[bold]Contact:[/bold] {contact.get('name', '')} | "
            f"{contact.get('role', '')} at {contact.get('company', '')}",
            border_style="cyan",
        )
    )

    # Show signals
    safe_signals = result.get("safe_signals", {})
    strong = safe_signals.get("strong", [])
    weak = safe_signals.get("weak", [])
    if strong or weak:
        console.print(f"\n[bold]Signals used:[/bold]")
        for s in strong:
            console.print(f"  [green]●[/green] {s} [dim](strong)[/dim]")
        for s in weak:
            console.print(f"  [yellow]●[/yellow] {s} [dim](weak)[/dim]")

    if escalation_flag:
        notes = result.get("escalation_notes", "")
        console.print(
            Panel(
                f"[bold yellow]⚠ ESCALATION[/bold yellow]\n{notes}",
                border_style="yellow",
            )
        )

    # Show ranked gifts
    if not ranked_gifts:
        console.print("[red]No recommendations generated.[/red]")
        return

    console.print(f"\n[bold]Top {len(ranked_gifts)} Gift Recommendations:[/bold]\n")

    for gift in ranked_gifts:
        rank = gift.get("rank", "?")
        name = gift.get("gift_name", "Unknown")
        url = gift.get("product_url", "")
        store = gift.get("store", "")
        price = gift.get("estimated_price", "N/A")
        confidence = gift.get("confidence", 0.0)
        risk = gift.get("risk_level", "unknown")
        why = gift.get("why_this_gift", "")
        message = gift.get("personalised_message", "")

        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(risk, "white")

        console.print(
            Panel(
                f"[bold]#{rank}: {name}[/bold]\n"
                f"[dim]{store}[/dim] | {price}\n"
                f"Confidence: [{risk_color}]{confidence:.2f}[/{risk_color}] ({risk} risk)\n\n"
                f"[italic]{why}[/italic]\n\n"
                f"Message: [cyan]\"{message}\"[/cyan]\n\n"
                f"[dim]URL: {url}[/dim]",
                border_style=risk_color,
            )
        )


def _interactive_review() -> str:
    """Prompt user for a review action."""
    console.print("\n[bold]Review Actions:[/bold]")
    console.print("  [green]a[/green] - Approve")
    console.print("  [red]r[/red] - Reject (re-rank with your feedback)")
    console.print("  [blue]e[/blue] - Edit a specific field")
    console.print("  [yellow]g[/yellow] - Regenerate (re-score + re-rank)")

    action_map = {
        "a": "approve",
        "approve": "approve",
        "r": "reject",
        "reject": "reject",
        "e": "edit",
        "edit": "edit",
        "g": "regenerate",
        "regenerate": "regenerate",
    }

    while True:
        choice = typer.prompt("\nYour action [a/r/e/g]").strip().lower()
        if choice in action_map:
            return action_map[choice]
        console.print("[red]Invalid choice. Enter a, r, e, or g.[/red]")


def _print_provider_status(provider_info: dict):
    """Print LLM provider availability."""
    table = Table(title="LLM Providers", show_header=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="bold")

    for name, available in provider_info.items():
        status = "[green]✓ Available[/green]" if available else "[red]✗ Not available[/red]"
        table.add_row(name, status)

    console.print(table)

    if not any(provider_info.values()):
        console.print(
            "[bold red]No LLM provider available![/bold red]\n"
            "Run: ollama pull qwen3 && ollama serve\n"
            "Or set GROQ_API_KEY / GEMINI_API_KEY in .env"
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
