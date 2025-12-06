"""CLI entrypoints."""

import click
import pymupdf
from langchain.chat_models import init_chat_model
from rich.console import Console

from pdfalive.processors.toc_generator import DEFAULT_REQUEST_DELAY_SECONDS, TOCGenerator


console = Console()


@click.group(context_settings=dict(show_default=True))
def cli() -> None:
    """pdfalive - Bring PDF files alive with the magic of LLMs."""
    pass


@cli.command("generate-toc")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--model-identifier", type=str, default="claude-sonnet-4-5-20250929", help="LLM model to use.")
@click.option("--force", is_flag=True, default=False, help="Force overwrite existing TOC if present.")
@click.option("--show-token-usage", is_flag=True, default=True, help="Display token usage statistics.")
@click.option(
    "--request-delay",
    type=float,
    default=DEFAULT_REQUEST_DELAY_SECONDS,
    help="Delay in seconds between LLM calls (for rate limiting).",
)
def generate_toc(
    input_file: str,
    output_file: str,
    model_identifier: str,
    force: bool,
    show_token_usage: bool,
    request_delay: float,
) -> None:
    """Generate a table of contents for a PDF file."""
    console.print(
        f"Generating TOC for [bold cyan]{input_file}[/bold cyan] "
        f"using model [bold magenta]{model_identifier}[/bold magenta]..."
    )

    doc = pymupdf.open(input_file)
    llm = init_chat_model(model=model_identifier)
    processor = TOCGenerator(doc=doc, llm=llm)

    usage = processor.run(output_file=output_file, force=force, request_delay=request_delay)

    console.print(f"[bold green]All done.[/bold green] Saved modified PDF to [bold cyan]{output_file}[/bold cyan].")

    if show_token_usage:
        console.print()
        console.print("[bold]Token Usage:[/bold]")
        console.print(f"  LLM calls: [cyan]{usage.llm_calls}[/cyan]")
        console.print(f"  Input tokens: [cyan]{usage.input_tokens:,}[/cyan] (estimated)")
        console.print(f"  Output tokens: [cyan]{usage.output_tokens:,}[/cyan] (estimated)")
        console.print(f"  Total tokens: [cyan]{usage.total_tokens:,}[/cyan]")
