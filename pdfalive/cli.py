"""CLI entrypoints."""

from pathlib import Path

import click
import pymupdf
from langchain.chat_models import init_chat_model
from langsmith import traceable
from rich.console import Console
from rich.table import Table

from pdfalive.processors.ocr_detection import NoTextDetectionStrategy
from pdfalive.processors.ocr_processor import OCRProcessor
from pdfalive.processors.rename_processor import RenameProcessor
from pdfalive.processors.toc_generator import (
    DEFAULT_REQUEST_DELAY_SECONDS,
    TOCGenerator,
    apply_toc_to_document,
)


console = Console()


@click.group(context_settings=dict(show_default=True))
def cli() -> None:
    """pdfalive - Bring PDF files alive with the magic of LLMs."""
    pass


@cli.command("generate-toc")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--model-identifier", type=str, default="gpt-5.1", help="LLM model to use.")
@click.option("--force", is_flag=True, default=False, help="Force overwrite existing TOC if present.")
@click.option("--show-token-usage", is_flag=True, default=True, help="Display token usage statistics.")
@click.option(
    "--request-delay",
    type=float,
    default=DEFAULT_REQUEST_DELAY_SECONDS,
    help="Delay in seconds between LLM calls (for rate limiting).",
)
@click.option(
    "--ocr/--no-ocr",
    default=True,
    help="Enable/disable automatic OCR for scanned PDFs without text.",
)
@click.option(
    "--ocr-language",
    type=str,
    default="eng",
    help="Tesseract language code for OCR (e.g., 'eng', 'deu', 'fra').",
)
@click.option(
    "--ocr-dpi",
    type=int,
    default=300,
    help="DPI resolution for OCR processing.",
)
@click.option(
    "--ocr-output",
    is_flag=True,
    default=False,
    help="Include OCR text layer in output (makes PDF searchable).",
)
@click.option(
    "--postprocess/--no-postprocess",
    default=False,
    help="Enable/disable LLM postprocessing to clean up and improve the generated TOC.",
)
@traceable
def generate_toc(
    input_file: str,
    output_file: str,
    model_identifier: str,
    force: bool,
    show_token_usage: bool,
    request_delay: float,
    ocr: bool,
    ocr_language: str,
    ocr_dpi: int,
    ocr_output: bool,
    postprocess: bool,
) -> None:
    """Generate a table of contents for a PDF file."""
    console.print(
        f"Generating TOC for [bold cyan]{input_file}[/bold cyan] "
        f"using model [bold magenta]{model_identifier}[/bold magenta]..."
    )

    doc = pymupdf.open(input_file)
    original_doc = None  # Keep reference to original if we need to discard OCR
    performed_ocr = False

    # Check if OCR is needed and perform it if enabled
    if ocr:
        console.print("[cyan]Checking if document needs OCR...[/cyan]")
        ocr_processor = OCRProcessor(
            detection_strategy=NoTextDetectionStrategy(),
            language=ocr_language,
            dpi=ocr_dpi,
        )

        needs_ocr = ocr_processor.needs_ocr(doc)
        if needs_ocr:
            console.print("[yellow]Insufficient text detected in PDF. Performing OCR to extract text...[/yellow]")

            # If --ocr-output is not set, keep the original document for final output
            if not ocr_output:
                console.print("[dim]  OCR text used for TOC generation only (use --ocr-output to include)[/dim]")
                original_doc = doc
                doc = pymupdf.open(input_file)  # Reopen for OCR processing

            # process_in_memory returns a NEW document with OCR text layer
            ocr_doc = ocr_processor.process_in_memory(doc, show_progress=True)
            if ocr_output:
                doc.close()
            doc = ocr_doc
            performed_ocr = True
            console.print("[green]OCR completed.[/green]")

    llm = init_chat_model(model=model_identifier)
    processor = TOCGenerator(doc=doc, llm=llm)

    usage = processor.run(output_file=output_file, force=force, request_delay=request_delay, postprocess=postprocess)

    # If --ocr-output is not set and we performed OCR, apply TOC to original and save that instead
    if not ocr_output and performed_ocr and original_doc is not None:
        console.print("[cyan]Applying TOC to original document (discarding OCR text layer)...[/cyan]")
        toc = doc.get_toc()
        apply_toc_to_document(original_doc, toc, output_file)
        original_doc.close()
        doc.close()
    else:
        if original_doc is not None:
            original_doc.close()

    console.print(f"[bold green]All done.[/bold green] Saved modified PDF to [bold cyan]{output_file}[/bold cyan].")

    if show_token_usage:
        console.print()
        console.print("[bold]Token Usage:[/bold]")
        console.print(f"  LLM calls: [cyan]{usage.llm_calls}[/cyan]")
        console.print(f"  Input tokens: [cyan]{usage.input_tokens:,}[/cyan] (estimated)")
        console.print(f"  Output tokens: [cyan]{usage.output_tokens:,}[/cyan] (estimated)")
        console.print(f"  Total tokens: [cyan]{usage.total_tokens:,}[/cyan]")


@cli.command("extract-text")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--ocr-language",
    type=str,
    default="eng",
    help="Tesseract language code for OCR (e.g., 'eng', 'deu', 'fra').",
)
@click.option(
    "--ocr-dpi",
    type=int,
    default=300,
    help="DPI resolution for OCR processing.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force OCR even if document already has text.",
)
@traceable
def extract_text(
    input_file: str,
    output_file: str,
    ocr_language: str,
    ocr_dpi: int,
    force: bool,
) -> None:
    """Extract text from a PDF using OCR and save to a new PDF with text layer."""
    console.print(f"Processing [bold cyan]{input_file}[/bold cyan]...")
    console.print(
        f"  Language: [cyan]{ocr_language}[/cyan], DPI: [cyan]{ocr_dpi}[/cyan], Force OCR: [cyan]{force}[/cyan]"
    )

    doc = pymupdf.open(input_file)

    ocr_processor = OCRProcessor(
        detection_strategy=NoTextDetectionStrategy(),
        language=ocr_language,
        dpi=ocr_dpi,
    )

    needs_ocr = ocr_processor.needs_ocr(doc)
    console.print(f"  OCR detection: document {'needs' if needs_ocr else 'does not need'} OCR")

    if needs_ocr or force:
        if needs_ocr:
            console.print("[yellow]Insufficient text detected in PDF. Performing OCR...[/yellow]")
        else:
            console.print("[yellow]Force OCR enabled. Performing OCR...[/yellow]")

        # process_in_memory returns a NEW document with OCR text layer
        ocr_doc = ocr_processor.process_in_memory(doc, show_progress=True)
        doc.close()
        console.print("[green]OCR completed.[/green]")

        # Save the document with OCR text layer
        ocr_doc.save(output_file)
        ocr_doc.close()

        console.print(f"[bold green]Done.[/bold green] Saved to [bold cyan]{output_file}[/bold cyan].")
    else:
        doc.close()
        console.print("[green]Document already has sufficient extractable text. No OCR needed.[/green]")
        console.print("Use --force-ocr to process anyway.")


@cli.command("rename")
@click.argument("input_files", type=click.Path(exists=True), nargs=-1, required=True)
@click.option(
    "-q",
    "--query",
    type=str,
    required=True,
    help="Renaming instruction query describing how to rename the files.",
)
@click.option("--model-identifier", type=str, default="gpt-5.1", help="LLM model to use.")
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    default=False,
    help="Automatically apply renames without asking for confirmation.",
)
@traceable
def rename(
    input_files: tuple[str, ...],
    query: str,
    model_identifier: str,
    yes: bool,
) -> None:
    """Rename files using LLM-powered intelligent renaming.

    Provide one or more input files and a renaming instruction query.
    The LLM will suggest new names based on your instruction.

    Examples:

        pdfalive rename --query "Add 'REVIEWED_' prefix" *.pdf

        pdfalive rename -q "Rename to '[Author] - [Title] (Year).pdf'" paper1.pdf paper2.pdf
    """
    console.print(
        f"Renaming [bold cyan]{len(input_files)}[/bold cyan] file(s) "
        f"using model [bold magenta]{model_identifier}[/bold magenta]..."
    )
    console.print(f"Query: [italic]{query}[/italic]")
    console.print()

    # Convert to Path objects
    paths = [Path(f) for f in input_files]

    # Initialize LLM and processor
    llm = init_chat_model(model=model_identifier)
    processor = RenameProcessor(llm=llm)

    # Generate rename suggestions
    console.print("[cyan]Generating rename suggestions...[/cyan]")
    try:
        result = processor.generate_renames(paths, query)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1) from e

    if not result.operations:
        console.print("[yellow]No rename operations suggested.[/yellow]")
        return

    # Resolve full paths
    resolved = processor._resolve_full_paths(result.operations, paths)

    if not resolved:
        console.print("[yellow]No valid rename operations to apply.[/yellow]")
        return

    # Build operation lookup for display
    op_lookup = {op.input_filename: op for op in result.operations}

    # Display proposed renames in a table
    console.print()
    console.print("[bold]Proposed renames:[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Original", style="cyan")
    table.add_column("New Name", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Reasoning", style="dim")

    for source, target in resolved:
        op = op_lookup.get(source.name)
        confidence_str = f"{op.confidence:.0%}" if op else "N/A"
        reasoning = op.reasoning if op else ""

        # Color-code confidence
        if op and op.confidence >= 0.9:
            confidence_style = "green"
        elif op and op.confidence >= 0.7:
            confidence_style = "yellow"
        else:
            confidence_style = "red"

        table.add_row(
            source.name,
            target.name,
            f"[{confidence_style}]{confidence_str}[/{confidence_style}]",
            reasoning[:50] + "..." if len(reasoning) > 50 else reasoning,
        )

    console.print(table)
    console.print()

    # Ask for confirmation unless --yes is provided
    if not yes and not click.confirm("Apply these renames?", default=False):
        console.print("[yellow]Aborted. No files were renamed.[/yellow]")
        return

    # Apply renames
    console.print("[cyan]Applying renames...[/cyan]")
    try:
        processor.apply_renames(resolved)
    except (FileNotFoundError, FileExistsError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise SystemExit(1) from e

    console.print(f"[bold green]Successfully renamed {len(resolved)} file(s).[/bold green]")
