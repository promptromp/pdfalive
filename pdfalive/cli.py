"""CLI entrypoints."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import click
import pymupdf
from langchain.chat_models import init_chat_model
from langsmith import traceable
from rich.console import Console
from rich.table import Table

from pdfalive.config import load_config_as_default_map
from pdfalive.processors.ocr_detection import NoTextDetectionStrategy
from pdfalive.processors.ocr_processor import OCRProcessor
from pdfalive.processors.rename_processor import RenameProcessor
from pdfalive.processors.toc_generator import (
    DEFAULT_REQUEST_DELAY_SECONDS,
    TOCGenerator,
    apply_toc_to_document,
)


def _save_inplace(temp_file: str, target_file: str) -> None:
    """Replace target file with temp file contents, preserving original permissions.

    PyMuPDF cannot save directly to the same file it opened (requires incremental save
    which isn't always possible). This helper saves to a temp file first, then replaces
    the original.

    Args:
        temp_file: Path to the temporary file containing the new content.
        target_file: Path to the original file to replace.
    """
    # Preserve original file permissions
    original_stat = os.stat(target_file)
    shutil.move(temp_file, target_file)
    os.chmod(target_file, original_stat.st_mode)


console = Console()


def _load_config_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> None:
    """Eager callback to load config file and set default_map on the context.

    Args:
        ctx: Click context.
        param: The parameter that triggered this callback (--config).
        value: The config file path provided by the user, or None.
    """
    # Convert string path to Path if provided
    config_path = Path(value) if value else None

    try:
        default_map = load_config_as_default_map(config_path)
    except FileNotFoundError:
        raise click.BadParameter(f"Config file not found: {value}", param=param) from None
    except Exception as e:
        raise click.BadParameter(f"Error loading config file: {e}", param=param) from None

    if default_map:
        # Merge with existing default_map (if any), config takes lower precedence
        existing_map: dict[str, Any] = dict(ctx.default_map) if ctx.default_map else {}
        for cmd_name, cmd_defaults in default_map.items():
            if cmd_name in existing_map:
                # Existing defaults take precedence
                merged = dict(cmd_defaults)
                merged.update(existing_map[cmd_name])
                existing_map[cmd_name] = merged
            else:
                existing_map[cmd_name] = cmd_defaults
        ctx.default_map = existing_map


@click.group(context_settings=dict(show_default=True))
@click.option(
    "-c",
    "--config",
    type=click.Path(dir_okay=False),
    default=None,
    callback=_load_config_callback,
    is_eager=True,
    expose_value=False,
    help="Path to TOML config file. Auto-detected from pdfalive.toml if not specified.",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """pdfalive - Bring PDF files alive with the magic of LLMs."""
    ctx.ensure_object(dict)


@cli.command("generate-toc")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(), required=False)
@click.option("--model-identifier", type=str, default="gpt-5.2", help="LLM model to use.")
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
@click.option(
    "--inplace",
    is_flag=True,
    default=False,
    help="Modify the input file in place instead of creating a new output file.",
)
@traceable
def generate_toc(
    input_file: str,
    output_file: str | None,
    model_identifier: str,
    force: bool,
    show_token_usage: bool,
    request_delay: float,
    ocr: bool,
    ocr_language: str,
    ocr_dpi: int,
    ocr_output: bool,
    postprocess: bool,
    inplace: bool,
) -> None:
    """Generate a table of contents for a PDF file."""
    # Validate that either output_file is provided or --inplace is set
    if not inplace and not output_file:
        raise click.UsageError("Either OUTPUT_FILE must be provided or --inplace must be set.")
    if inplace and output_file:
        raise click.UsageError("Cannot specify both OUTPUT_FILE and --inplace.")

    # Determine the actual output path
    # For inplace mode, we use a temp file then replace the original (PyMuPDF limitation)
    if inplace:
        input_path = Path(input_file)
        temp_fd, temp_path = tempfile.mkstemp(suffix=input_path.suffix, dir=input_path.parent)
        os.close(temp_fd)
        actual_output_file = temp_path
        console.print(
            f"Generating TOC for [bold cyan]{input_file}[/bold cyan] [yellow](in place)[/yellow] "
            f"using model [bold magenta]{model_identifier}[/bold magenta]..."
        )
    else:
        assert output_file is not None  # Validated above
        actual_output_file = output_file
        console.print(
            f"Generating TOC for [bold cyan]{input_file}[/bold cyan] "
            f"using model [bold magenta]{model_identifier}[/bold magenta]..."
        )

    # Show active options summary
    options_parts = []
    if force:
        options_parts.append("force")
    if postprocess:
        options_parts.append("postprocess")
    if ocr:
        options_parts.append(f"ocr ({ocr_language}, {ocr_dpi} DPI)")
        if ocr_output:
            options_parts.append("ocr-output")
    else:
        options_parts.append("ocr disabled")
    console.print(f"  Options: [cyan]{', '.join(options_parts)}[/cyan]")

    try:
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

        usage = processor.run(
            output_file=actual_output_file, force=force, request_delay=request_delay, postprocess=postprocess
        )

        # If --ocr-output is not set and we performed OCR, apply TOC to original and save that instead
        if not ocr_output and performed_ocr and original_doc is not None:
            console.print("[cyan]Applying TOC to original document (discarding OCR text layer)...[/cyan]")
            toc = doc.get_toc()
            apply_toc_to_document(original_doc, toc, actual_output_file)
            original_doc.close()
            doc.close()
        else:
            if original_doc is not None:
                original_doc.close()

        # For inplace mode, replace the original file with the temp file
        if inplace:
            _save_inplace(actual_output_file, input_file)
            console.print(f"[bold green]All done.[/bold green] Modified [bold cyan]{input_file}[/bold cyan] in place.")
        else:
            console.print(
                f"[bold green]All done.[/bold green] Saved modified PDF to [bold cyan]{actual_output_file}[/bold cyan]."
            )

        if show_token_usage:
            usage.print_summary(console)
    except Exception:
        # Clean up temp file on failure to avoid leaking it in the input file's directory
        if inplace and os.path.exists(actual_output_file):
            os.unlink(actual_output_file)
        raise


@cli.command("extract-text")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(), required=False)
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
@click.option(
    "--inplace",
    is_flag=True,
    default=False,
    help="Modify the input file in place instead of creating a new output file.",
)
@traceable
def extract_text(
    input_file: str,
    output_file: str | None,
    ocr_language: str,
    ocr_dpi: int,
    force: bool,
    inplace: bool,
) -> None:
    """Extract text from a PDF using OCR and save to a new PDF with text layer."""
    # Validate that either output_file is provided or --inplace is set
    if not inplace and not output_file:
        raise click.UsageError("Either OUTPUT_FILE must be provided or --inplace must be set.")
    if inplace and output_file:
        raise click.UsageError("Cannot specify both OUTPUT_FILE and --inplace.")

    # Determine the actual output path
    # For inplace mode, we use a temp file then replace the original (PyMuPDF limitation)
    if inplace:
        input_path = Path(input_file)
        temp_fd, temp_path = tempfile.mkstemp(suffix=input_path.suffix, dir=input_path.parent)
        os.close(temp_fd)
        actual_output_file = temp_path
    else:
        assert output_file is not None  # Validated above
        actual_output_file = output_file

    if inplace:
        console.print(f"Processing [bold cyan]{input_file}[/bold cyan] [yellow](in place)[/yellow]...")
    else:
        console.print(f"Processing [bold cyan]{input_file}[/bold cyan]...")
    console.print(
        f"  Language: [cyan]{ocr_language}[/cyan], DPI: [cyan]{ocr_dpi}[/cyan], Force OCR: [cyan]{force}[/cyan]"
    )

    try:
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
            ocr_doc.save(actual_output_file)
            ocr_doc.close()

            # For inplace mode, replace the original file with the temp file
            if inplace:
                _save_inplace(actual_output_file, input_file)
                console.print(
                    f"[bold green]All Done.[/bold green] Modified [bold cyan]{input_file}[/bold cyan] in place."
                )
            else:
                console.print(
                    f"[bold green]All Done.[/bold green] Saved to [bold cyan]{actual_output_file}[/bold cyan]."
                )
        else:
            doc.close()
            # Clean up temp file if it was created but not used
            if inplace:
                os.unlink(actual_output_file)
            console.print("[green]Document already has sufficient extractable text. No OCR needed.[/green]")
            console.print("Use --force to process anyway.")
    except Exception:
        # Clean up temp file on failure to avoid leaking it in the input file's directory
        if inplace and os.path.exists(actual_output_file):
            os.unlink(actual_output_file)
        raise


@cli.command("rename")
@click.argument("input_files", type=click.Path(exists=True), nargs=-1, required=False)
@click.option(
    "-q",
    "--query",
    type=str,
    required=True,
    help="Renaming instruction query describing how to rename the files.",
)
@click.option(
    "-f",
    "--input-file",
    "input_file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Read input file paths from a text file (one path per line). Mutually exclusive with INPUT_FILES argument.",
)
@click.option("--model-identifier", type=str, default="gpt-5.2", help="LLM model to use.")
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    default=False,
    help="Automatically apply renames without asking for confirmation.",
)
@click.option("--show-token-usage", is_flag=True, default=True, help="Display token usage statistics.")
@traceable
def rename(
    input_files: tuple[str, ...],
    query: str,
    input_file: str | None,
    model_identifier: str,
    yes: bool,
    show_token_usage: bool,
) -> None:
    """Rename files using LLM-powered intelligent renaming.

    Provide one or more input files and a renaming instruction query.
    The LLM will suggest new names based on your instruction.

    Input files can be provided either as arguments or via --input-file (-f) option
    which reads paths from a text file (one per line). This is useful when dealing
    with many files or long filenames that would exceed command-line limits.

    Examples:

        pdfalive rename --query "Add 'REVIEWED_' prefix" *.pdf

        pdfalive rename -q "Rename to '[Author] - [Title] (Year).pdf'" paper1.pdf paper2.pdf

        # Read paths from a file (useful with find/xargs):
        find . -name "*.pdf" > files.txt
        pdfalive rename -q "Standardize filenames" -f files.txt
    """
    # Validate mutual exclusivity and resolve input files
    if input_files and input_file:
        raise click.UsageError("Cannot specify both INPUT_FILES arguments and --input-file option.")
    if not input_files and not input_file:
        raise click.UsageError("Either INPUT_FILES arguments or --input-file option must be provided.")

    # Resolve the final list of file paths
    if input_file:
        # Read paths from the input file
        input_file_path = Path(input_file)
        file_paths: list[str] = []
        with input_file_path.open() as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                file_path = Path(line)
                if not file_path.exists():
                    raise click.UsageError(f"File not found: '{line}' (line {line_num} in {input_file})")
                file_paths.append(line)
        if not file_paths:
            raise click.UsageError(f"No valid file paths found in {input_file}")
        resolved_input_files: tuple[str, ...] = tuple(file_paths)
    else:
        resolved_input_files = input_files

    console.print(
        f"Renaming [bold cyan]{len(resolved_input_files)}[/bold cyan] file(s) "
        f"using model [bold magenta]{model_identifier}[/bold magenta]..."
    )
    console.print(f"Query: [italic]{query}[/italic]")
    console.print()

    # Convert to Path objects
    paths = [Path(f) for f in resolved_input_files]

    # Initialize LLM and processor
    llm = init_chat_model(model=model_identifier)
    processor = RenameProcessor(llm=llm)

    # Generate rename suggestions
    console.print("[cyan]Generating rename suggestions...[/cyan]")
    try:
        result, usage = processor.generate_renames(paths, query)
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

    table = Table(
        show_header=True,
        header_style="bold",
        padding=(1, 1),  # Add vertical and horizontal padding for better readability
        show_lines=True,  # Add horizontal lines between rows
    )
    table.add_column("Original", style="cyan", overflow="fold")
    table.add_column("New Name", style="green", overflow="fold")
    table.add_column("Confidence", justify="right")
    table.add_column("Reasoning", style="dim", overflow="fold")

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
        if show_token_usage:
            usage.print_summary(console)
        return

    # Apply renames
    console.print("[cyan]Applying renames...[/cyan]")
    apply_result = processor.apply_renames(resolved)

    # Report any failures
    for error in apply_result.failed:
        console.print(f"[bold red]Error:[/bold red] {error.error}")

    # Report summary
    if apply_result.success_count > 0:
        console.print(f"[bold green]Successfully renamed {apply_result.success_count} file(s).[/bold green]")
    if apply_result.failure_count > 0:
        console.print(f"[bold yellow]Failed to rename {apply_result.failure_count} file(s).[/bold yellow]")

    if show_token_usage:
        usage.print_summary(console)
