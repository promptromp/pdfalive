"""CLI entrypoints."""

import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import click
import pymupdf
from langsmith import traceable
from rich.table import Table
from rich.text import Text

from pdfalive.config import load_config_as_default_map
from pdfalive.console import TextPart, create_console, safe_text, truncate_visible, visible
from pdfalive.evaluation.metrics import SUMMARY_PAGE_TOLERANCE, EvalReport, GoldenEntry
from pdfalive.evaluation.runner import EvalCase, EvalMode, discover_cases, run_eval_case
from pdfalive.llm import build_chat_model
from pdfalive.models.rename import PlannedRename, RejectedRename, RenameBatch, RenameInputs
from pdfalive.models.toc import TOCEntry
from pdfalive.processors.ocr_detection import NoTextDetectionStrategy
from pdfalive.processors.ocr_processor import OCRProcessor
from pdfalive.processors.rename_processor import RenameProcessor
from pdfalive.processors.toc_generator import (
    DEFAULT_REQUEST_DELAY_SECONDS,
    TOCGenerator,
    apply_toc_to_document,
)
from pdfalive.tokens import TokenUsage


# Rename preview table: reasoning shown wider than this many terminal cells is cut, ending in an ellipsis.
_RENAME_REASONING_PREVIEW_WIDTH = 50

# Exit status of a command that ran but did not fully succeed (a threshold missed, a rename
# failed), as opposed to Click's usage errors (2).
_EXIT_FAILURE = 1

# Rename preview table: confidence at or above these thresholds shows green / yellow, below them red.
_HIGH_CONFIDENCE_THRESHOLD = 0.9
_MEDIUM_CONFIDENCE_THRESHOLD = 0.7


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


console = create_console()


def _in_place_note(inplace: bool) -> tuple[TextPart, ...]:
    """The " (in place)" marker that follows the input path in startup lines, or nothing."""
    return (" ", ("(in place)", "yellow")) if inplace else ()


def _done_line(path: str, inplace: bool) -> Text:
    """The closing line of a PDF-writing command, naming the input modified in place or the output written."""
    if inplace:
        return safe_text(("All done.", "bold green"), " Modified ", (path, "bold cyan"), " in place.")
    return safe_text(("All done.", "bold green"), " Saved to ", (path, "bold cyan"), ".")


def _confidence_style(confidence: float) -> str:
    """Style of a rename's confidence cell: green when high, yellow when medium, red when low."""
    if confidence >= _HIGH_CONFIDENCE_THRESHOLD:
        return "green"
    if confidence >= _MEDIUM_CONFIDENCE_THRESHOLD:
        return "yellow"
    return "red"


def _rejected_file(rejection: RejectedRename, display_names: dict[Path, str]) -> str:
    """How a skipped suggestion's file is shown: its input's display name or, when its id names
    no file of its request, the id and the name the suggestion gave.
    """
    if rejection.source is not None:
        return display_names[rejection.source]
    return f'file id {rejection.op.file_id} ("{rejection.op.input_filename}")'


def _print_rejected_renames(rejected: list[RejectedRename], display_names: dict[Path, str]) -> None:
    """List the rename suggestions that will not be applied, each with the reason."""
    if not rejected:
        return
    console.print(f"Skipped {len(rejected)} suggestion(s):", style="yellow")
    for rejection in rejected:
        console.print(
            safe_text(
                "  - ",
                _rejected_file(rejection, display_names),
                " -> ",
                rejection.op.output_filename,
                ": ",
                (rejection.reason, "yellow"),
            )
        )


def _rename_preview_table(planned: list[PlannedRename], display_names: dict[Path, str]) -> Table:
    """The table of planned renames shown before confirmation, each input shown by its display name."""
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

    for rename in planned:
        reasoning = safe_text(truncate_visible(rename.op.reasoning, _RENAME_REASONING_PREVIEW_WIDTH))
        table.add_row(
            safe_text(display_names[rename.source]),
            safe_text(rename.target.name),
            safe_text((f"{rename.op.confidence:.0%}", _confidence_style(rename.op.confidence))),
            reasoning,
        )
    return table


def _review_and_apply_renames(
    processor: RenameProcessor, batches: list[RenameBatch], inputs: RenameInputs, yes: bool
) -> bool:
    """Preview the suggested renames, report the rejected ones, and apply the rest once confirmed.

    Returns:
        False if a confirmed rename failed, or, under --yes, if any suggestion was rejected,
        whether or not the others were applied: nobody reads the skipped list of an
        unattended run. True otherwise, including when nothing was suggested, every
        suggestion kept the current name, or the prompt was declined.
    """
    if not any(batch.operations for batch in batches):
        console.print("No rename operations suggested.", style="yellow")
        return True

    resolved = processor.resolve_renames(batches, inputs)
    # Inputs sharing a name are shown with their directories, so they can be told apart.
    display_names = inputs.display_names()
    _print_rejected_renames(resolved.rejected, display_names)
    rejected_unattended = yes and bool(resolved.rejected)
    if not resolved.planned:
        console.print("No valid rename operations to apply.", style="yellow")
        return not rejected_unattended

    console.print()
    console.print("Proposed renames:", style="bold")
    console.print(_rename_preview_table(resolved.planned, display_names))
    console.print()

    if not yes and not click.confirm("Apply these renames?", default=False):
        console.print("Aborted. No files were renamed.", style="yellow")
        return True

    console.print("Applying renames...", style="cyan")
    apply_result = processor.apply_renames(resolved.source_target_pairs)

    for error in apply_result.failed:
        console.print(safe_text(("Error:", "bold red"), " ", error.error))
    if apply_result.success_count > 0:
        console.print(f"Successfully renamed {apply_result.success_count} file(s).", style="bold green")
    if apply_result.failure_count > 0:
        console.print(f"Failed to rename {apply_result.failure_count} file(s).", style="bold yellow")
    return apply_result.all_succeeded and not rejected_unattended


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
        raise click.BadParameter(f"Config file not found: {visible(str(value))}", param=param) from None
    except Exception as e:
        raise click.BadParameter(f"Error loading config file: {visible(str(e), multiline=True)}", param=param) from None

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
@click.option("--model-identifier", type=str, default="gpt-5.6", help="LLM model to use.")
@click.option("--force", is_flag=True, default=False, help="Force overwrite existing TOC if present.")
@click.option("--show-token-usage/--no-show-token-usage", default=True, help="Display token usage statistics.")
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
@click.option(
    "--reasoning-effort",
    type=str,
    default=None,
    help="Reasoning effort level for models that support it (e.g. 'none', 'low', 'medium', 'high', 'xhigh'). "
    "Accepted levels vary by model; omitted by default, leaving the provider's own default.",
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
    reasoning_effort: str | None,
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
    else:
        assert output_file is not None  # Validated above
        actual_output_file = output_file
    console.print(
        safe_text(
            "Generating TOC for ",
            (input_file, "bold cyan"),
            *_in_place_note(inplace),
            " using model ",
            (model_identifier, "bold magenta"),
            "...",
        )
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
    console.print(safe_text("  Options: ", (", ".join(options_parts), "cyan")))

    try:
        doc = pymupdf.open(input_file)
        original_doc = None  # Keep reference to original if we need to discard OCR
        performed_ocr = False

        # Check if OCR is needed and perform it if enabled
        if ocr:
            console.print("Checking if document needs OCR...", style="cyan")
            ocr_processor = OCRProcessor(
                detection_strategy=NoTextDetectionStrategy(),
                language=ocr_language,
                dpi=ocr_dpi,
            )

            needs_ocr = ocr_processor.needs_ocr(doc)
            if needs_ocr:
                console.print("Insufficient text detected in PDF. Performing OCR to extract text...", style="yellow")

                # If --ocr-output is not set, keep the original document for final output
                if not ocr_output:
                    console.print("  OCR text used for TOC generation only (use --ocr-output to include)", style="dim")
                    original_doc = doc
                    doc = pymupdf.open(input_file)  # Reopen for OCR processing

                # process_in_memory returns a NEW document with OCR text layer
                ocr_doc = ocr_processor.process_in_memory(doc, show_progress=True)
                if ocr_output:
                    doc.close()
                doc = ocr_doc
                performed_ocr = True
                console.print("OCR completed.", style="green")

        llm = build_chat_model(model_identifier, reasoning_effort)
        processor = TOCGenerator(doc=doc, llm=llm)

        usage = processor.run(
            output_file=actual_output_file, force=force, request_delay=request_delay, postprocess=postprocess
        )

        # If --ocr-output is not set and we performed OCR, apply TOC to original and save that instead
        if not ocr_output and performed_ocr and original_doc is not None:
            console.print("Applying TOC to original document (discarding OCR text layer)...", style="cyan")
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
        console.print(_done_line(input_file if inplace else actual_output_file, inplace))

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

    console.print(safe_text("Processing ", (input_file, "bold cyan"), *_in_place_note(inplace), "..."))
    console.print(
        safe_text(
            "  Language: ",
            (ocr_language, "cyan"),
            ", DPI: ",
            (str(ocr_dpi), "cyan"),
            ", Force OCR: ",
            (str(force), "cyan"),
        )
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
                console.print("Insufficient text detected in PDF. Performing OCR...", style="yellow")
            else:
                console.print("Force OCR enabled. Performing OCR...", style="yellow")

            # process_in_memory returns a NEW document with OCR text layer
            ocr_doc = ocr_processor.process_in_memory(doc, show_progress=True)
            doc.close()
            console.print("OCR completed.", style="green")

            # Save the document with OCR text layer
            ocr_doc.save(actual_output_file)
            ocr_doc.close()

            # For inplace mode, replace the original file with the temp file
            if inplace:
                _save_inplace(actual_output_file, input_file)
            console.print(_done_line(input_file if inplace else actual_output_file, inplace))
        else:
            doc.close()
            # Clean up temp file if it was created but not used
            if inplace:
                os.unlink(actual_output_file)
            console.print("Document already has sufficient extractable text. No OCR needed.", style="green")
            console.print("Use --force to process anyway.")
    except Exception:
        # Clean up temp file on failure to avoid leaking it in the input file's directory
        if inplace and os.path.exists(actual_output_file):
            os.unlink(actual_output_file)
        raise


@cli.command("eval")
@click.option(
    "--evals-dir",
    type=click.Path(exists=True, file_okay=False),
    default="evals",
    help="Directory containing golden/ case files and cassettes/.",
)
@click.option(
    "--mode",
    type=click.Choice(["replay", "record", "live"]),
    default="replay",
    help="replay: recorded LLM responses (free); record: live LLM, writes cassette; live: live LLM only.",
)
@click.option(
    "--case",
    "case_names",
    multiple=True,
    help="Run only the named case(s). May be repeated. Defaults to all cases.",
)
@click.option("--model-identifier", type=str, default="gpt-5.6", help="LLM model for record/live modes.")
@click.option(
    "--loose-replay",
    is_flag=True,
    default=False,
    help="Replay responses by call order even if prompts have drifted since recording.",
)
@click.option(
    "--request-delay",
    type=float,
    default=None,
    help="Delay in seconds between LLM calls. Defaults to 0 for replay, pipeline default otherwise.",
)
@click.option(
    "--num-processes",
    type=int,
    default=None,
    help="Parallel processes for feature extraction. Defaults to CPU count - 1.",
)
@click.option("--min-f1", type=float, default=None, help="Exit with code 1 if any case's F1 falls below this.")
@click.option(
    "--min-page-accuracy",
    type=float,
    default=None,
    help="Exit with code 1 if any case's page accuracy (±1 page) falls below this.",
)
@click.option(
    "--reasoning-effort",
    type=str,
    default=None,
    help="Reasoning effort level for models that support it (e.g. 'none', 'low', 'medium', 'high', 'xhigh'). "
    "Accepted levels vary by model; omitted by default, leaving the provider's own default.",
)
@traceable
def eval_command(
    evals_dir: str,
    mode: str,
    case_names: tuple[str, ...],
    model_identifier: str,
    loose_replay: bool,
    request_delay: float | None,
    num_processes: int | None,
    min_f1: float | None,
    min_page_accuracy: float | None,
    reasoning_effort: str | None,
) -> None:
    """Evaluate TOC generation quality against golden (ground truth) data.

    Discovers cases from golden files under EVALS_DIR/golden/ and scores the
    pipeline's output against them. Use --mode record once (live LLM) to create
    a cassette, then --mode replay for free, deterministic evaluation runs.
    """
    cases = discover_cases(Path(evals_dir))

    if case_names:
        cases_by_name = {case.name: case for case in cases}
        unknown = [name for name in case_names if name not in cases_by_name]
        if unknown:
            available = ", ".join(sorted(cases_by_name)) or "(none)"
            raise click.UsageError(f"Unknown case(s): {visible(', '.join(unknown))}. Available: {visible(available)}")
        cases = [cases_by_name[name] for name in case_names]

    if not cases:
        raise click.UsageError(f"No evaluation cases found under {visible(evals_dir)}/golden/.")

    eval_mode = cast(EvalMode, mode)
    reports: list[tuple[EvalCase, EvalReport]] = []
    for case in cases:
        console.print(
            safe_text("Evaluating case ", (case.name, "bold cyan"), " (", (eval_mode, "magenta"), " mode)...")
        )
        report = run_eval_case(
            case,
            mode=eval_mode,
            model_identifier=model_identifier,
            strict_replay=not loose_replay,
            request_delay=request_delay,
            num_processes=num_processes,
            reasoning_effort=reasoning_effort,
        )
        reports.append((case, report))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Case", style="cyan")
    table.add_column("Golden", justify="right")
    table.add_column("Generated", justify="right")
    table.add_column("Matched", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Page =", justify="right")
    table.add_column("Page ±1", justify="right")
    table.add_column("Level", justify="right")

    for case, report in reports:
        summary = report.summary()
        table.add_row(
            safe_text(case.name),
            str(summary["golden_entries"]),
            str(summary["generated_entries"]),
            str(summary["matched"]),
            f"{summary['precision']:.2f}",
            f"{summary['recall']:.2f}",
            f"{summary['f1']:.2f}",
            f"{summary['page_accuracy_exact']:.2f}",
            f"{summary['page_accuracy_within_1']:.2f}",
            f"{summary['level_accuracy']:.2f}",
        )

    console.print()
    console.print(table)

    for case, report in reports:
        entry_blocks: tuple[tuple[str, Sequence[GoldenEntry | TOCEntry]], ...] = (
            ("missing golden", report.missing),
            ("spurious generated", report.spurious),
        )
        for kind, entries in entry_blocks:
            if not entries:
                continue
            console.print()
            console.print(safe_text((f"{case.name}: {len(entries)} {kind} entries:", "yellow")))
            for entry in entries:
                console.print(safe_text(f"  - {entry.title} (page {entry.page_number}, level {entry.level})"))

    failures = []
    for case, report in reports:
        if min_f1 is not None and report.f1 < min_f1:
            failures.append(f"{case.name}: F1 {report.f1:.2f} < {min_f1:.2f}")
        # Gate on the same tolerance the summary table's "Page ±N" column uses.
        page_accuracy = report.page_accuracy(tolerance=SUMMARY_PAGE_TOLERANCE)
        if min_page_accuracy is not None and page_accuracy < min_page_accuracy:
            failures.append(
                f"{case.name}: page accuracy (±{SUMMARY_PAGE_TOLERANCE}) {page_accuracy:.2f} < {min_page_accuracy:.2f}"
            )

    if failures:
        console.print()
        for failure in failures:
            console.print(safe_text(("Threshold failure:", "bold red"), " ", failure))
        raise SystemExit(_EXIT_FAILURE)


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
@click.option("--model-identifier", type=str, default="gpt-5.6", help="LLM model to use.")
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    default=False,
    help="Automatically apply renames without asking for confirmation.",
)
@click.option("--show-token-usage/--no-show-token-usage", default=True, help="Display token usage statistics.")
@click.option(
    "--reasoning-effort",
    type=str,
    default=None,
    help="Reasoning effort level for models that support it (e.g. 'none', 'low', 'medium', 'high', 'xhigh'). "
    "Accepted levels vary by model; omitted by default, leaving the provider's own default.",
)
@traceable
def rename(
    input_files: tuple[str, ...],
    query: str,
    input_file: str | None,
    model_identifier: str,
    yes: bool,
    show_token_usage: bool,
    reasoning_effort: str | None,
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
                    raise click.UsageError(
                        f"File not found: '{visible(line)}' (line {line_num} in {visible(input_file)})"
                    )
                file_paths.append(line)
        if not file_paths:
            raise click.UsageError(f"No valid file paths found in {visible(input_file)}")
        resolved_input_files: tuple[str, ...] = tuple(file_paths)
    else:
        resolved_input_files = input_files

    # Each file once, in input order: a file given more than once, in any spelling, is
    # listed (and counted) once, and files sharing a name in different directories are
    # separate inputs, told apart by their ids in each request.
    inputs = RenameInputs(Path(f) for f in resolved_input_files)

    console.print(
        safe_text(
            "Renaming ",
            (str(len(inputs)), "bold cyan"),
            " file(s) using model ",
            (model_identifier, "bold magenta"),
            "...",
        )
    )
    console.print(safe_text("Query: ", (query, "italic")))
    console.print()

    # Initialize LLM and processor
    llm = build_chat_model(model_identifier, reasoning_effort)
    processor = RenameProcessor(llm=llm)

    # Generate rename suggestions
    console.print("Generating rename suggestions...", style="cyan")
    # The processor records each LLM call here as it completes, so the calls are
    # reported even if a later batch, the prompt, or applying the renames fails.
    usage = TokenUsage()
    try:
        batches = processor.generate_renames(inputs, query, usage=usage)
        all_succeeded = _review_and_apply_renames(processor, batches, inputs, yes)
    finally:
        if show_token_usage and usage.llm_calls:
            usage.print_summary(console)
    if not all_succeeded:
        raise SystemExit(_EXIT_FAILURE)
